// Smoke test: load the unpacked extension in Chromium and exercise the popup end to end.
// Run with `npm run smoke` (first time: `npx playwright install chromium`).
import { test as base, expect, chromium } from '@playwright/test';
import fs from 'node:fs';
import { EXT_DIR } from '../helpers/extension.js';

const test = base.extend({
  // One browser per test so extension storage starts empty
  context: async ({}, use) => {
    const context = await chromium.launchPersistentContext('', {
      channel: 'chromium',
      headless: true,
      args: [`--disable-extensions-except=${EXT_DIR}`, `--load-extension=${EXT_DIR}`]
    });
    await use(context);
    await context.close();
  },
  worker: async ({ context }, use) => {
    const sw = context.serviceWorkers()[0] || (await context.waitForEvent('serviceworker'));
    await use(sw);
  },
  popupUrl: async ({ worker }, use) => {
    const extensionOrigin = worker.url().match(/^chrome-extension:\/\/[^/]+/)[0];
    await use(`${extensionOrigin}/popup/popup.html`);
  },
  // Opens the popup page and records uncaught errors / console.error output
  openPopup: async ({ context, popupUrl }, use) => {
    const errors = [];
    await use(async () => {
      const page = await context.newPage();
      page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
      page.on('console', (m) => m.type() === 'error' && errors.push(`console: ${m.text()}`));
      await page.goto(popupUrl);
      await expect(page.locator('#appVersion')).toContainText('v');
      return page;
    });
    expect(errors, 'popup must not log errors').toEqual([]);
  }
});

const storage = {
  get: (page, keys) => page.evaluate((k) => chrome.storage.local.get(k), keys),
  set: (page, items) => page.evaluate((i) => chrome.storage.local.set(i), items),
  clear: (page) => page.evaluate(() => chrome.storage.local.clear())
};

test('service worker boots with the manifest version', async ({ worker }) => {
  const version = await worker.evaluate(() => chrome.runtime.getManifest().version);
  const manifest = JSON.parse(fs.readFileSync(`${EXT_DIR}/manifest.json`, 'utf8'));
  expect(version).toBe(manifest.version);
});

test('popup opens and every tab switches', async ({ openPopup }) => {
  const page = await openPopup();
  const manifest = JSON.parse(fs.readFileSync(`${EXT_DIR}/manifest.json`, 'utf8'));
  await expect(page.locator('#appVersion')).toHaveText(`v${manifest.version} • Manifest V3`);

  for (const tab of ['tab-test', 'tab-scores', 'tab-settings', 'tab-slide']) {
    await page.click(`.tab-btn[data-tab="${tab}"]`);
    await expect(page.locator(`#${tab}`)).toHaveClass(/active/);
  }
});

test('AI profile saved in the editor survives a reload', async ({ openPopup }) => {
  const page = await openPopup();
  await page.click('.tab-btn[data-tab="tab-settings"]');
  await page.click('#btnAddAiProfile');
  await page.selectOption('#settingApiProvider', 'openai');
  await page.fill('#settingApiKey', 'sk-smoke-test');
  await page.click('#btnSaveAiProfile');

  const { aiProfiles, aiAssign } = await storage.get(page, ['aiProfiles', 'aiAssign']);
  expect(aiProfiles).toHaveLength(1);
  expect(aiProfiles[0]).toMatchObject({ provider: 'openai', apiKey: 'sk-smoke-test' });
  expect(aiAssign).toEqual({ slide: aiProfiles[0].id, exam: aiProfiles[0].id });

  await page.reload();
  await expect(page.locator('#aiProfileList .ai-profile-row')).toHaveCount(1);
  await expect(page.locator('#assignExam')).toHaveValue(aiProfiles[0].id);
});

test('legacy single-provider settings are migrated to a profile', async ({ openPopup }) => {
  const page = await openPopup();
  await storage.clear(page);
  await storage.set(page, { apiProvider: 'deepseek', apiKey: 'sk-old', apiModel: 'deepseek-chat', apiEndpoint: '' });
  await page.reload();
  await expect(page.locator('#aiProfileList .ai-profile-row')).toHaveCount(1);

  const s = await storage.get(page, null);
  expect(s.aiProfiles).toEqual([{ id: 'p1', provider: 'deepseek', endpoint: '', apiKey: 'sk-old', model: 'deepseek-chat' }]);
  expect(s.aiAssign).toEqual({ slide: 'p1', exam: 'p1' });
  expect(s.apiKey).toBeUndefined();
});

test('settings backup exports and restores', async ({ openPopup }, testInfo) => {
  const page = await openPopup();
  const profiles = [{ id: 'pX', provider: 'gemini', endpoint: '', apiKey: 'AIza-backup', model: 'gemini-2.0-flash' }];
  await storage.set(page, { aiProfiles: profiles, aiAssign: { slide: 'pX', exam: 'pX' }, delayMs: 250, savedAnswers: 'tmp' });
  await page.reload();
  await page.click('.tab-btn[data-tab="tab-settings"]');

  const [download] = await Promise.all([page.waitForEvent('download'), page.click('#btnExportSettings')]);
  const file = testInfo.outputPath('backup.json');
  await download.saveAs(file);
  const backup = JSON.parse(fs.readFileSync(file, 'utf8'));
  expect(backup.app).toBe('edux-slayers');
  expect(backup.settings.aiProfiles).toEqual(profiles);
  expect(backup.settings.savedAnswers).toBeUndefined();

  await storage.clear(page);
  await page.setInputFiles('#importSettingsFile', file);
  await page.waitForEvent('load');
  const restored = await storage.get(page, ['aiProfiles', 'aiAssign', 'delayMs']);
  expect(restored).toEqual({ aiProfiles: profiles, aiAssign: { slide: 'pX', exam: 'pX' }, delayMs: 250 });
});

test('update check answers from the service worker', async ({ openPopup }) => {
  const page = await openPopup();
  const res = await page.evaluate(() => chrome.runtime.sendMessage({ action: 'CHECK_UPDATE', force: true }));
  // Network may be offline in CI: accept a clean failure, but the handler must answer
  expect(typeof res.success).toBe('boolean');
  if (res.success) expect(res.latest).toMatch(/^\d+(\.\d+)*$/);
});

test('exercise AI requests are answered by the service worker', async ({ openPopup }) => {
  const page = await openPopup();
  await storage.set(page, { aiProfiles: [], aiAssign: {} });
  const res = await page.evaluate(() => chrome.runtime.sendMessage({ action: 'AI_SOLVE_EXAM', promptText: 'x' }));
  expect(res).toEqual({ success: false, message: 'Chưa có cấu hình AI. Vào tab Cài đặt → Cấu hình AI để thêm.' });
});

// A minimal exam page served at an EDUX URL, so the manifest's content scripts are injected into it
const EXAM_PAGE = `<!doctype html><html><head><meta charset="utf-8"><title>Bài tập smoke</title></head><body>
  <div class="bg-white">
    <span>Câu 1</span>
    <div class="prose"><p>Thủ đô Việt Nam là?</p></div>
    <div class="border rounded-lg cursor-pointer"><span class="flex-shrink-0">A</span><p>Hà Nội</p></div>
    <div class="border rounded-lg cursor-pointer"><span class="flex-shrink-0">B</span><p>Huế</p></div>
  </div>
  <div class="bg-white">
    <span>Câu 2</span>
    <div class="prose"><p>Nước sôi ở bao nhiêu độ C?</p></div>
    <input type="text">
  </div>
</body></html>`;

test('content scripts load on EDUX pages and extract questions', async ({ context, openPopup }) => {
  await context.route('https://edux.cmcu.edu.vn/**', (route) =>
    route.fulfill({ status: 200, contentType: 'text/html; charset=utf-8', body: EXAM_PAGE })
  );
  const edux = await context.newPage();
  const pageErrors = [];
  edux.on('pageerror', (e) => pageErrors.push(e.message));
  await edux.goto('https://edux.cmcu.edu.vn/smoke-exam');

  // MAIN-world network interceptor (content/injected.js)
  await expect.poll(() => edux.evaluate(() => window.__EDUX_SLAYERS_INTERCEPTOR_ACTIVE__)).toBe(true);

  // Isolated-world scripts: content.js answers only if every script before it loaded
  const popup = await openPopup();
  const extractFromEduxTab = () =>
    popup.evaluate(async () => {
      const [tab] = await chrome.tabs.query({ url: 'https://edux.cmcu.edu.vn/*' });
      return chrome.tabs.sendMessage(tab.id, { action: 'EXTRACT_QUESTIONS' }).catch((e) => ({ error: e.message }));
    });
  await expect.poll(async () => typeof (await extractFromEduxTab()).promptText).toBe('string');

  const extracted = await extractFromEduxTab();
  expect(extracted.promptText).toContain('Thủ đô Việt Nam là?');
  expect(extracted.promptText).toContain('Hà Nội');
  expect(extracted.promptText).toContain('Nước sôi ở bao nhiêu độ C?');
  expect(pageErrors).toEqual([]);
});

// Local OpenAI-compatible server standing in for the AI provider (a "custom" profile on localhost needs no key)
async function startFakeAi(answer) {
  const { createServer } = await import('node:http');
  const requests = [];
  const server = createServer((req, res) => {
    let body = '';
    req.on('data', (c) => (body += c));
    req.on('end', () => {
      requests.push({ url: req.url, body: JSON.parse(body || '{}') });
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ choices: [{ message: { content: answer } }] }));
    });
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  return { requests, endpoint: `http://127.0.0.1:${server.address().port}/v1`, close: () => server.close() };
}

test('exercise API mode: extract → AI (via service worker) → answers shown and filled', async ({ context, openPopup }) => {
  const answer = '[{"so_cau": 1, "dap_an": "A"}, {"so_cau": 2, "dap_an": "100"}]';
  const ai = await startFakeAi(answer);
  try {
    await context.route('https://edux.cmcu.edu.vn/**', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'text/html; charset=utf-8',
        body: EXAM_PAGE.replace('<body>', '<body><div role="dialog" data-state="open">').replace('</body>', '</div></body>')
      })
    );
    const edux = await context.newPage();
    await edux.goto('https://edux.cmcu.edu.vn/smoke-exam');

    const popup = await openPopup();
    await storage.set(popup, {
      aiProfiles: [{ id: 'local', provider: 'custom', endpoint: ai.endpoint, apiKey: '', model: 'fake-model' }],
      aiAssign: { slide: 'local', exam: 'local' },
      autoSubmit: false
    });
    await popup.reload();
    await popup.click('.tab-btn[data-tab="tab-test"]');
    await popup.click('#btnSolveAI');

    await expect(popup.locator('#testLog')).toContainText('AI đã giải xong', { timeout: 20_000 });
    await expect(popup.locator('#autoAnswersBox')).toHaveValue(answer);
    expect((await storage.get(popup, 'savedAnswers')).savedAnswers).toBe(answer);

    expect(ai.requests).toHaveLength(1);
    const { url, body } = ai.requests[0];
    expect(url).toBe('/v1/chat/completions');
    expect(body.model).toBe('fake-model');
    expect(body.messages[0].content).toContain('"so_cau" và "dap_an"');
    expect(body.messages[1].content).toContain('Thủ đô Việt Nam là?');

    // Fill step reached the page and wrote an answer. (Which answer lands where depends on EDUX's
    // one-question-per-view dialog, which this static page doesn't reproduce; v2.5.0 behaves the same.)
    await expect(edux.locator('input[type="text"]')).not.toHaveValue('', { timeout: 15_000 });
  } finally {
    ai.close();
  }
});
