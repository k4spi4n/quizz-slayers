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
