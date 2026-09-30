// Golden tests for the background service worker: the exact HTTP request built for each AI provider,
// how responses are parsed into answers, Laya, and the update check. Snapshots recorded from v2.5.0.
import { test, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { loadBackground, json, text, plain } from './helpers/extension.js';

const bg = await loadBackground();

const SLIDE = { action: 'AI_SOLVE_SLIDE', question: 'Thủ đô Việt Nam?', choices: ['Huế', 'Hà Nội', 'Đà Nẵng'] };
const openAiReply = (content) => json({ choices: [{ message: { content } }] });
const geminiReply = (t) => json({ candidates: [{ content: { parts: [{ text: t }] } }] });

function useProfile(profile, extra = {}) {
  bg.storage.reset({ aiProfiles: [{ id: 'p1', ...profile }], aiAssign: { slide: 'p1', exam: 'p1' }, ...extra });
}

beforeEach(() => {
  bg.net.reset();
  bg.storage.reset();
});

const PROVIDER_CASES = {
  gemini_default: [{ provider: 'gemini', apiKey: 'AIzaKEY', model: '' }, geminiReply('{"index": 1}')],
  gemini_prefixed_model: [{ provider: 'gemini', apiKey: 'AIzaKEY', model: 'gemini/gemini-2.5-pro' }, geminiReply('{"index": 1}')],
  gemini_custom_base: [
    { provider: 'gemini', apiKey: 'AIzaKEY', model: 'gemini-2.0-flash', endpoint: 'https://proxy.example.com/v1beta/' },
    geminiReply('{"index": 1}')
  ],
  openai: [{ provider: 'openai', apiKey: 'sk-test', model: 'gpt-4o' }, openAiReply('{"index": 1}')],
  openai_default_model: [{ provider: 'openai', apiKey: 'sk-test', model: '' }, openAiReply('{"index": 1}')],
  deepseek: [{ provider: 'deepseek', apiKey: 'sk-ds', model: '' }, openAiReply('{"index": 1}')],
  openrouter_headers: [{ provider: 'openrouter', apiKey: 'sk-or-v1-x', model: 'google/gemini-2.0-flash-001' }, openAiReply('{"index": 1}')],
  ollama_no_key: [{ provider: 'ollama', apiKey: '', model: '' }, openAiReply('{"index": 1}')],
  inception_reasoning: [
    { provider: 'inception', apiKey: 'ik', model: 'mercury-2.5', reasoningEffort: 'instant' },
    openAiReply('{"index": 1}')
  ],
  inception_invalid_reasoning_ignored: [
    { provider: 'inception', apiKey: 'ik', model: '', reasoningEffort: 'ultra' },
    openAiReply('{"index": 1}')
  ],
  reasoning_ignored_for_openai: [{ provider: 'openai', apiKey: 'sk', model: 'o3', reasoningEffort: 'high' }, openAiReply('{"index": 1}')],
  custom_default_endpoint: [{ provider: 'custom', apiKey: '', model: 'local-model', endpoint: '' }, openAiReply('{"index": 1}')],
  custom_full_path: [
    { provider: 'custom', apiKey: 'k', model: 'm', endpoint: 'https://api.example.com/v1/chat/completions' },
    openAiReply('{"index": 1}')
  ],
  custom_bare_host: [{ provider: 'custom', apiKey: 'k', model: 'm', endpoint: 'https://api.example.com' }, openAiReply('{"index": 1}')],
  custom_gemini_endpoint: [
    { provider: 'custom', apiKey: 'AIzaK', model: 'gemini-x', endpoint: 'https://generativelanguage.googleapis.com/v1beta' },
    geminiReply('{"index": 1}')
  ]
};

for (const [name, [profile, reply]] of Object.entries(PROVIDER_CASES)) {
  test(`AI request: ${name}`, async (t) => {
    useProfile(profile);
    bg.net.respond(reply);
    const res = await bg.dispatch(SLIDE);
    t.assert.snapshot({ request: bg.net.calls, response: plain(res) });
  });
}

test('AI request: legacy single-provider settings (before aiProfiles migration)', async (t) => {
  bg.storage.reset({ apiProvider: 'deepseek', apiKey: 'sk-legacy', apiModel: 'deepseek-reasoner', apiEndpoint: '' });
  bg.net.respond(openAiReply('{"index": 2}'));
  const res = await bg.dispatch(SLIDE);
  t.assert.snapshot({ request: bg.net.calls, response: plain(res) });
});

test('AI request: slide and exam use their assigned profiles', async () => {
  bg.storage.reset({
    aiProfiles: [
      { id: 'fast', provider: 'openai', apiKey: 'sk-a', model: 'fast-model' },
      { id: 'smart', provider: 'openai', apiKey: 'sk-b', model: 'smart-model' }
    ],
    aiAssign: { slide: 'fast', exam: 'smart' }
  });
  bg.net.respond(openAiReply('{"index": 0}'), openAiReply('[{"so_cau":1,"dap_an":"A"}]'));
  await bg.dispatch(SLIDE);
  await bg.dispatch({ action: 'AI_SOLVE_EXAM', promptText: 'PROMPT' });
  assert.deepEqual(
    bg.net.calls.map((c) => c.body.model),
    ['fast-model', 'smart-model']
  );
});

test('AI errors: missing key, no profiles, HTTP error message', async (t) => {
  const out = {};
  useProfile({ provider: 'openai', apiKey: '', model: '' });
  out.missingKey = plain(await bg.dispatch(SLIDE));

  bg.storage.reset({ aiProfiles: [], aiAssign: {} });
  out.noProfiles = plain(await bg.dispatch(SLIDE));

  useProfile({ provider: 'openai', apiKey: 'sk', model: '' });
  bg.net.respond(json({ error: { message: 'Invalid API key' } }, 401));
  out.openAi401 = plain(await bg.dispatch(SLIDE));

  useProfile({ provider: 'gemini', apiKey: 'AIza', model: '' });
  bg.net.respond(json({ error: { message: 'Quota exceeded' } }, 429));
  out.gemini429 = plain(await bg.dispatch(SLIDE));

  t.assert.snapshot(out);
});

test('AI request: retries once without reasoning_effort when the server rejects it', async (t) => {
  useProfile({ provider: 'inception', apiKey: 'ik', model: 'mercury-2', reasoningEffort: 'high' });
  bg.net.respond(json({ error: { message: 'Unknown parameter: reasoning_effort' } }, 400), openAiReply('{"index": 2}'));
  const res = await bg.dispatch(SLIDE);
  t.assert.snapshot({ bodies: bg.net.calls.map((c) => c.body), response: plain(res) });
});

const SLIDE_REPLIES = {
  fenced: openAiReply('```json\n{"index": 2}\n```'),
  string_index: openAiReply('{"index": "0"}'),
  digit_fallback: openAiReply('Đáp án đúng là 1 vì ...'),
  text_match: openAiReply('Hà Nội'),
  reasoning_content_only: json({ choices: [{ message: { content: '', reasoning_content: '{"index": 2}' } }] }),
  sse_stream: text(
    'data: {"choices":[{"delta":{"content":"{\\"ind"}}]}\n\ndata: {"choices":[{"delta":{"content":"ex\\": 1}"}}]}\n\ndata: [DONE]\n'
  ),
  out_of_range: openAiReply('{"index": 7}'),
  unparseable: openAiReply('Tôi không biết')
};

for (const [name, reply] of Object.entries(SLIDE_REPLIES)) {
  test(`slide answer parsing: ${name}`, async (t) => {
    useProfile({ provider: 'openai', apiKey: 'sk', model: 'gpt-4o-mini' });
    bg.net.respond(reply);
    t.assert.snapshot(plain(await bg.dispatch(SLIDE)));
  });
}

test('slide: rejects missing question/choices without calling the API', async (t) => {
  useProfile({ provider: 'openai', apiKey: 'sk', model: '' });
  const res = await bg.dispatch({ action: 'AI_SOLVE_SLIDE', question: '', choices: [] });
  assert.equal(bg.net.calls.length, 0);
  t.assert.snapshot(plain(res));
});

test('exam: request and fenced answer cleanup', async (t) => {
  useProfile({ provider: 'openai', apiKey: 'sk', model: 'gpt-4o' });
  bg.net.respond(openAiReply('```json\n[{"so_cau": 1, "dap_an": "A"}]\n```'));
  const res = await bg.dispatch({ action: 'AI_SOLVE_EXAM', promptText: 'ĐỀ BÀI' });
  t.assert.snapshot({ request: bg.net.calls, response: plain(res) });
});

test('laya: health and ranked solve', async (t) => {
  bg.storage.reset({ layaEndpoint: 'http://127.0.0.1:9000/', layaApiKey: 'secret' });
  bg.net.respond(
    json({ status: 'ok', loaded: ['multilingual'], device: 'cpu' }),
    json(
      {
        answers: { answer: { probabilities: { Huế: 0.1, 'Hà Nội': 0.7, 'Hà Nội (3)': 0.2 } } },
        routing: { model: 'multilingual' }
      },
      200,
      { 'X-Inference-Time-Ms': '42' }
    )
  );
  const health = await bg.dispatch({ action: 'LAYA_HEALTH' });
  const solve = await bg.dispatch({ action: 'LAYA_SOLVE_SLIDE', question: 'Q?', choices: ['Huế', 'Hà Nội', 'Hà Nội'] });
  t.assert.snapshot({ request: bg.net.calls, health: plain(health), solve: plain(solve) });
});

test('laya: connection failure message', async (t) => {
  bg.net.respond(() => {
    throw new TypeError('fetch failed');
  });
  t.assert.snapshot(plain(await bg.dispatch({ action: 'LAYA_HEALTH' })));
});

test('update check: parses the latest tag and caches it', async () => {
  bg.net.respond(() => ({ url: 'https://github.com/k4spi4n/quizz-slayers/releases/tag/v99.1.0' }));
  const fresh = await bg.dispatch({ action: 'CHECK_UPDATE', force: true });
  assert.equal(fresh.success, true);
  assert.equal(fresh.latest, '99.1.0');
  assert.equal(fresh.hasUpdate, true);
  assert.match(bg.net.calls[0].url, /\/releases\/latest$/);
  assert.equal(bg.net.calls[0].method, 'HEAD');

  const cached = await bg.dispatch({ action: 'CHECK_UPDATE' });
  assert.equal(bg.net.calls.length, 1, 'second check within 6h must use the cache');
  assert.equal(cached.latest, '99.1.0');
});

test('update check: unparseable redirect is an error', async () => {
  bg.net.respond(() => ({ url: 'https://github.com/login' }));
  const res = await bg.dispatch({ action: 'CHECK_UPDATE', force: true });
  assert.equal(res.success, false);
});

test('compareVersions', () => {
  const c = bg.compareVersions;
  assert.equal(c('2.5.0', '2.4.0'), 1);
  assert.equal(c('2.4.0', '2.4.0'), 0);
  assert.equal(c('2.4.10', '2.4.9'), 1);
  assert.equal(c('2.4', '2.4.1'), -1);
  assert.equal(c('3', '2.99.99'), 1);
});
