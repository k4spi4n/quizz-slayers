// Loads extension code into Node for tests, with just enough of the browser/chrome API stubbed.
// The helpers expose a stable interface so tests don't change when the files behind them move.
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath, pathToFileURL } from 'node:url';

// EDUX_EXT_DIR lets the smoke test run against another checkout (e.g. to compare with an older commit)
export const EXT_DIR = process.env.EDUX_EXT_DIR
  ? path.resolve(process.env.EDUX_EXT_DIR)
  : path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../EDUX-EXTENSION');

const read = (rel) => fs.readFileSync(path.join(EXT_DIR, rel), 'utf8');

function createStorage(initial = {}) {
  let data = structuredClone(initial);
  const pick = (keys) => {
    if (keys == null) return structuredClone(data);
    const list = typeof keys === 'string' ? [keys] : Array.isArray(keys) ? keys : Object.keys(keys);
    const out = {};
    for (const k of list) if (k in data) out[k] = structuredClone(data[k]);
    return out;
  };
  const withCallback = (result, cb) => {
    if (typeof cb === 'function') cb(result);
    return Promise.resolve(result);
  };
  return {
    reset(next = {}) {
      data = structuredClone(next);
    },
    dump: () => structuredClone(data),
    api: {
      get: (keys, cb) => withCallback(pick(keys), cb),
      set: (items, cb) => {
        Object.assign(data, structuredClone(items));
        return withCallback(undefined, cb);
      },
      remove: (keys, cb) => {
        for (const k of [].concat(keys)) delete data[k];
        return withCallback(undefined, cb);
      }
    }
  };
}

function createChrome(storage) {
  const listeners = {};
  const event = (name) => ({ addListener: (fn) => (listeners[name] = fn) });
  return {
    listeners,
    chrome: {
      runtime: {
        onInstalled: event('onInstalled'),
        onStartup: event('onStartup'),
        onMessage: event('onMessage'),
        getManifest: () => JSON.parse(read('manifest.json')),
        getURL: (p) => `chrome-extension://test/${p}`,
        sendMessage: () => Promise.resolve()
      },
      tabs: { onUpdated: event('onUpdated') },
      scripting: { executeScript: () => Promise.resolve() },
      action: { setBadgeText: () => {}, setBadgeBackgroundColor: () => {} },
      storage: { local: storage.api }
    }
  };
}

// Programmable fetch: queue responses, inspect the requests that were made
function createFetch() {
  const calls = [];
  let responders = [];
  const fetch = async (url, init = {}) => {
    const call = {
      url: String(url),
      method: init.method || 'GET',
      headers: { ...(init.headers || {}) },
      body: init.body ? JSON.parse(init.body) : undefined
    };
    calls.push(call);
    const next = responders.shift();
    if (!next) throw new Error(`Unexpected fetch: ${call.url}`);
    return next(call);
  };
  return {
    fetch,
    calls,
    reset() {
      calls.length = 0;
      responders = [];
    },
    respond(...fns) {
      responders.push(...fns);
    }
  };
}

export const json = (obj, status = 200, headers = {}) => () =>
  new Response(JSON.stringify(obj), { status, headers: { 'Content-Type': 'application/json', ...headers } });
export const text = (body, status = 200) => () => new Response(body, { status });

/**
 * Background service worker harness.
 * dispatch(message) resolves with whatever the onMessage handler passes to sendResponse.
 */
export async function loadBackground() {
  const storage = createStorage();
  const net = createFetch();
  const { chrome, listeners } = createChrome(storage);

  // The service worker is an ES module that reads `chrome` / `fetch` as globals
  globalThis.chrome = chrome;
  globalThis.fetch = net.fetch;
  await import(pathToFileURL(path.join(EXT_DIR, 'background/index.js')).href);
  const { compareVersions } = await import(pathToFileURL(path.join(EXT_DIR, 'shared/version.js')).href);

  return {
    storage,
    net,
    compareVersions,
    dispatch(message) {
      return new Promise((resolve) => {
        const keepOpen = listeners.onMessage(message, {}, resolve);
        if (keepOpen !== true) resolve(undefined);
      });
    }
  };
}

/**
 * Exam (Bài tập) text-processing API used by the content scripts: answer parsing and prompt building.
 */
export function loadExamApi({ title = 'Bài tập EDUX' } = {}) {
  const storage = createStorage();
  const { chrome } = createChrome(storage);
  const ctx = { console, chrome, document: { title }, setTimeout, clearTimeout };
  ctx.window = ctx;
  vm.createContext(ctx);

  for (const file of ['dom-utils.js', 'answer-parser.js', 'exam-dom.js', 'exam-prompt.js', 'exam-solver.js']) {
    vm.runInContext(read(`content/${file}`), ctx, { filename: file });
  }

  const parser = ctx.window.EduxAnswerParser;
  const prompt = ctx.window.EduxExamPrompt;
  return {
    loadAnswersFromInput: parser.loadAnswersFromInput,
    sanitizeAiResponse: parser.sanitizeAiResponse,
    parseTrueFalseAnswers: parser.parseTrueFalseAnswers,
    normalizeAnswersPayload: parser.normalizeAnswersPayload,
    // The exam solver passes document.title as the fallback title
    buildCompactPromptPayload: (payload) => prompt.buildCompactPromptPayload(payload, ctx.document.title),
    generateStandardPromptText: prompt.generateStandardPromptText,
    solver: ctx.window.EduxTestSolver
  };
}

// vm objects come from another realm; round-trip through JSON so snapshots/deepEqual compare plain data
export const plain = (v) => (v === undefined ? v : JSON.parse(JSON.stringify(v)));
