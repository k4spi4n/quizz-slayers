// Guards the wiring between manifest.json, shared lists and the content-script namespaces.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { EXT_DIR, loadExamApi } from './helpers/extension.js';
import { CONTENT_SCRIPTS, INJECTED_SCRIPT } from '../EDUX-EXTENSION/shared/content-scripts.js';

const manifest = JSON.parse(fs.readFileSync(path.join(EXT_DIR, 'manifest.json'), 'utf8'));

test('manifest content scripts match shared/content-scripts.js (popup re-injects this list)', () => {
  const isolated = manifest.content_scripts.find((cs) => !cs.world);
  assert.deepEqual(isolated.js, CONTENT_SCRIPTS);
});

test('injected script path is the same everywhere', () => {
  const main = manifest.content_scripts.find((cs) => cs.world === 'MAIN');
  assert.deepEqual(main.js, [INJECTED_SCRIPT]);
  assert.deepEqual(manifest.web_accessible_resources[0].resources, [INJECTED_SCRIPT]);
});

test('every file the manifest references exists', () => {
  const files = [
    manifest.background.service_worker,
    manifest.action.default_popup,
    ...Object.values(manifest.icons),
    ...manifest.content_scripts.flatMap((cs) => [...(cs.js || []), ...(cs.css || [])]),
    ...manifest.web_accessible_resources.flatMap((r) => r.resources),
  ];
  for (const f of files) assert.ok(fs.existsSync(path.join(EXT_DIR, f)), `missing ${f}`);
});

test('EduxTestSolver keeps the API content.js calls', () => {
  const { solver } = loadExamApi();
  for (const fn of [
    'startExercise',
    'fillTestAnswers',
    'extractQuestions',
    'setCapturedExamData',
    'getCapturedExamData',
  ]) {
    assert.equal(typeof solver[fn], 'function', fn);
  }
});

test('prompt payload falls back to the page title', () => {
  const exam = loadExamApi({ title: 'Bài kiểm tra số 3 - EDUX' });
  assert.equal(
    exam.buildCompactPromptPayload({ data: { exam_data: {} } }).title,
    'Bài kiểm tra số 3 - EDUX',
  );
  assert.equal(
    exam.buildCompactPromptPayload({ data: { title: 'Từ API', exam_data: {} } }).title,
    'Từ API',
  );
});

test('provider dropdown in popup.html matches shared/providers.js', async () => {
  const { PROVIDERS } = await import('../EDUX-EXTENSION/shared/providers.js');
  const html = fs.readFileSync(path.join(EXT_DIR, 'popup/popup.html'), 'utf8');
  const select = html.match(/<select id="settingApiProvider"[\s\S]*?<\/select>/)[0];
  const options = [...select.matchAll(/<option value="([^"]+)"/g)].map((m) => m[1]);
  assert.deepEqual(options.sort(), Object.keys(PROVIDERS).sort());
});
