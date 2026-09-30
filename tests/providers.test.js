// Unit tests for shared/providers.js helpers used by both the popup and the background.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  PROVIDERS,
  displayModel,
  isProfileReady,
  validReasoningEffort,
  reasoningEffortsFor,
} from '../EDUX-EXTENSION/shared/providers.js';

test('every provider has the fields the settings form needs', () => {
  for (const [id, p] of Object.entries(PROVIDERS)) {
    for (const field of ['name', 'placeholderEndpoint', 'keyPlaceholder']) {
      assert.ok(p[field], `${id}.${field}`);
    }
    assert.equal(typeof p.endpoint, 'string', `${id}.endpoint`);
    assert.ok(Array.isArray(p.models), `${id}.models`);
  }
});

test('displayModel matches the model actually sent when a profile leaves it empty', () => {
  assert.equal(displayModel('my-model', 'openai'), 'my-model');
  assert.equal(displayModel('', 'gemini'), 'gemini-2.0-flash');
  assert.equal(displayModel('', 'openai'), 'gpt-4o-mini');
  assert.equal(displayModel('', 'deepseek'), 'deepseek-chat');
  assert.equal(displayModel('', 'openrouter'), 'gpt-4o-mini');
  assert.equal(displayModel('', 'inception'), 'mercury-2.5');
  assert.equal(displayModel('', 'ollama'), 'llama3.2');
  assert.equal(displayModel('', 'custom'), 'gpt-4o-mini');
  assert.equal(displayModel('', 'unknown'), 'gemini-2.0-flash');
});

test('isProfileReady: key, Ollama, or a local endpoint', () => {
  assert.equal(isProfileReady(null), false);
  assert.equal(isProfileReady({ provider: 'openai', apiKey: '' }), false);
  assert.equal(isProfileReady({ provider: 'openai', apiKey: 'sk' }), true);
  assert.equal(isProfileReady({ provider: 'ollama', apiKey: '' }), true);
  assert.equal(
    isProfileReady({ provider: 'custom', apiKey: '', endpoint: 'http://127.0.0.1:1234/v1' }),
    true,
  );
});

test('reasoning effort only for providers that support it', () => {
  assert.deepEqual(reasoningEffortsFor('inception'), ['instant', 'low', 'medium', 'high']);
  assert.deepEqual(reasoningEffortsFor('openai'), []);
  assert.equal(validReasoningEffort('inception', 'high'), 'high');
  assert.equal(validReasoningEffort('inception', 'ultra'), '');
  assert.equal(validReasoningEffort('openai', 'high'), '');
});
