// Per-question wait while auto-filling an exercise (content/exam-solver.js questionDelayMs).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadExamApi } from './helpers/extension.js';

const { questionDelayMs } = loadExamApi().solver;
const lowest = () => 0;
const highest = () => 1;

test('no config or zero means no wait (default behavior)', () => {
  assert.equal(questionDelayMs(undefined), 0);
  assert.equal(questionDelayMs(null), 0);
  assert.equal(questionDelayMs({ mode: 'fixed', fixed: 0 }), 0);
});

test('fixed mode waits exactly the configured seconds', () => {
  assert.equal(questionDelayMs({ mode: 'fixed', fixed: 5 }), 5000);
  assert.equal(questionDelayMs({ mode: 'fixed', fixed: 1.5 }), 1500);
  // Unknown mode falls back to fixed
  assert.equal(questionDelayMs({ fixed: 2 }), 2000);
});

test('random mode stays within [min, max]', () => {
  const config = { mode: 'random', min: 3, max: 8 };
  assert.equal(questionDelayMs(config, lowest), 3000);
  assert.equal(questionDelayMs(config, highest), 8000);
  assert.equal(
    questionDelayMs(config, () => 0.5),
    5500,
  );
  for (let i = 0; i < 200; i++) {
    const ms = questionDelayMs(config);
    assert.ok(ms >= 3000 && ms <= 8000, `${ms} out of range`);
  }
});

test('random mode tolerates min > max and equal bounds', () => {
  assert.equal(questionDelayMs({ mode: 'random', min: 10, max: 2 }, lowest), 2000);
  assert.equal(questionDelayMs({ mode: 'random', min: 10, max: 2 }, highest), 10000);
  assert.equal(
    questionDelayMs({ mode: 'random', min: 4, max: 4 }, () => 0.7),
    4000,
  );
});

test('bad input is clamped: negative/NaN -> 0, capped at 10 minutes', () => {
  assert.equal(questionDelayMs({ mode: 'fixed', fixed: -3 }), 0);
  assert.equal(questionDelayMs({ mode: 'fixed', fixed: 'abc' }), 0);
  assert.equal(questionDelayMs({ mode: 'fixed', fixed: 99999 }), 600_000);
  assert.equal(questionDelayMs({ mode: 'random', min: -5, max: 2 }, lowest), 0);
});
