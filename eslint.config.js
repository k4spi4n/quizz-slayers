import js from '@eslint/js';
import globals from 'globals';

export default [
  { ignores: ['legacy/**', 'node_modules/**', 'test-results/**', 'playwright-report/**'] },
  js.configs.recommended,
  {
    rules: {
      'no-unused-vars': ['error', { args: 'none', caughtErrors: 'none' }],
      'no-empty': ['error', { allowEmptyCatch: true }],
      // Style-only; kept as warnings so lint failures always mean a real bug
      'no-useless-escape': 'warn',
      'no-useless-assignment': 'warn',
      'preserve-caught-error': 'warn'
    }
  },
  {
    // Content scripts, injected.js, background.js, popup.js: classic scripts sharing window.Edux* namespaces
    files: ['EDUX-EXTENSION/**/*.js'],
    languageOptions: {
      sourceType: 'script',
      globals: { ...globals.browser, ...globals.webextensions }
    }
  },
  {
    // Service worker and code shared with the popup are ES modules
    files: ['EDUX-EXTENSION/background/**/*.js', 'EDUX-EXTENSION/shared/**/*.js'],
    languageOptions: { sourceType: 'module' }
  },
  {
    files: ['tests/**/*.js', 'tools/**/*.js', 'eslint.config.js'],
    languageOptions: { globals: { ...globals.node } }
  },
  {
    // page.evaluate() callbacks run inside the extension; Playwright fixtures require `({}, use)`
    files: ['tests/smoke/**/*.js'],
    languageOptions: { globals: { ...globals.browser, ...globals.webextensions } },
    rules: { 'no-empty-pattern': 'off' }
  }
];
