/**
 * EDUX Slayers - DOM Utilities
 * High-precision browser interaction utilities matching Playwright standards
 */

(function () {
  'use strict';

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function safeIsVisible(el) {
    if (!el) return false;
    try {
      const style = window.getComputedStyle(el);
      if (style.display === 'none') return false;
      if (style.visibility === 'hidden') return false;
      if (parseFloat(style.opacity) === 0) return false;

      const rect = el.getBoundingClientRect();
      if (rect.width === 0 && rect.height === 0) return false;

      return true;
    } catch (e) {
      return false;
    }
  }

  /**
   * Safe enabled check matching Playwright's safe_is_enabled().
   * Handles disabled attribute, aria-disabled="true", data-disabled="true",
   * and Tailwind disabled classes without false positives on prefixes like disabled:cursor-not-allowed.
   */
  function safeIsEnabled(el) {
    if (!el || !safeIsVisible(el)) return false;
    try {
      if (el.disabled) return false;
      if (el.getAttribute('aria-disabled') === 'true') return false;
      if (el.getAttribute('data-disabled') === 'true') return false;

      // Check real computed style in browser
      const style = window.getComputedStyle(el);
      if (style.pointerEvents === 'none') return false;

      // Check standalone classes only (classList.contains does not match disabled:cursor-not-allowed)
      const classList = el.classList;
      if (classList && classList.contains('cursor-not-allowed') && classList.contains('pointer-events-none')) {
        return false;
      }

      return true;
    } catch (e) {
      return false;
    }
  }

  /**
   * Safe click with full Pointer/Mouse event dispatching for React/Vue synthetic events.
   * Calculates actual element coordinates and sets valid button/buttons states.
   */
  function safeClick(el) {
    if (!el) return false;
    try {
      if (typeof el.scrollIntoView === 'function') {
        el.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: 'instant' });
      }

      const rect = el.getBoundingClientRect();
      const clientX = rect.left + rect.width / 2;
      const clientY = rect.top + rect.height / 2;

      const baseOpts = {
        bubbles: true,
        cancelable: true,
        view: window,
        composed: true,
        clientX,
        clientY,
        screenX: (window.screenX || 0) + clientX,
        screenY: (window.screenY || 0) + clientY
      };

      el.dispatchEvent(new PointerEvent('pointerover', baseOpts));
      el.dispatchEvent(new MouseEvent('mouseover', baseOpts));
      el.dispatchEvent(new PointerEvent('pointerdown', { ...baseOpts, button: 0, buttons: 1, pointerId: 1, pointerType: 'mouse', isPrimary: true }));
      el.dispatchEvent(new MouseEvent('mousedown', { ...baseOpts, button: 0, buttons: 1 }));
      if (typeof el.focus === 'function') el.focus();
      el.dispatchEvent(new PointerEvent('pointerup', { ...baseOpts, button: 0, buttons: 0, pointerId: 1, pointerType: 'mouse', isPrimary: true }));
      el.dispatchEvent(new MouseEvent('mouseup', { ...baseOpts, button: 0, buttons: 0 }));

      if (typeof el.click === 'function') {
        el.click();
      } else {
        el.dispatchEvent(new MouseEvent('click', { ...baseOpts, button: 0, buttons: 0 }));
      }
      return true;
    } catch (e) {
      console.error('[EDUX Slayers] Click error:', e);
      try {
        if (typeof el.click === 'function') {
          el.click();
          return true;
        }
      } catch (err) {}
      return false;
    }
  }

  /**
   * Locate the active dialog modal if present.
   * Matches Playwright: get_active_dialog()
   * Only matches genuine modal overlays (dialog role, aria-modal, fixed/absolute overlay).
   */
  function getActiveDialog() {
    const selectors = [
      "div[role='dialog'][data-state='open']",
      "div[role='dialog'][data-slot='dialog-content']",
      "div[data-slot='dialog-content'][data-state='open']",
      "div[data-slot='dialog-content']",
      "div[role='dialog']",
      "[aria-modal='true']"
    ];
    for (const sel of selectors) {
      try {
        const els = document.querySelectorAll(sel);
        for (const el of els) {
          if (!safeIsVisible(el)) continue;
          if (el.getAttribute('data-state') === 'closed') continue;
          if (el.getAttribute('aria-hidden') === 'true') continue;
          if (el.closest('[data-state="closed"]')) continue;
          if (el.closest('[aria-hidden="true"]')) continue;
          if (el.closest('nav, aside, header')) continue;
          if (el.tagName === 'NAV' || el.tagName === 'ASIDE') continue;

          const style = window.getComputedStyle(el);
          if (
            el.getAttribute('role') === 'dialog' ||
            el.getAttribute('aria-modal') === 'true' ||
            style.position === 'fixed' ||
            style.position === 'absolute' ||
            parseInt(style.zIndex, 10) > 10
          ) {
            return el;
          }
        }
      } catch (e) {}
    }
    return null;
  }

  /**
   * Find "Trang sau" / completion button inside popup quiz or in-page quiz.
   * Strictly matches completion buttons and ignores slide numbers, sidebars, and lecture navigation.
   * Matches Playwright: get_dialog_next_page_button()
   */

  async function waitForHidden(el, timeoutMs = 800) {
    if (!el) return;
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      if (!safeIsVisible(el) || !safeIsEnabled(el)) return;
      await sleep(35);
    }
  }

  /**
   * Logger gửi về popup: in ra console kèm tiền tố và gửi { type, message, logType, ...extra() }.
   * Popup đóng thì không có ai nhận -> bỏ qua lỗi.
   */
  function createLogger(prefix, type, extra = () => ({})) {
    return function logMessage(msg, logType = 'info') {
      console.log(`${prefix} ${msg}`);
      try {
        chrome.runtime.sendMessage({ type, message: msg, logType, ...extra() });
      } catch (e) {}
    };
  }

  window.EduxDOM = {
    createLogger,
    sleep,
    safeIsVisible,
    safeIsEnabled,
    safeClick,
    getActiveDialog,
    waitForHidden
  };
})();
