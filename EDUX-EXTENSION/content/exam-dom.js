/**
 * EDUX Slayers - Exam DOM
 * Tìm hộp thoại bài tập, nút bấm, nhãn câu hỏi và điền giá trị vào ô nhập trên giao diện EDUX.
 */

(function () {
  'use strict';

  const { safeIsVisible, safeIsEnabled } = window.EduxDOM;
  const { normalizeText, QUESTION_LABEL_RE } = window.EduxAnswerParser;

  /**
   * Cập nhật giá trị input/textarea cho React synthetic events (hỗ trợ cả contenteditable)
   */
  function setNativeValue(el, value) {
    if (!el) return;

    if (el.isContentEditable) {
      el.focus();
      el.textContent = value;
      el.dispatchEvent(new InputEvent('input', { bubbles: true, composed: true }));
      el.dispatchEvent(new Event('change', { bubbles: true, composed: true }));
      el.dispatchEvent(new Event('blur', { bubbles: true, composed: true }));
      return;
    }

    const proto = el.tagName === 'TEXTAREA' ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;

    if (el._valueTracker) {
      el._valueTracker.setValue('');
    }

    el.dispatchEvent(new Event('focus', { bubbles: true, composed: true }));

    if (setter) {
      setter.call(el, value);
    } else {
      el.value = value;
    }

    el.dispatchEvent(new InputEvent('input', { bubbles: true, composed: true }));
    el.dispatchEvent(new Event('change', { bubbles: true, composed: true }));
    el.dispatchEvent(new Event('blur', { bubbles: true, composed: true }));
  }

  /**
   * Trích xuất danh sách lựa chọn trong câu hỏi trắc nghiệm
   * Mô phỏng extract_options() từ EDUX-TEST-SOLVER
   */
  function extractOptionsFromEls(optionEls) {
    return optionEls.map((node, index) => {
      let letter = (
        node.querySelector('span.flex-shrink-0, span[class*="rounded-full"], div[class*="rounded-full"]')?.textContent ||
        ''
      ).trim();
      let text = (
        node.querySelector('div.prose p, p, span.text-gray-900, div.text-gray-900')?.textContent ||
        node.textContent ||
        ''
      ).trim();

      if (!letter) {
        const match = text.match(/^([A-D])[\.\)\:\s]/i);
        if (match) {
          letter = match[1].toUpperCase();
        } else if (index < 4) {
          letter = String.fromCharCode(65 + index);
        }
      }

      return { node, letter, text };
    });
  }

  /**
   * Tìm danh sách các khối mệnh đề Đúng/Sai thực sự (loại trừ hoàn toàn container cha)
   * Mỗi block hợp lệ phải chứa đúng 1 nút 'Đúng' và 1 nút 'Sai'
   */
  function findTrueFalseBlocks(root) {
    if (!root) return [];

    // 1. Tìm tất cả button có nhãn 'Đúng' hoặc 'Sai' đang hiển thị
    const allButtons = Array.from(root.querySelectorAll('button')).filter((b) => {
      const text = (b.textContent || '').trim();
      return safeIsVisible(b) && (text === 'Đúng' || text === 'Sai');
    });

    if (allButtons.length < 2) return [];

    // 2. Tìm khối bọc trực tiếp của từng cặp nút
    const candidateRows = new Set();
    for (const btn of allButtons) {
      const row = btn.closest(
        'div.border.border-gray-200.rounded-lg.p-3.bg-gray-50, ' +
        'div.border.border-gray-200, ' +
        'div.border.rounded-lg, ' +
        'div.border, ' +
        'div[class*="bg-gray"]'
      );
      if (row && safeIsVisible(row)) {
        candidateRows.add(row);
      }
    }

    // 3. Lọc chỉ lấy các phần tử chứa đúng 1 nút 'Đúng' và 1 nút 'Sai'
    let blocks = Array.from(candidateRows).filter((row) => {
      const btns = Array.from(row.querySelectorAll('button')).map((b) => (b.textContent || '').trim());
      const dungCount = btns.filter((t) => t === 'Đúng').length;
      const saiCount = btns.filter((t) => t === 'Sai').length;
      return dungCount === 1 && saiCount === 1;
    });

    // 4. Nếu không tìm thấy qua candidateRows (ví dụ layout tùy biến), quét từ các div con
    if (blocks.length === 0) {
      const parentDivs = Array.from(root.querySelectorAll('div')).filter((d) => {
        if (!safeIsVisible(d)) return false;
        const btns = Array.from(d.querySelectorAll('button')).map((b) => (b.textContent || '').trim());
        const dungCount = btns.filter((t) => t === 'Đúng').length;
        const saiCount = btns.filter((t) => t === 'Sai').length;
        return dungCount === 1 && saiCount === 1;
      });
      blocks = parentDivs;
    }

    // 5. Loại bỏ các phần tử cha nếu còn bao bọc phần tử con khác trong danh sách (chỉ giữ leaf blocks)
    blocks = blocks.filter((row) => {
      return !blocks.some((other) => other !== row && row.contains(other));
    });

    // 6. Sắp xếp các block theo thứ tự xuất hiện trên trang (top-down)
    blocks.sort((a, b) => {
      const pos = a.compareDocumentPosition(b);
      if (pos & Node.DOCUMENT_POSITION_FOLLOWING) return -1;
      if (pos & Node.DOCUMENT_POSITION_PRECEDING) return 1;
      return 0;
    });

    return blocks;
  }

  /**
   * Kiểm tra một phần tử có phải là dialog bài tập/câu hỏi thực sự hay không
   */
  function isExamDialog(el) {
    if (!el || !safeIsVisible(el)) return false;
    // Bỏ qua navigation bar, sidebar, header, footer
    if (el.closest('nav, aside, header, footer') || el.tagName === 'NAV' || el.tagName === 'ASIDE') {
      return false;
    }
    // Bỏ qua nếu đang ở trạng thái closed hoặc aria-hidden
    if (el.getAttribute('data-state') === 'closed' || el.getAttribute('aria-hidden') === 'true') {
      return false;
    }
    if (el.closest('[data-state="closed"]') || el.closest('[aria-hidden="true"]')) {
      return false;
    }

    // Bỏ qua các card kết quả lần làm trước hoặc phần tổng kết điểm số
    const text = (el.textContent || '').toLowerCase();
    if (
      text.includes('kết quả làm bài') ||
      text.includes('bài kiểm tra lúc') ||
      (text.includes('thời gian:') && text.includes('nhận xét:')) ||
      text.includes('số lần đã làm')
    ) {
      return false;
    }

    // Phải có nhãn câu hỏi "Câu X" HOẶC nút chuyển câu / nộp bài
    const hasQLabel = !!findQuestionLabel(el);
    const hasExamBtn = !!findButtonByText(['Nộp bài', 'Câu tiếp', 'Câu tiếp theo'], el, false);

    if (hasQLabel || hasExamBtn) {
      return true;
    }

    return false;
  }

  /**
   * Tìm dialog làm bài tập hiện tại (đảm bảo không nhận nhầm sidebar/menu)
   */
  function getActiveExamDialog() {
    const candidateSelectors = [
      "div[role='dialog'][data-state='open']",
      "div[role='dialog'][data-slot='dialog-content']",
      "div[data-slot='dialog-content'][data-state='open']",
      "div[data-slot='dialog-content']",
      "div[role='dialog']",
      "[aria-modal='true']"
    ];

    for (const sel of candidateSelectors) {
      try {
        const els = document.querySelectorAll(sel);
        for (const el of els) {
          if (isExamDialog(el)) {
            return el;
          }
        }
      } catch (e) {}
    }

    // Kiểm tra các div fixed / overlay có z-index cao
    const dialogs = Array.from(document.querySelectorAll('div.fixed, div.absolute')).filter(safeIsVisible);
    for (const d of dialogs) {
      const style = window.getComputedStyle(d);
      if (parseInt(style.zIndex, 10) >= 20 && isExamDialog(d)) {
        return d;
      }
    }

    return null;
  }

  /**
   * Tìm nút "Làm bài tập", "Làm lại bài tập", hoặc "Bài tập AI" thông minh & toàn diện
   */
  function findStartButton() {
    // 1. Ưu tiên tìm trực tiếp trong thẻ BUTTON, [role="button"], A
    const clickables = Array.from(document.querySelectorAll('button, [role="button"], a'));
    for (const btn of clickables) {
      if (!safeIsVisible(btn)) continue;
      if (btn.closest('nav, aside, header')) continue;
      const text = normalizeText(btn.textContent);
      if (
        text.includes('làm bài tập') ||
        text.includes('làm lại bài tập') ||
        text.includes('làm lại') ||
        text.includes('bắt đầu làm bài')
      ) {
        return { element: btn, type: 'start_quiz' };
      }
    }

    // 2. Tìm qua các thẻ văn bản con (span, p, div) có chứa chữ 'làm bài tập'
    const textEls = Array.from(document.querySelectorAll('span, p, div, h1, h2, h3, h4, b, strong'));
    for (const el of textEls) {
      if (!safeIsVisible(el)) continue;
      if (el.closest('nav, aside, header')) continue;
      const text = normalizeText(el.textContent);
      if (
        text === 'làm bài tập' ||
        text === 'làm lại' ||
        ((text.includes('làm bài tập') || text.includes('làm lại bài tập')) && text.length < 30)
      ) {
        const clickable =
          el.closest('button, [role="button"], a, div[class*="cursor-pointer"], div[class*="btn"], div[class*="bg-"]') ||
          el;
        return { element: clickable, type: 'start_quiz' };
      }
    }

    // 3. Ưu tiên nút "Bài tập AI" của các bài học trên trang môn học (/subject?id=...)
    for (const btn of clickables) {
      if (!safeIsVisible(btn)) continue;
      if (btn.closest('nav, aside, header')) continue;
      const text = normalizeText(btn.textContent);
      if (text.includes('bài tập ai')) {
        return { element: btn, type: 'open_lesson_exercise' };
      }
    }

    const lessonEls = Array.from(document.querySelectorAll('span, p, div'));
    for (const el of lessonEls) {
      if (!safeIsVisible(el)) continue;
      if (el.closest('nav, aside, header')) continue;
      const text = normalizeText(el.textContent);
      if (text.includes('bài tập ai') && text.length < 30) {
        const clickable =
          el.closest('button, [role="button"], a, div[class*="cursor-pointer"], div[class*="btn"]') || el;
        return { element: clickable, type: 'open_lesson_exercise' };
      }
    }

    return null;
  }

  /**
   * Tìm nhãn "Câu X" trong dialog (ưu tiên span theo chuẩn Playwright)
   */
  function findQuestionLabel(container) {
    const root = container || document;
    const spans = Array.from(root.querySelectorAll('span')).filter((el) => {
      if (el.closest('nav, aside, header')) return false;
      const text = (el.textContent || '').trim();
      return safeIsVisible(el) && text.length < 40 && QUESTION_LABEL_RE.test(text);
    });
    if (spans.length > 0) return spans[0];

    return (
      Array.from(root.querySelectorAll('p, div, h3, h4, b, strong')).find((el) => {
        if (el.closest('nav, aside, header')) return false;
        const text = (el.textContent || '').trim();
        return safeIsVisible(el) && text.length < 40 && QUESTION_LABEL_RE.test(text);
      }) || null
    );
  }

  /**
   * Tìm button theo tên/nhãn chữ với độ chính xác cao theo chuẩn Playwright
   */
  function findButtonByText(names, root = document, mustBeVisible = true, mustBeEnabled = true) {
    const nameList = Array.isArray(names) ? names : [names];
    const lowerTargets = nameList.map((n) => normalizeText(n));

    // Pass 1: Tìm trực tiếp trong các thẻ BUTTON, [role="button"], A
    const actualButtons = Array.from(root.querySelectorAll('button, [role="button"], a'));
    for (const target of lowerTargets) {
      for (const btn of actualButtons) {
        if (mustBeVisible && !safeIsVisible(btn)) continue;
        if (mustBeEnabled && !safeIsEnabled(btn)) continue;
        const text = normalizeText(btn.textContent);
        if (text === target || (text.includes(target) && text.length <= target.length + 20)) {
          return btn;
        }
      }
    }

    // Pass 2: Tìm trong các thẻ con ngắn (span, p, b, strong) nằm trong button/clickable
    const textEls = Array.from(root.querySelectorAll('span, p, b, strong'));
    for (const target of lowerTargets) {
      for (const el of textEls) {
        if (mustBeVisible && !safeIsVisible(el)) continue;
        const text = normalizeText(el.textContent);
        if (text === target || (text.includes(target) && text.length <= target.length + 15)) {
          const parent = el.closest('button, [role="button"], a, div[class*="cursor-pointer"]');
          if (parent) {
            if (mustBeVisible && !safeIsVisible(parent)) continue;
            if (mustBeEnabled && !safeIsEnabled(parent)) continue;
            return parent;
          }
        }
      }
    }

    return null;
  }

  /**
   * Tìm nút số trang/câu hỏi trong pagination bar
   */
  function findPaginationButton(qIndex, root = document) {
    const target = String(qIndex).trim();
    const buttons = Array.from(root.querySelectorAll('button, [role="button"]'));
    return (
      buttons.find((btn) => {
        if (!safeIsVisible(btn) || !safeIsEnabled(btn)) return false;
        return (btn.textContent || '').trim() === target;
      }) || null
    );
  }

  window.EduxExamDOM = {
    setNativeValue,
    extractOptionsFromEls,
    findTrueFalseBlocks,
    isExamDialog,
    getActiveExamDialog,
    findStartButton,
    findQuestionLabel,
    findButtonByText,
    findPaginationButton
  };
})();
