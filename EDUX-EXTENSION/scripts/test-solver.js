/**
 * EDUX Slayers - Test Solver Engine v2.3.0 (Mô phỏng chính xác EDUX-TEST-SOLVER)
 * Tự động trích xuất đề bài tập, chuẩn hóa prompt AI, parse đáp án đa định dạng
 * và tự động điền bài tập từng bước trên giao diện EDUX theo chuẩn Playwright.
 */

(function () {
  'use strict';

  const { sleep, safeIsVisible, safeIsEnabled, safeClick, getActiveDialog } = window.EduxDOM;

  let currentCapturedExamData = null;

  function setCapturedExamData(data) {
    if (data && typeof data === 'object') {
      currentCapturedExamData = data;
      logMessage('📡 Đã ghi nhận dữ liệu đề bài tập từ hệ thống EDUX.', 'info');
    }
  }

  function getCapturedExamData() {
    return currentCapturedExamData;
  }

  function logMessage(msg, logType = 'info') {
    console.log('[EDUX Slayers Bài Tập] ' + msg);
    try {
      chrome.runtime.sendMessage({
        type: 'TEST_LOG',
        message: msg,
        logType
      });
    } catch (e) {}
  }

  const QUESTION_LABEL_RE = /^Câu\s+(\d+)/i;
  const TF_TOKEN_RE = /(\d+)\s*\.\s*(đúng|sai|true|false|d|đ|s|t|f|1|0)/gi;

  function normalizeText(text) {
    return (text || '')
      .replace(/\u00a0/g, ' ')
      .toLowerCase()
      .trim()
      .replace(/\s+/g, ' ');
  }

  function parseQuestionIndex(text) {
    const match = (text || '').trim().match(QUESTION_LABEL_RE);
    return match ? parseInt(match[1], 10) : null;
  }

  /**
   * Mô phỏng sanitize_ai_response() từ EDUX-TEST-SOLVER
   * Xóa markdown code block ```json ... ```
   */
  function sanitizeAiResponse(raw) {
    let text = (raw || '').trim();
    if (text.charCodeAt(0) === 0xfeff) {
      text = text.substring(1).trim();
    }

    // 1. Trích xuất code block markdown ở bất kỳ đâu trong text
    const codeBlockMatch = text.match(/```(?:json|jsonl|javascript|js)?\s*([\s\S]*?)\s*```/i);
    if (codeBlockMatch) {
      text = codeBlockMatch[1].trim();
    }

    // 2. Unquote nếu text bị JSON-stringified (ví dụ: "\"[{\\\"so_cau\\\"...}\]\"")
    if (
      (text.startsWith('"') && text.endsWith('"') && text.length >= 2) ||
      (text.startsWith("'") && text.endsWith("'") && text.length >= 2)
    ) {
      try {
        const unquoted = JSON.parse(text);
        if (typeof unquoted === 'string') {
          text = unquoted.trim();
        }
      } catch (e) {
        text = text.slice(1, -1).trim();
      }
    }

    // 3. Bỏ quote bọc ngoài thừa nếu chuỗi bị paste dính nháy: "[{"so_cau": ...
    if (/^["']\s*(\[|\{)/.test(text)) {
      text = text.replace(/^["']\s*/, '');
    }
    if (/(\]|\})\s*["']$/.test(text)) {
      text = text.replace(/\s*["']$/, '');
    }

    return text.trim();
  }

  /**
   * Mô phỏng parse_true_false_answers() từ EDUX-TEST-SOLVER
   */
  function parseTrueFalseAnswers(answerValue, expectedCount) {
    const normalized = normalizeText(answerValue);
    const result = [];

    TF_TOKEN_RE.lastIndex = 0;
    let match;
    const regexMatches = [];
    while ((match = TF_TOKEN_RE.exec(normalized)) !== null) {
      regexMatches.push(match[2].toLowerCase());
    }

    if (regexMatches.length > 0) {
      for (const token of regexMatches) {
        result.push(['đúng', 'd', 'đ', 'true', 't', '1'].includes(token));
      }
      return result;
    }

    const tokens = normalized.match(/[a-zà-ỹ]+|\d/g) || [];
    for (const token of tokens) {
      if (['đúng', 'd', 'đ', 'true', 't', '1'].includes(token)) {
        result.push(true);
      } else if (['sai', 's', 'false', 'f', '0'].includes(token)) {
        result.push(false);
      }
      if (result.length >= expectedCount) break;
    }
    return result;
  }

  /**
   * Mô phỏng normalize_answers_payload() từ EDUX-TEST-SOLVER
   * Hỗ trợ đa dạng trường số câu (so_cau, cau, question, id, stt, q) và đáp án (dap_an, answer, ans, tra_loi, da, a)
   */
  function normalizeAnswersPayload(data) {
    const answers = {};
    if (Array.isArray(data)) {
      data.forEach((item) => {
        if (typeof item !== 'object' || item === null) return;
        const idx =
          item.so_cau ??
          item.soCau ??
          item.cau ??
          item.cau_so ??
          item.cau_hoi ??
          item.question ??
          item.question_id ??
          item.id ??
          item.stt ??
          item.q;
        const ans =
          item.dap_an ??
          item.dapAn ??
          item.answer ??
          item.ans ??
          item.tra_loi ??
          item.da ??
          item.a;
        if (idx == null || ans == null) return;
        const idxInt = parseInt(idx, 10);
        if (isNaN(idxInt)) return;
        if (Array.isArray(ans)) {
          answers[idxInt] = ans.map((x) => String(x)).join(', ');
        } else {
          answers[idxInt] = String(ans).trim();
        }
      });
    } else if (typeof data === 'object' && data !== null) {
      Object.entries(data).forEach(([key, value]) => {
        const idxInt = parseInt(key, 10);
        if (isNaN(idxInt)) return;
        if (Array.isArray(value)) {
          answers[idxInt] = value.map((x) => String(x)).join(', ');
        } else {
          answers[idxInt] = String(value).trim();
        }
      });
    }
    return answers;
  }

  /**
   * Trích xuất đáp án từ từng object chunk với tracking độ sâu ngoặc nhọn
   * Chống chịu tối đa với: unescaped quotes, cắt cụt chuỗi, nested brackets, sai cú pháp JSON
   */
  function extractFromObjectChunks(text) {
    const answers = {};
    let depth = 0;
    let startIdx = -1;
    const chunks = [];

    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      if (char === '{') {
        if (depth === 0) {
          startIdx = i;
        }
        depth++;
      } else if (char === '}') {
        depth--;
        if (depth === 0 && startIdx !== -1) {
          chunks.push(text.substring(startIdx + 1, i));
          startIdx = -1;
        }
      }
    }

    if (depth > 0 && startIdx !== -1) {
      chunks.push(text.substring(startIdx + 1));
    }

    for (const chunk of chunks) {
      const qMatch = chunk.match(/["']?(?:so_cau|soCau|cau|cau_so|cau_hoi|question|id|stt|q)["']?\s*:\s*(\d+)/i);
      if (!qMatch) continue;
      const qNum = parseInt(qMatch[1], 10);
      if (!qNum) continue;

      let ansMatch = chunk.match(/["']?(?:dap_an|dapAn|answer|ans|tra_loi|da|a)["']?\s*:\s*([\s\S]*)$/i);
      if (ansMatch) {
        let rawAns = ansMatch[1].trim();
        if (rawAns.endsWith(',')) rawAns = rawAns.slice(0, -1).trim();
        if (rawAns.startsWith('[') && rawAns.endsWith(']')) {
          try {
            const arr = JSON.parse(rawAns);
            if (Array.isArray(arr)) rawAns = arr.join(', ');
          } catch (e) {
            rawAns = rawAns.slice(1, -1).trim();
          }
        } else {
          if ((rawAns.startsWith('"') && rawAns.endsWith('"')) || (rawAns.startsWith("'") && rawAns.endsWith("'"))) {
            rawAns = rawAns.slice(1, -1);
          } else if (rawAns.startsWith('"')) {
            rawAns = rawAns.replace(/^"/, '').replace(/"\s*$/, '');
          }
        }
        answers[qNum] = rawAns.trim();
      } else {
        const ansBeforeMatch = chunk.match(
          /["']?(?:dap_an|dapAn|answer|ans|tra_loi|da|a)["']?\s*:\s*([\s\S]*?)(?:,\s*["']?(?:so_cau|soCau|cau|question|id)\b)/i
        );
        if (ansBeforeMatch) {
          let rawAns = ansBeforeMatch[1].trim();
          if ((rawAns.startsWith('"') && rawAns.endsWith('"')) || (rawAns.startsWith("'") && rawAns.endsWith("'"))) {
            rawAns = rawAns.slice(1, -1);
          }
          answers[qNum] = rawAns.trim();
        }
      }
    }

    return answers;
  }

  /**
   * Mô phỏng load_answers_from_jsonl_line() từ EDUX-TEST-SOLVER
   * Hỗ trợ JSONL 1 dòng, JSON Array, JSON Object, concatenated JSON, truncated JSON, và văn bản dòng (1. A, 2. B)
   */
  function loadAnswersFromInput(rawText) {
    let clean = sanitizeAiResponse(rawText);

    // 1. Thử parse JSON Array hoặc Object trực tiếp (hỗ trợ double-encoded)
    if (clean.startsWith('[') || clean.startsWith('{')) {
      try {
        let data = JSON.parse(clean);
        if (typeof data === 'string') {
          try {
            data = JSON.parse(data);
          } catch (e) {}
        }
        if (typeof data === 'object' && data !== null && !Array.isArray(data) && 'answers' in data) {
          data = data.answers;
        }
        const parsed = normalizeAnswersPayload(data);
        if (Object.keys(parsed).length > 0) return parsed;
      } catch (e) {}
    }

    // 2. Thử làm sạch JSON (bỏ trailing comma, tự đóng ngoặc nếu bị cắt)
    let repaired = clean.replace(/,\s*([\}\]])/g, '$1');
    if (repaired.startsWith('[') && !repaired.endsWith(']')) {
      const attemptEndings = [']', '"}]', '}]', '"\n}]'];
      for (const ending of attemptEndings) {
        try {
          let data = JSON.parse(repaired + ending);
          const parsed = normalizeAnswersPayload(data);
          if (Object.keys(parsed).length > 0) return parsed;
        } catch (e) {}
      }
    }

    // 3. Object-by-object chunking (bất chấp unescaped quotes, cắt cụt, sai cú pháp)
    const chunkAnswers = extractFromObjectChunks(clean);
    if (Object.keys(chunkAnswers).length > 0) {
      return chunkAnswers;
    }

    // 4. Thử tách concatenated JSON objects hoặc JSONL:
    const normalizedJson = clean.replace(/}\s*,\s*{/g, '}\n{').replace(/}\s*{/g, '}\n{');
    const jsonItems = [];
    for (const chunk of normalizedJson.split('\n')) {
      let trimmed = chunk.trim();
      if (!trimmed) continue;
      if (trimmed.startsWith('[') && trimmed.length > 1) trimmed = trimmed.substring(1).trim();
      if (trimmed.endsWith(']') && trimmed.length > 1) trimmed = trimmed.substring(0, trimmed.length - 1).trim();
      if (trimmed.endsWith(',')) trimmed = trimmed.substring(0, trimmed.length - 1).trim();
      try {
        jsonItems.push(JSON.parse(trimmed));
      } catch (e) {}
    }
    if (jsonItems.length > 0) {
      const parsed = normalizeAnswersPayload(jsonItems);
      if (Object.keys(parsed).length > 0) return parsed;
    }

    // 5. Fallback: Parse từng dòng dạng "1. A", "2: B", "Câu 3: Đúng", "4 - C"
    const answers = {};
    const lines = clean.split('\n');
    lines.forEach((l) => {
      const match = l.trim().match(/^(?:câu\s*)?(\d+)\s*[\.:\-\)]\s*(.+)$/i);
      if (match) {
        answers[parseInt(match[1], 10)] = match[2].trim();
      }
    });

    return answers;
  }

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

  /**
   * Mô phỏng build_compact_prompt_payload() từ EDUX-TEST-SOLVER
   */
  function buildCompactPromptPayload(payloadJson) {
    const data = (payloadJson && payloadJson.data) || (payloadJson || {});
    const examData = data.exam_data || (typeof data === 'object' && !data.multiple_choice ? {} : data);

    const compact = {
      title: data.title || document.title || 'Bài tập EDUX',
      total_questions: data.total_questions || 0,
      multiple_choice: [],
      fill_in_blank: [],
      essay: [],
      true_false: []
    };

    for (const item of examData.multiple_choice || (Array.isArray(data) ? data : []) || []) {
      if (item && (item.question || item.options)) {
        compact.multiple_choice.push({
          id: item.id || item.so_cau,
          question: item.question,
          options: item.options
        });
      }
    }

    for (const item of examData.fill_in_blank || []) {
      if (item && item.question) {
        compact.fill_in_blank.push({
          id: item.id || item.so_cau,
          question: item.question
        });
      }
    }

    for (const item of examData.essay || []) {
      if (item && item.question) {
        compact.essay.push({
          id: item.id || item.so_cau,
          question: item.question
        });
      }
    }

    for (const item of examData.true_false || []) {
      if (item && (item.question || item.statements)) {
        const statements = (item.statements || []).map((s) => (typeof s === 'string' ? s : s?.text || ''));
        compact.true_false.push({
          id: item.id || item.so_cau,
          question: item.question,
          statements
        });
      }
    }

    compact.total_questions =
      compact.multiple_choice.length +
      compact.fill_in_blank.length +
      compact.essay.length +
      compact.true_false.length;

    return compact;
  }

  /**
   * Tạo chuỗi prompt hoàn chỉnh chuẩn theo EDUX-TEST-SOLVER
   */
  function generateStandardPromptText(compactPayload) {
    const payloadText = JSON.stringify(compactPayload, null, 2);
    const promptInstructions =
      'YÊU CẦU ĐỘ CHÍNH XÁC CAO NHẤT (100% ACADEMIC ACCURACY):\n' +
      '1. Bạn là chuyên gia khảo thí và học thuật cao cấp. Hãy phân tích kỹ từng câu hỏi, đọc các lựa chọn loại trừ và xác định câu trả lời đúng tuyệt đối.\n' +
      '2. Định dạng đầu ra: BẮT BUỘC trả về JSONL một dòng duy nhất (hoặc JSON Array các object) để hệ thống tự động parse;\n' +
      '3. Mỗi phần tử có 2 trường: "so_cau" và "dap_an";\n' +
      '4. Đối với câu trắc nghiệm: "dap_an" là chữ cái A, B, C, hoặc D (hoặc nội dung chính xác của đáp án);\n' +
      '5. Đối với câu đúng/sai: "dap_an" là mảng giá trị Đúng/Sai theo thứ tự từng mệnh đề trong câu (ví dụ: [true, false, true, true]);\n' +
      '6. Đối với câu điền khuyết / tự luận: "dap_an" là từ/cụm từ chuẩn xác cần điền;\n' +
      '7. Tuyệt đối không giải thích, không viết thêm bất kỳ lời dẫn nào ngoài JSON.';

    return payloadText.trim() + '\n\n' + promptInstructions + '\n';
  }

  /**
   * Trích xuất câu hỏi từ dữ liệu intercepted hoặc cào từ DOM
   */
  function extractQuestions(overrideData) {
    let sourceData = overrideData || currentCapturedExamData;

    // Nếu chưa có trong RAM, thử đọc từ sessionStorage (chia sẻ giữa MAIN world & content script)
    if (!sourceData) {
      try {
        const stored = sessionStorage.getItem('__EDUX_LAST_EXAM_DATA__');
        if (stored) {
          sourceData = JSON.parse(stored);
          currentCapturedExamData = sourceData;
        }
      } catch (e) {}
    }

    if (sourceData) {
      const compact = buildCompactPromptPayload(sourceData);
      if (compact.total_questions > 0) {
        const promptText = generateStandardPromptText(compact);
        logMessage(`✓ Đã trích xuất ${compact.total_questions} câu từ dữ liệu bài tập!`, 'success');
        return { questions: compact, promptText, fromApi: true };
      }
    }

    // Fallback: Quét trực tiếp từ DOM nếu dialog bài tập đang mở
    const searchRoot = getActiveExamDialog() || getActiveDialog() || document;
    const questionLabels = Array.from(searchRoot.querySelectorAll('p, div, span, h3, h4')).filter((el) => {
      if (el.closest('nav, aside, header, footer')) return false;
      return safeIsVisible(el) && QUESTION_LABEL_RE.test((el.textContent || '').trim());
    });

    if (questionLabels.length > 0) {
      const compact = {
        title: document.title || 'Bài tập EDUX',
        total_questions: 0,
        multiple_choice: [],
        fill_in_blank: [],
        essay: [],
        true_false: []
      };

      questionLabels.forEach((labelEl) => {
        if (labelEl.closest('nav, aside, header, footer')) return;
        const qNum = parseQuestionIndex(labelEl.textContent);
        if (qNum === null) return;

        let container =
          labelEl.closest('div.border, div.rounded-xl, div.shadow, section, article') || labelEl.parentElement;
        if (!container) return;

        const questionText =
          (container.querySelector('div.prose p, p.text-gray-800') || {}).textContent?.trim() ||
          labelEl.textContent.trim();

        const tfBlocks = Array.from(
          container.querySelectorAll("div.border, div.rounded-lg, div[class*='bg-gray']")
        ).filter((el) => {
          if (!safeIsVisible(el)) return false;
          const btns = Array.from(el.querySelectorAll('button')).map((b) => (b.textContent || '').trim());
          return btns.includes('Đúng') && btns.includes('Sai');
        });

        const textarea = container.querySelector('textarea');
        const input = container.querySelector("input:not([type='hidden']):not([type='checkbox']):not([type='radio'])");
        const optionEls = Array.from(
          container.querySelectorAll(
            'div.relative.flex.items-center.space-x-2.p-2.border.rounded-lg.cursor-pointer, div.border.rounded-lg.cursor-pointer'
          )
        ).filter(safeIsVisible);

        if (tfBlocks.length > 0) {
          const statements = tfBlocks.map((b) => (b.querySelector('p, span') || b).textContent?.trim() || '');
          compact.true_false.push({ id: qNum, question: questionText, statements });
        } else if (textarea) {
          compact.essay.push({ id: qNum, question: questionText });
        } else if (input) {
          compact.fill_in_blank.push({ id: qNum, question: questionText });
        } else if (optionEls.length > 0) {
          const options = {};
          optionEls.forEach((opt, idx) => {
            const letter =
              (opt.querySelector('span.flex-shrink-0') || {}).textContent?.trim().replace(/\.$/, '') ||
              String.fromCharCode(65 + idx);
            const text = (opt.querySelector('div.prose p, p') || opt).textContent?.trim() || '';
            options[letter] = text;
          });
          compact.multiple_choice.push({ id: qNum, question: questionText, options });
        }
      });

      compact.total_questions =
        compact.multiple_choice.length +
        compact.fill_in_blank.length +
        compact.essay.length +
        compact.true_false.length;

      if (compact.total_questions > 0) {
        const promptText = generateStandardPromptText(compact);
        logMessage(`✓ Đã quét ${compact.total_questions} câu hỏi từ giao diện bài tập.`, 'info');
        return { questions: compact, promptText, fromApi: false };
      }
    }

    logMessage('⚠️ Chưa bắt được gói tin đề bài. Hãy bấm nút "🚀 Mở bài" hoặc F5 tải lại trang để bắt đề!', 'warn');
    return { questions: null, promptText: '', message: 'Chưa bắt được gói tin đề bài tập.' };
  }

  /**
   * Bấm nút "Làm bài tập" hoặc "Bài tập AI" trên trang web
   */
  async function startExercise() {
    // 1. Nếu trên trang đang thấy nút "Làm bài tập", "Làm lại", "Bài tập AI" thì luôn ưu tiên bấm nút
    // để mở bài (kể cả khi đã có kết quả trước đó trên trang)
    const match = findStartButton();

    if (match && safeIsVisible(match.element)) {
      // Xóa cache đề cũ để bắt buộc đợi đề mới của bài tập hiện tại
      currentCapturedExamData = null;
      try {
        sessionStorage.removeItem('__EDUX_LAST_EXAM_DATA__');
        sessionStorage.removeItem('__EDUX_LAST_EXAM_URL__');
        chrome.storage.local.remove('lastExamData');
      } catch (e) {}

      if (match.type === 'start_quiz') {
        const btnText = (match.element.textContent || 'Làm bài tập').trim();
        logMessage(`Đã tìm thấy nút '${btnText}'. Đang bấm để mở bài...`, 'info');
        safeClick(match.element);
        try {
          if (typeof match.element.click === 'function') match.element.click();
        } catch (e) {}

        // Chờ đề bài tập được bắt hoặc dialog xuất hiện (tối đa 12 giây)
        for (let i = 0; i < 48; i++) {
          await sleep(250);

          // Kiểm tra nếu xuất hiện hộp thoại xác nhận làm bài (khi đã làm bài 1 lần trước đó):
          const confirmBtn = findButtonByText(
            ['bắt đầu làm bài', 'làm lại', 'xác nhận', 'đồng ý', 'bắt đầu'],
            document,
            true,
            true
          );
          if (confirmBtn && confirmBtn !== match.element && safeIsVisible(confirmBtn)) {
            logMessage("Đã phát hiện hộp thoại xác nhận làm bài. Bấm xác nhận...", 'info');
            safeClick(confirmBtn);
          }

          // Kiểm tra xem đã bắt được packet chưa
          const captured = getCapturedExamData();
          if (captured) {
            const compact = buildCompactPromptPayload(captured);
            logMessage(`🎉 Đã mở bài và bắt được gói tin đề (${compact.total_questions} câu)!`, 'success');
            return { success: true, opened: true, questions: compact };
          }

          // Hoặc kiểm tra dialog câu hỏi đã xuất hiện
          const dialog = getActiveExamDialog();
          if (dialog) {
            logMessage('🎉 Cửa sổ làm bài tập đã mở thành công!', 'success');
            const extracted = extractQuestions();
            return { success: true, opened: true, questions: extracted.questions };
          }

          // Sau 2 giây nếu vẫn chưa mở, thử kích hoạt lại nút bấm
          if (i === 8 || i === 20) {
            logMessage("Đang thử kích hoạt lại nút mở bài tập...", 'info');
            safeClick(match.element);
            try {
              if (typeof match.element.click === 'function') match.element.click();
            } catch (e) {}
          }
        }

        return { success: true, opened: false, message: "Đã bấm 'Làm bài tập', đang chờ hệ thống tải câu hỏi..." };
      }

      if (match.type === 'open_lesson_exercise') {
        logMessage("Đã tìm thấy bài học. Bấm 'Bài tập AI' để mở...", 'info');
        safeClick(match.element);
        try {
          if (typeof match.element.click === 'function') match.element.click();
        } catch (e) {}

        // Chờ màn hình có nút "Làm bài tập" xuất hiện (tối đa 6 giây)
        for (let i = 0; i < 30; i++) {
          await sleep(200);
          const nextMatch = findStartButton();
          if (nextMatch && nextMatch.type === 'start_quiz') {
            logMessage("Đã mở bài tập! Tiếp tục bấm nút 'Làm bài tập'...", 'info');
            safeClick(nextMatch.element);
            try {
              if (typeof nextMatch.element.click === 'function') nextMatch.element.click();
            } catch (e) {}

            // Chờ dialog làm bài xuất hiện hoặc bắt được gói tin
            for (let j = 0; j < 48; j++) {
              await sleep(250);

              const confirmBtn = findButtonByText(
                ['bắt đầu làm bài', 'làm lại', 'xác nhận', 'đồng ý', 'bắt đầu'],
                document,
                true,
                true
              );
              if (confirmBtn && confirmBtn !== nextMatch.element && safeIsVisible(confirmBtn)) {
                logMessage("Đã phát hiện hộp thoại xác nhận làm bài. Bấm xác nhận...", 'info');
                safeClick(confirmBtn);
              }

              const captured = getCapturedExamData();
              if (captured) {
                const compact = buildCompactPromptPayload(captured);
                logMessage(`🎉 Đã mở bài và bắt được gói tin đề (${compact.total_questions} câu)!`, 'success');
                return { success: true, opened: true, questions: compact };
              }
              const dialog = getActiveExamDialog();
              if (dialog) {
                logMessage('🎉 Cửa sổ làm bài tập đã mở thành công!', 'success');
                const extracted = extractQuestions();
                return { success: true, opened: true, questions: extracted.questions };
              }
            }
            return { success: true, opened: true };
          }
        }
        return { success: true, opened: false, message: "Đã mở màn hình bài tập. Hãy bấm 'Làm bài tập' trên trang." };
      }
    }

    // 2. Nếu không thấy nút bấm trên trang, kiểm tra nếu dialog câu hỏi ĐÃ thực sự mở sẵn
    const existingDialog = getActiveExamDialog();
    if (existingDialog) {
      const qLabel = findQuestionLabel(existingDialog);
      logMessage(
        `Cửa sổ bài tập đã được mở sẵn sàng${qLabel ? ' (' + qLabel.textContent.trim() + ')' : ''}.`,
        'success'
      );
      const extracted = extractQuestions();
      return { success: true, opened: true, questions: extracted.questions };
    }

    return { success: false, message: "Không tìm thấy nút 'Làm bài tập' hoặc 'Bài tập AI' trên trang." };
  }

  /**
   * Bắt đầu điền đáp án bài tập
   */
  async function fillTestAnswers(rawText, options = {}) {
    const answers = loadAnswersFromInput(rawText);
    const questionIndices = Object.keys(answers);
    if (questionIndices.length === 0) {
      return { success: false, message: 'Không thể phân tích bất kỳ đáp án hợp lệ nào từ nội dung đã nhập!' };
    }

    logMessage(`🚀 Bắt đầu điền ${questionIndices.length} câu trả lời cho bài tập...`, 'info');

    // Chờ hoặc lấy dialog bài tập
    let dialog = getActiveExamDialog();
    if (!dialog) {
      logMessage("Cửa sổ bài tập chưa mở. Đang tự động mở bài tập để điền...", 'info');
      const startRes = await startExercise();
      if (startRes && startRes.opened) {
        await sleep(350);
        dialog = getActiveExamDialog();
      }
    }

    if (!dialog) {
      for (let wait = 0; wait < 15; wait++) {
        await sleep(200);
        dialog = getActiveExamDialog() || getActiveDialog();
        if (dialog && safeIsVisible(dialog)) break;
      }
    }

    let result;
    if (dialog && safeIsVisible(dialog)) {
      result = await fillTestDialog(dialog, answers, options);
    } else {
      result = fillTestFullPage(answers);
    }

    return result;
  }

  /**
   * Vòng lặp điền bài tập từng bước mô phỏng chính xác test_solver.py lines 483-623
   */
  async function fillTestDialog(dialog, answers, options = {}) {
    let filledCount = 0;
    const autoSubmit = options.autoSubmit !== false;
    const maxIterations = Object.keys(answers).length + 15;
    let iterations = 0;

    // Đảm bảo bắt đầu từ câu 1 nếu hiện tại đang đứng ở câu khác
    let firstLabel = findQuestionLabel(dialog);
    let startIdx = firstLabel ? parseQuestionIndex(firstLabel.textContent) : null;
    if (startIdx && startIdx > 1) {
      logMessage(`[INFO] Đang ở câu ${startIdx}. Tự động quay lại câu 1 để giải toàn bộ bài tập...`, 'info');
      const btn1 = findPaginationButton(1, dialog);
      if (btn1) {
        safeClick(btn1);
        await sleep(350);
      } else {
        for (let b = 0; b < startIdx; b++) {
          const prevBtn = findButtonByText(['Câu trước'], dialog, true, true);
          if (!prevBtn) break;
          safeClick(prevBtn);
          await sleep(150);
          const cur = findQuestionLabel(dialog);
          if (cur && parseQuestionIndex(cur.textContent) === 1) break;
        }
      }
    }

    while (iterations < maxIterations) {
      iterations++;

      // 1. Chờ label "Câu X" xuất hiện (tối đa 4 giây mỗi câu)
      let labelEl = null;
      for (let wait = 0; wait < 20; wait++) {
        labelEl = findQuestionLabel(dialog);
        if (labelEl) break;
        await sleep(200);
      }

      if (!labelEl) {
        logMessage('[WARN] Không tìm thấy nhãn câu hỏi. Dừng tiến trình.', 'warn');
        break;
      }

      const labelText = labelEl.textContent.trim();
      const questionIndex = parseQuestionIndex(labelText);
      if (questionIndex === null) {
        logMessage(`[WARN] Không parse được số câu từ: "${labelText}"`, 'warn');
        break;
      }

      const answerValue = (answers[questionIndex] || '').trim();

      if (!answerValue) {
        logMessage(`[WARN] Không có đáp án cho câu ${questionIndex}, bỏ qua.`, 'warn');
      } else {
        // Chờ ít nhất 1 phần tử tương tác của câu hỏi xuất hiện (True/False, Textarea, Input, ContentEditable, Options)
        let trueFalseBlocks = [];
        let textareaEl = null;
        let inputEls = [];
        let contentEditableEl = null;
        let optionEls = [];

        for (let wait = 0; wait < 15; wait++) {
          trueFalseBlocks = Array.from(
            dialog.querySelectorAll("div.border, div.rounded-lg, div[class*='bg-gray']")
          ).filter((el) => {
            if (!safeIsVisible(el)) return false;
            const btns = Array.from(el.querySelectorAll('button')).map((b) => (b.textContent || '').trim());
            return btns.includes('Đúng') && btns.includes('Sai');
          });

          textareaEl = dialog.querySelector('textarea');
          if (textareaEl && !safeIsVisible(textareaEl)) textareaEl = null;

          inputEls = Array.from(
            dialog.querySelectorAll("input:not([type='hidden']):not([type='checkbox']):not([type='radio'])")
          ).filter(safeIsVisible);

          contentEditableEl = dialog.querySelector('[contenteditable="true"], [role="textbox"]');
          if (contentEditableEl && !safeIsVisible(contentEditableEl)) contentEditableEl = null;

          optionEls = Array.from(
            dialog.querySelectorAll(
              "div.relative.flex.items-center.space-x-2.p-2.border.rounded-lg.cursor-pointer, div.border.rounded-lg.cursor-pointer, [role='radio']"
            )
          ).filter(safeIsVisible);

          if (
            trueFalseBlocks.length > 0 ||
            textareaEl ||
            inputEls.length > 0 ||
            contentEditableEl ||
            optionEls.length > 0
          ) {
            break;
          }
          await sleep(200);
        }

        if (trueFalseBlocks.length > 0) {
          const tfAnswers = parseTrueFalseAnswers(answerValue, trueFalseBlocks.length);
          logMessage(`[INFO] Câu ${questionIndex}: điền Đúng/Sai (${tfAnswers.length} mệnh đề)`, 'info');
          for (let i = 0; i < trueFalseBlocks.length; i++) {
            const block = trueFalseBlocks[i];
            const targetName = (i < tfAnswers.length ? tfAnswers[i] : true) ? 'Đúng' : 'Sai';
            const btn = Array.from(block.querySelectorAll('button')).find(
              (b) => (b.textContent || '').trim() === targetName
            );
            if (btn) safeClick(btn);
          }
          filledCount++;
        } else if (textareaEl) {
          logMessage(`[INFO] Câu ${questionIndex}: điền tự luận`, 'info');
          setNativeValue(textareaEl, answerValue);
          filledCount++;
        } else if (inputEls.length > 0) {
          logMessage(`[INFO] Câu ${questionIndex}: điền ô trống (${inputEls.length} ô)`, 'info');
          if (inputEls.length === 1) {
            setNativeValue(inputEls[0], answerValue);
          } else {
            let parts = [];
            try {
              const parsed = JSON.parse(answerValue);
              if (Array.isArray(parsed)) parts = parsed.map(String);
            } catch (e) {}
            if (parts.length === 0) {
              parts = answerValue.split(/[,;\n]/).map((s) => s.trim()).filter(Boolean);
            }
            for (let i = 0; i < inputEls.length; i++) {
              const val = i < parts.length ? parts[i] : answerValue;
              setNativeValue(inputEls[i], val);
            }
          }
          filledCount++;
        } else if (contentEditableEl) {
          logMessage(`[INFO] Câu ${questionIndex}: điền vùng nhập văn bản (contenteditable)`, 'info');
          setNativeValue(contentEditableEl, answerValue);
          filledCount++;
        } else {
          // Trắc nghiệm nhiều lựa chọn
          let currentOptionEls = optionEls;

          if (currentOptionEls.length === 0) {
            logMessage(`[WARN] Câu ${questionIndex}: không tìm thấy lựa chọn đáp án.`, 'warn');
          } else {
            logMessage(`[INFO] Câu ${questionIndex}: chọn '${answerValue}'`, 'info');
            const optionsList = extractOptionsFromEls(currentOptionEls);
            let chosenIndex = -1;
            const trimmedAns = answerValue.trim();

            // 1. Khớp theo ký tự đầu A, B, C, D (hỗ trợ "A", "A.", "A: ", "(A)")
            const letterMatch =
              trimmedAns.match(/^[\(\[]?([A-D])[\.\)\:\s]/i) ||
              (trimmedAns.length === 1 && trimmedAns.match(/^([A-D])$/i));

            if (letterMatch) {
              const targetLetter = letterMatch[1].toUpperCase();
              for (let i = 0; i < optionsList.length; i++) {
                const optL = optionsList[i].letter.toUpperCase().replace(/[^A-D]/g, '');
                if (optL === targetLetter || optionsList[i].letter.toUpperCase().startsWith(targetLetter)) {
                  chosenIndex = i;
                  break;
                }
              }
            }

            // 2. Khớp theo nội dung text nếu chưa tìm thấy bằng ký tự
            if (chosenIndex === -1) {
              const textWithoutLetter = trimmedAns.replace(/^[\(\[]?[A-D][\.\)\:\s\-]+/i, '').trim();
              const target = normalizeText(textWithoutLetter || trimmedAns);
              if (target) {
                for (let i = 0; i < optionsList.length; i++) {
                  const optText = normalizeText(optionsList[i].text);
                  if (optText && (optText.includes(target) || target.includes(optText))) {
                    chosenIndex = i;
                    break;
                  }
                }
              }
            }

            // 3. Khớp theo số thứ tự (1..4)
            if (chosenIndex === -1 && /^[1-4]$/.test(trimmedAns)) {
              const idx = parseInt(trimmedAns, 10) - 1;
              if (optionsList[idx]) chosenIndex = idx;
            }

            if (chosenIndex === -1) {
              logMessage(`[WARN] Câu ${questionIndex}: Không khớp được lựa chọn nào cho '${answerValue}'.`, 'warn');
            } else {
              safeClick(optionsList[chosenIndex].node);
              filledCount++;
            }
          }
        }
      }

      await sleep(250);

      // Kiểm tra nút "Nộp bài" và "Câu tiếp"
      const submitBtn = findButtonByText(['Nộp bài'], dialog, true, true);
      const nextBtn = findButtonByText(['Câu tiếp', 'Câu tiếp theo'], dialog, true, true);

      // Nếu chỉ có nút Nộp bài hoặc không còn Câu tiếp
      if (!nextBtn && submitBtn) {
        if (autoSubmit) {
          logMessage("🎉 Đã đến câu cuối. Tự động bấm nút 'Nộp bài'...", 'success');
          safeClick(submitBtn);
          await sleep(500);
          const confirmBtn = findButtonByText(['Xác nhận', 'Đồng ý', 'Chắc chắn'], document, true, true);
          if (confirmBtn) {
            logMessage("✓ Bấm xác nhận nộp bài...", 'info');
            safeClick(confirmBtn);
          }
        } else {
          logMessage("✓ Đã hoàn thành điền câu cuối. Bạn có thể bấm 'Nộp bài'.", 'success');
        }
        break;
      }

      if (nextBtn) {
        const currentLabel = labelText;
        const progressEl = dialog.querySelector('span.text-gray-700');
        const currentProgress = progressEl ? progressEl.textContent.trim() : '';

        safeClick(nextBtn);

        // Chờ câu tiếp theo xuất hiện (label đổi HOẶC progress đổi - mô phỏng test_solver.py lines 602-618)
        let changed = false;
        for (let i = 0; i < 35; i++) {
          await sleep(150);
          const newLabelEl = findQuestionLabel(dialog);
          const newLabel = newLabelEl ? newLabelEl.textContent.trim() : '';
          const newProgressEl = dialog.querySelector('span.text-gray-700');
          const newProgress = newProgressEl ? newProgressEl.textContent.trim() : '';

          if ((newLabel && newLabel !== currentLabel) || (newProgress && newProgress !== currentProgress)) {
            changed = true;
            break;
          }
        }

        // Fallback: nếu bấm nextBtn không đổi, thử bấm pagination button kế tiếp
        if (!changed) {
          const nextIndex = questionIndex + 1;
          const paginationBtn = findPaginationButton(nextIndex, dialog);
          if (paginationBtn) {
            logMessage(`[INFO] Thử chuyển câu bằng nút số ${nextIndex}...`, 'info');
            safeClick(paginationBtn);
            for (let i = 0; i < 20; i++) {
              await sleep(150);
              const newLabelEl = findQuestionLabel(dialog);
              if (newLabelEl && newLabelEl.textContent.trim() !== currentLabel) {
                changed = true;
                break;
              }
            }
          }
        }

        if (!changed) {
          const endSubmitBtn = findButtonByText(['Nộp bài'], dialog, true, true);
          if (endSubmitBtn && autoSubmit) {
            logMessage("🎉 Không còn câu tiếp theo. Tự động bấm nút 'Nộp bài'...", 'success');
            safeClick(endSubmitBtn);
            await sleep(500);
            const confirmBtn = findButtonByText(['Xác nhận', 'Đồng ý', 'Chắc chắn'], document, true, true);
            if (confirmBtn) safeClick(confirmBtn);
          } else {
            logMessage('[WARN] Câu tiếp theo chưa hiển thị kịp hoặc đã đến cuối bài.', 'warn');
          }
          break;
        }
      } else if (submitBtn) {
        if (autoSubmit) {
          logMessage("🎉 Tự động bấm nút 'Nộp bài'...", 'success');
          safeClick(submitBtn);
          await sleep(500);
          const confirmBtn = findButtonByText(['Xác nhận', 'Đồng ý', 'Chắc chắn'], document, true, true);
          if (confirmBtn) safeClick(confirmBtn);
        }
        break;
      } else {
        // Không còn nút Câu tiếp và không có nút Nộp bài (bài tập tự lưu)
        logMessage("🎉 Đã hoàn thành câu cuối cùng của bài tập!", 'success');
        break;
      }
    }

    logMessage(`🎉 Hoàn tất! Đã điền xong ${filledCount} câu trong bài tập.`, 'success');

    // Dọn dẹp cache đề đã nộp và thông báo cho Popup cập nhật UI
    currentCapturedExamData = null;
    try {
      sessionStorage.removeItem('__EDUX_LAST_EXAM_DATA__');
      sessionStorage.removeItem('__EDUX_LAST_EXAM_URL__');
      chrome.storage.local.remove('lastExamData');
      chrome.runtime.sendMessage({ type: 'EXAM_SUBMITTED' });
    } catch (e) {}

    return { success: true, filledCount };
  }

  /**
   * Fallback khi bài tập hiển thị cả trang (không phải modal)
   */
  function fillTestFullPage(answers) {
    let filledCount = 0;
    const questionLabels = Array.from(document.querySelectorAll('p, div, span, h3, h4')).filter((el) => {
      return safeIsVisible(el) && QUESTION_LABEL_RE.test((el.textContent || '').trim());
    });

    questionLabels.forEach((labelEl) => {
      const qNum = parseQuestionIndex(labelEl.textContent);
      if (qNum === null) return;

      const targetAns = answers[qNum];
      if (!targetAns) return;

      let container =
        labelEl.closest('div.border, div.rounded-xl, div.shadow, section, article') || labelEl.parentElement;
      if (!container) return;

      const tfBlocks = Array.from(
        container.querySelectorAll("div.border, div.rounded-lg, div[class*='bg-gray']")
      ).filter((el) => {
        if (!safeIsVisible(el)) return false;
        const btns = Array.from(el.querySelectorAll('button')).map((b) => (b.textContent || '').trim());
        return btns.includes('Đúng') && btns.includes('Sai');
      });

      if (tfBlocks.length > 0) {
        const tfAnswers = parseTrueFalseAnswers(targetAns, tfBlocks.length);
        tfBlocks.forEach((block, i) => {
          const shouldBeTrue = i < tfAnswers.length ? tfAnswers[i] : true;
          const btn = Array.from(block.querySelectorAll('button')).find(
            (b) => (b.textContent || '').trim() === (shouldBeTrue ? 'Đúng' : 'Sai')
          );
          if (btn) safeClick(btn);
        });
        filledCount++;
        logMessage(`✓ Câu ${qNum}: đã chọn Đúng/Sai`, 'success');
        return;
      }

      const textarea = container.querySelector('textarea');
      if (textarea && safeIsVisible(textarea)) {
        setNativeValue(textarea, targetAns);
        filledCount++;
        logMessage(`✓ Câu ${qNum}: đã điền tự luận`, 'success');
        return;
      }

      const input = container.querySelector("input:not([type='hidden']):not([type='checkbox']):not([type='radio'])");
      if (input && safeIsVisible(input)) {
        setNativeValue(input, targetAns);
        filledCount++;
        logMessage(`✓ Câu ${qNum}: đã điền ô trống`, 'success');
        return;
      }

      const contentEditable = container.querySelector('[contenteditable="true"], [role="textbox"]');
      if (contentEditable && safeIsVisible(contentEditable)) {
        setNativeValue(contentEditable, targetAns);
        filledCount++;
        logMessage(`✓ Câu ${qNum}: đã điền vùng nhập văn bản`, 'success');
        return;
      }

      const optionEls = Array.from(
        container.querySelectorAll(
          'div.relative.flex.items-center.space-x-2.p-2.border.rounded-lg.cursor-pointer, div.border.rounded-lg.cursor-pointer, label'
        )
      ).filter(safeIsVisible);

      for (const opt of optionEls) {
        const optText = (opt.textContent || '').trim();
        const letterSpan = (opt.querySelector('span.flex-shrink-0') || {}).textContent?.trim() || '';
        const targetUpper = targetAns.toUpperCase();

        if (
          letterSpan === targetUpper ||
          letterSpan.startsWith(targetUpper + '.') ||
          optText.startsWith(targetUpper + '.') ||
          (targetAns.length > 1 && normalizeText(optText).includes(normalizeText(targetAns)))
        ) {
          safeClick(opt);
          filledCount++;
          logMessage(`✓ Câu ${qNum}: đã chọn ${targetAns}`, 'success');
          break;
        }
      }
    });

    logMessage(`🎉 Đã tự động điền xong ${filledCount} câu hỏi bài tập.`, 'success');
    return { success: true, filledCount };
  }

  // =========================================================================
  // Xuất API toàn cục cho Extension
  // =========================================================================
  window.EduxTestSolver = {
    fillTestAnswers,
    extractQuestions,
    startExercise,
    setCapturedExamData,
    getCapturedExamData,
    getActiveExamDialog,
    findStartButton,
    loadAnswersFromInput,
    sanitizeAiResponse,
    buildCompactPromptPayload,
    generateStandardPromptText
  };
})();
