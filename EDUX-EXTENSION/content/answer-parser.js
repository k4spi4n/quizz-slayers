/**
 * EDUX Slayers - Answer Parser
 * Đọc đáp án AI/chatbot trả về (JSON, JSONL, JSON bị cắt, dạng "1. A"...) thành { số câu: đáp án }.
 * Thuần xử lý chuỗi, không đụng DOM — có unit test trong tests/exam-text.test.js.
 */

(function () {
  'use strict';

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
   * Hỗ trợ mảng boolean, chuỗi token tiếng Việt/Anh, số 1/0, JSON array/object
   */
  function parseTrueFalseAnswers(answerValue, expectedCount) {
    if (Array.isArray(answerValue)) {
      const parsed = answerValue.slice(0, expectedCount).map((v) => {
        if (typeof v === 'boolean') return v;
        const s = normalizeText(String(v));
        return ['đúng', 'd', 'đ', 'true', 't', '1'].includes(s);
      });
      return parsed;
    }

    if (typeof answerValue === 'object' && answerValue !== null) {
      const vals = Object.entries(answerValue)
        .sort(([k1], [k2]) => {
          const n1 = parseInt(k1, 10);
          const n2 = parseInt(k2, 10);
          if (!isNaN(n1) && !isNaN(n2)) return n1 - n2;
          return k1.localeCompare(k2);
        })
        .map(([_, v]) => v);
      return parseTrueFalseAnswers(vals, expectedCount);
    }

    const raw = String(answerValue || '').trim();

    // 1. Thử JSON.parse nếu chuỗi bắt đầu bằng [ hoặc {
    if ((raw.startsWith('[') && raw.endsWith(']')) || (raw.startsWith('{') && raw.endsWith('}'))) {
      try {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed) || (typeof parsed === 'object' && parsed !== null)) {
          return parseTrueFalseAnswers(parsed, expectedCount);
        }
      } catch (e) {}
    }

    const normalized = normalizeText(raw);
    const result = [];

    // 2. Thử khớp dạng "1. Đúng, 2. Sai" hoặc "1: Đúng, 2: Sai"
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

    // 3. Quét tất cả token từ ngữ hoặc số
    const tokens = normalized.match(/[a-zà-ỹ]+|\d/g) || [];
    for (const token of tokens) {
      if (['đúng', 'd', 'đ', 'true', 't', '1'].includes(token)) {
        result.push(true);
      } else if (['sai', 's', 'false', 'f', '0'].includes(token)) {
        result.push(false);
      }
      if (expectedCount && result.length >= expectedCount) break;
    }
    return result;
  }

  /**
   * Mô phỏng normalize_answers_payload() từ EDUX-TEST-SOLVER
   * Hỗ trợ đa dạng trường số câu (so_cau, cau, question, id, stt, q) và đáp án (dap_an, answer, ans, tra_loi, da, a)
   * Hỗ trợ cả đáp án dạng mảng [true, false] hoặc object {"1": true, "2": false}
   */
  function normalizeAnswersPayload(data) {
    const answers = {};

    function formatAnsValue(val) {
      if (val == null) return '';
      if (Array.isArray(val)) {
        return val.map((x) => String(x)).join(', ');
      }
      if (typeof val === 'object' && val !== null) {
        return Object.entries(val)
          .sort(([k1], [k2]) => {
            const n1 = parseInt(k1, 10);
            const n2 = parseInt(k2, 10);
            if (!isNaN(n1) && !isNaN(n2)) return n1 - n2;
            return k1.localeCompare(k2);
          })
          .map(([_, v]) => String(v))
          .join(', ');
      }
      return String(val).trim();
    }

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
        answers[idxInt] = formatAnsValue(ans);
      });
    } else if (typeof data === 'object' && data !== null) {
      Object.entries(data).forEach(([key, value]) => {
        const idxInt = parseInt(key, 10);
        if (isNaN(idxInt)) return;
        answers[idxInt] = formatAnsValue(value);
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

  window.EduxAnswerParser = {
    QUESTION_LABEL_RE,
    TF_TOKEN_RE,
    normalizeText,
    parseQuestionIndex,
    sanitizeAiResponse,
    parseTrueFalseAnswers,
    normalizeAnswersPayload,
    extractFromObjectChunks,
    loadAnswersFromInput
  };
})();
