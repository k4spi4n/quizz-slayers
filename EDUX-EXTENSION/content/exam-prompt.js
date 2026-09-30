/**
 * EDUX Slayers - Exam Prompt
 * Chuẩn hóa dữ liệu đề bài tập thành payload gọn và prompt gửi AI / chatbot.
 * Thuần xử lý dữ liệu, không đụng DOM — có unit test trong tests/exam-text.test.js.
 */

(function () {
  'use strict';

  /**
   * Mô phỏng build_compact_prompt_payload() từ EDUX-TEST-SOLVER
   * Chuẩn hóa gán ID tuần tự liên tục (1..N), tránh va chạm ID giữa các phần trắc nghiệm và đúng/sai
   */
  function buildCompactPromptPayload(payloadJson, fallbackTitle) {
    const data = (payloadJson && payloadJson.data) || payloadJson || {};
    const examData =
      data.exam_data || (typeof data === 'object' && !data.multiple_choice ? {} : data);

    const compact = {
      title: data.title || fallbackTitle || 'Bài tập EDUX',
      total_questions: data.total_questions || 0,
      multiple_choice: [],
      fill_in_blank: [],
      essay: [],
      true_false: [],
    };

    let nextFallbackIndex = 1;
    const usedIds = new Set();

    function resolveQuestionId(item) {
      const rawId = item?.id ?? item?.so_cau ?? item?.question_id ?? item?.stt ?? item?.order;
      const parsed = parseInt(rawId, 10);
      if (!isNaN(parsed) && parsed > 0 && !usedIds.has(parsed)) {
        usedIds.add(parsed);
        if (parsed >= nextFallbackIndex) {
          nextFallbackIndex = parsed + 1;
        }
        return parsed;
      }
      while (usedIds.has(nextFallbackIndex)) {
        nextFallbackIndex++;
      }
      const assigned = nextFallbackIndex;
      usedIds.add(assigned);
      nextFallbackIndex++;
      return assigned;
    }

    // 1. Trắc nghiệm (multiple_choice)
    for (const item of examData.multiple_choice || (Array.isArray(data) ? data : []) || []) {
      if (item && (item.question || item.options || item.title || item.content)) {
        compact.multiple_choice.push({
          id: resolveQuestionId(item),
          question: item.question || item.title || item.content || '',
          options: item.options || item.choices || {},
        });
      }
    }

    // 2. Điền khuyết (fill_in_blank)
    for (const item of examData.fill_in_blank || []) {
      if (item && (item.question || item.title || item.content)) {
        compact.fill_in_blank.push({
          id: resolveQuestionId(item),
          question: item.question || item.title || item.content || '',
        });
      }
    }

    // 3. Tự luận (essay)
    for (const item of examData.essay || []) {
      if (item && (item.question || item.title || item.content)) {
        compact.essay.push({
          id: resolveQuestionId(item),
          question: item.question || item.title || item.content || '',
        });
      }
    }

    // 4. Đúng / Sai (true_false)
    for (const item of examData.true_false || []) {
      if (item && (item.question || item.statements || item.title || item.content || item.items)) {
        const rawStatements = item.statements || item.items || item.sub_questions || [];
        const statements = rawStatements.map((s) => {
          if (typeof s === 'string') return s;
          return s?.text || s?.statement || s?.content || s?.title || '';
        });
        compact.true_false.push({
          id: resolveQuestionId(item),
          question: item.question || item.title || item.content || '',
          statements,
        });
      }
    }

    // 5. Mảng câu hỏi tổng hợp (examData.questions) nếu có
    if (
      Array.isArray(examData.questions) &&
      compact.multiple_choice.length === 0 &&
      compact.true_false.length === 0
    ) {
      for (const item of examData.questions) {
        const qType = (item.question_type || item.type || '').toLowerCase();
        if (qType.includes('choice') || item.options || item.choices) {
          compact.multiple_choice.push({
            id: resolveQuestionId(item),
            question: item.question || item.question_text || item.title || '',
            options: item.options || item.choices || {},
          });
        } else if (
          qType.includes('true') ||
          qType.includes('false') ||
          item.statements ||
          item.items
        ) {
          const rawStatements = item.statements || item.items || item.sub_questions || [];
          const statements = rawStatements.map((s) =>
            typeof s === 'string' ? s : s?.text || s?.statement || '',
          );
          compact.true_false.push({
            id: resolveQuestionId(item),
            question: item.question || item.question_text || item.title || '',
            statements,
          });
        } else if (qType.includes('essay')) {
          compact.essay.push({
            id: resolveQuestionId(item),
            question: item.question || item.question_text || item.title || '',
          });
        } else {
          compact.fill_in_blank.push({
            id: resolveQuestionId(item),
            question: item.question || item.question_text || item.title || '',
          });
        }
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

  window.EduxExamPrompt = {
    buildCompactPromptPayload,
    generateStandardPromptText,
  };
})();
