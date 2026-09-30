// Prompt gửi AI và cách đọc kết quả trả về, tách riêng để chỉnh prompt không đụng tới logic gọi API.

export function slideSystemPrompt(choiceCount) {
  return (
    'Bạn là chuyên gia giáo dục và bài giảng học tập với độ chính xác tuyệt đối.\n' +
    'Nhiệm vụ: Phân tích kỹ lưỡng câu hỏi và chọn duy nhất 1 đáp án chính xác nhất trong các lựa chọn được cung cấp.\n' +
    'Định dạng đầu ra BẮT BUỘC là JSON duy nhất: {"index": X} trong đó X là số thứ tự (từ 0 đến ' +
    (choiceCount - 1) +
    ') của lựa chọn đúng nhất.\n' +
    'Tuyệt đối không giải thích, không viết thêm bất kỳ chữ nào ngoài chuỗi JSON.'
  );
}

export function slideUserPrompt(question, choices) {
  return (
    `Câu hỏi bài giảng:\n${question}\n\n` +
    `Các lựa chọn:\n` +
    choices.map((c, i) => `${i}. ${c}`).join('\n') +
    `\n\nHãy chọn đáp án đúng nhất (trả về JSON dạng {"index": X}):`
  );
}

export const EXAM_SYSTEM_PROMPT =
  'Bạn là chuyên gia khảo thí và học thuật cao cấp hàng đầu, có độ chính xác tuyệt đối 100% trong việc giải quyết các bài kiểm tra trắc nghiệm, đúng/sai, điền khuyết và tự luận.\n' +
  'Yêu cầu:\n' +
  '1. Phân tích cẩn thận từng câu hỏi và các lựa chọn loại trừ để chọn phương án đúng tuyệt đối.\n' +
  '2. Trả về JSONL một dòng duy nhất (hoặc JSON Array các object);\n' +
  '3. Mỗi phần tử có "so_cau" và "dap_an";\n' +
  '4. "dap_an" là A/B/C/D hoặc từ/cụm từ cần điền;\n' +
  '5. Với câu đúng/sai, "dap_an" là mảng giá trị Đúng/Sai theo thứ tự từng mệnh đề (ví dụ: [true, false, true, true]);\n' +
  '6. Tuyệt đối không thêm lời dẫn hay giải thích.';

/**
 * Đọc chỉ số đáp án từ kết quả AI: JSON {"index": X} → chữ số đứng riêng → khớp nội dung đáp án.
 * Trả về -1 nếu không xác định được.
 */
export function parseSlideIndex(rawResult, choices) {
  let index = -1;
  try {
    const matchJson = rawResult.match(/\{[\s\S]*?\}/);
    if (matchJson) {
      const parsed = JSON.parse(matchJson[0]);
      if (typeof parsed.index === 'number') {
        index = parsed.index;
      } else if (typeof parsed.index === 'string' && /^\d+$/.test(parsed.index)) {
        index = parseInt(parsed.index, 10);
      }
    }
  } catch (e) {}

  const inRange = (i) => i >= 0 && i < choices.length;

  if (!inRange(index)) {
    const digitMatch = rawResult.match(/\b([0-9])\b/);
    if (digitMatch) {
      const d = parseInt(digitMatch[1], 10);
      if (inRange(d)) index = d;
    }
  }

  if (!inRange(index)) {
    const lowerRes = rawResult.toLowerCase();
    for (let i = 0; i < choices.length; i++) {
      const lowerC = choices[i].toLowerCase();
      if (lowerC.length > 2 && (lowerRes.includes(lowerC) || lowerC.includes(lowerRes))) {
        index = i;
        break;
      }
    }
  }

  return inRange(index) ? index : -1;
}
