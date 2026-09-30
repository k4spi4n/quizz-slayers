// Golden tests for the Bài tập text pipeline: AI response → answers map, and exam payload → prompt.
// Snapshots were recorded from v2.5.0 behavior; update them only for intentional changes
// (node --test --test-update-snapshots tests/).
import { test } from 'node:test';
import { loadExamApi, plain } from './helpers/extension.js';

const exam = loadExamApi();

const AI_RESPONSES = {
  jsonl_array:
    '[{"so_cau": 1, "dap_an": "A"}, {"so_cau": 2, "dap_an": "C"}, {"so_cau": 3, "dap_an": [true, false, true, true]}]',
  fenced_json:
    'Đây là đáp án:\n```json\n[{"so_cau": 1, "dap_an": "B"}, {"so_cau": 2, "dap_an": "D"}]\n```\nChúc bạn thi tốt!',
  object_map: '{"1": "A", "2": "B", "3": {"1": true, "2": false}}',
  wrapped_answers_key:
    '{"answers": [{"cau": 4, "answer": "C"}, {"question": 5, "ans": "hai mươi"}]}',
  double_encoded: JSON.stringify(JSON.stringify([{ so_cau: 1, dap_an: 'A' }])),
  quote_wrapped: '"[{"so_cau": 1, "dap_an": "A"}]"',
  bom_prefixed: '\ufeff[{"so_cau": 7, "dap_an": "B"}]',
  trailing_comma: '[{"so_cau": 1, "dap_an": "A"}, {"so_cau": 2, "dap_an": "B"},]',
  truncated:
    '[{"so_cau": 1, "dap_an": "A"}, {"so_cau": 2, "dap_an": "B"}, {"so_cau": 3, "dap_an": "C',
  unescaped_quotes: '[{"so_cau": 1, "dap_an": "câu "đúng" nhất"}, {"so_cau": 2, "dap_an": "B"}]',
  concatenated_objects: '{"so_cau": 1, "dap_an": "A"}{"so_cau": 2, "dap_an": "B"}',
  jsonl_lines:
    '{"so_cau": 1, "dap_an": "A"}\n{"so_cau": 2, "dap_an": "B"}\n{"so_cau": 3, "dap_an": "C"}',
  plain_lines: '1. A\n2: B\nCâu 3: Đúng, Sai, Đúng\n4 - C\n5) D',
  field_aliases:
    '[{"stt": 1, "tra_loi": "A"}, {"id": "2", "da": "B"}, {"q": 3, "a": "C"}, {"soCau": 4, "dapAn": "D"}]',
  invalid_items:
    '[{"so_cau": "x", "dap_an": "A"}, {"so_cau": 2}, null, "text", {"so_cau": 3, "dap_an": "C"}]',
  empty: '',
  prose_only: 'Tôi không chắc về đáp án của các câu hỏi này.',
};

for (const [name, raw] of Object.entries(AI_RESPONSES)) {
  test(`loadAnswersFromInput: ${name}`, (t) => {
    t.assert.snapshot(plain(exam.loadAnswersFromInput(raw)));
  });
}

test('sanitizeAiResponse strips fences, BOM and wrapping quotes', (t) => {
  t.assert.snapshot(
    [
      '```json\n[1,2]\n```',
      '```\n{"a":1}\n```',
      '\ufeff  [1] ',
      '"[{\\"so_cau\\":1}]"',
      "'[1]'",
      '"[{"so_cau": 1}]"',
      'no fences here',
    ].map((s) => exam.sanitizeAiResponse(s)),
  );
});

const TRUE_FALSE_VALUES = {
  booleans: [[true, false, true, true], 4],
  strings_vi: [['Đúng', 'Sai', 'đ', 's'], 4],
  numbers: [[1, 0, '1', '0'], 4],
  object_keyed: [{ 2: 'sai', 1: 'đúng', 10: 'true', 3: 'false' }, 4],
  json_string: ['[true, false, false, true]', 4],
  numbered_tokens: ['1. Đúng, 2. Sai, 3. Đúng, 4. Sai', 4],
  free_tokens: ['Đúng Sai Đúng Đúng Sai', 4],
  letters: ['D S D S', 4],
  english: ['true false TRUE', 3],
  truncates_to_expected: [[true, true, true, true, true], 2],
  empty: ['', 4],
};

for (const [name, [value, count]] of Object.entries(TRUE_FALSE_VALUES)) {
  test(`parseTrueFalseAnswers: ${name}`, (t) => {
    t.assert.snapshot(plain(exam.parseTrueFalseAnswers(value, count)));
  });
}

const EXAM_PAYLOADS = {
  full_sections: {
    data: {
      title: 'Kiểm tra chương 1',
      exam_data: {
        multiple_choice: [
          {
            id: 1,
            question: 'Thủ đô Việt Nam?',
            options: { A: 'Hà Nội', B: 'Huế', C: 'Đà Nẵng', D: 'TP.HCM' },
          },
          { id: 2, title: '2 + 2 = ?', choices: { A: '3', B: '4' } },
        ],
        fill_in_blank: [{ so_cau: 3, question: 'Nước sôi ở ___ độ C.' }],
        essay: [{ content: 'Trình bày vai trò của AI.' }],
        true_false: [
          {
            id: 5,
            question: 'Xét các mệnh đề sau:',
            statements: [
              'Trái đất tròn',
              { text: 'Mặt trời quay quanh Trái đất' },
              { statement: 'Nước là H2O' },
            ],
          },
        ],
      },
    },
  },
  colliding_ids: {
    data: {
      title: 'Trùng ID',
      exam_data: {
        multiple_choice: [
          { id: 1, question: 'MC 1', options: { A: 'x', B: 'y' } },
          { id: 2, question: 'MC 2', options: { A: 'x', B: 'y' } },
        ],
        true_false: [
          { id: 1, question: 'TF restarts at 1', items: ['a', 'b'] },
          { id: 2, question: 'TF 2', sub_questions: [{ content: 'c' }] },
        ],
      },
    },
  },
  generic_questions_array: {
    data: {
      exam_data: {
        questions: [
          {
            question_type: 'single_choice',
            question_text: 'Chọn một',
            options: { A: '1', B: '2' },
          },
          { type: 'true_false', question: 'Đúng hay sai', statements: ['p', 'q'] },
          { type: 'essay', title: 'Viết đoạn văn' },
          { type: 'short_answer', question: 'Điền từ' },
        ],
      },
    },
  },
  bare_array: [
    { question: 'Q1', options: { A: 'a' } },
    { question: 'Q2', options: { A: 'b' } },
  ],
  missing_title_uses_document_title: { data: { exam_data: { essay: [{ question: 'E1' }] } } },
  empty: {},
};

for (const [name, payload] of Object.entries(EXAM_PAYLOADS)) {
  test(`buildCompactPromptPayload: ${name}`, (t) => {
    t.assert.snapshot(plain(exam.buildCompactPromptPayload(payload)));
  });
}

test('generateStandardPromptText', (t) => {
  t.assert.snapshot(
    exam.generateStandardPromptText(exam.buildCompactPromptPayload(EXAM_PAYLOADS.full_sections)),
  );
});
