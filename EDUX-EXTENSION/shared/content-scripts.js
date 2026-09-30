// Thứ tự nạp content scripts: file sau dùng window.Edux* của file trước.
// manifest.json không import được file này — tests/providers.test.js kiểm tra hai nơi luôn khớp.
export const CONTENT_SCRIPTS = [
  'content/dom-utils.js',
  'content/answer-parser.js',
  'content/exam-dom.js',
  'content/exam-prompt.js',
  'content/exam-solver.js',
  'content/slide-solver.js',
  'content/score-tracker.js',
  'content/content.js'
];

// Chạy trong MAIN world của trang EDUX để nghe dữ liệu mạng (đề bài tập, giờ server)
export const INJECTED_SCRIPT = 'content/injected.js';
