// Khóa chrome.storage.local. Đổi tên khóa = mất cấu hình của người dùng khi cập nhật, nên chỉ thêm mới.

// Giá trị mặc định ghi khi cài lần đầu (không ghi đè giá trị đã có)
export const DEFAULT_SETTINGS = {
  delayMs: 100,
  autoNext: true,
  slideStats: { solved: 0, retries: 0 }
};

// Chỉ sao lưu cấu hình người dùng — bỏ qua dữ liệu tạm (đề, đáp án, cache điểm, thống kê)
export const BACKUP_KEYS = [
  'aiProfiles',
  'aiAssign',
  'delayMs',
  'autoNext',
  'autoSubmit',
  'slideMethod',
  'useAiSlide',
  'useAi',
  'layaEndpoint',
  'layaApiKey',
  'testWorkflowMode',
  'cachedModelsByProvider'
];
