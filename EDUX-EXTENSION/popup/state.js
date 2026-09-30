// Trạng thái dùng chung giữa các module popup (thay cho biến trong closure cũ)
export const state = {
  // [{ id, provider, endpoint, apiKey, model, reasoningEffort }]
  aiProfiles: [],
  // { slide: profileId, exam: profileId }
  aiAssign: {},
  // Model tải từ server theo provider: { openai: ["gpt-4o", ...] }
  cachedModelsByProvider: {},
  // Cấu hình đang mở trong form sửa (null = thêm mới)
  editingProfileId: null,
  currentSlideMethod: "ai",
};
