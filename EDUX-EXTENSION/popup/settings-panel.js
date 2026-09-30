// Tab Cài đặt: delay, tự chuyển slide, tự nộp bài, chờ mỗi câu bài tập, Laya server, nút Lưu cài đặt
import { UI, addLog } from './ui.js';
import { state } from './state.js';
import { getActiveTab, sendTabMessage } from './edux-tab.js';
import { checkLayaHealth } from './slide-panel.js';

// Chờ mỗi câu khi điền bài tập (giây). Mặc định không chờ, giữ nguyên hành vi cũ.
const DEFAULT_EXAM_DELAY = { mode: 'fixed', fixed: 0, min: 3, max: 8 };

function showExamDelayMode(mode) {
  if (UI.examDelayFixedGroup)
    UI.examDelayFixedGroup.style.display = mode === 'random' ? 'none' : 'flex';
  if (UI.examDelayRandomGroup)
    UI.examDelayRandomGroup.style.display = mode === 'random' ? 'flex' : 'none';
}

// Đọc thẳng từ form (giống "Tự động nộp bài") nên có hiệu lực ngay, chưa cần bấm Lưu.
// exam-solver.js tự chặn giá trị âm / quá lớn và tự đổi chỗ nếu "từ" > "đến".
export function getExamDelayConfig() {
  const seconds = (el, fallback) => {
    const v = parseFloat(el?.value);
    return Number.isFinite(v) && v >= 0 ? v : fallback;
  };
  return {
    mode: UI.settingExamDelayMode?.value === 'random' ? 'random' : 'fixed',
    fixed: seconds(UI.settingExamDelayFixed, DEFAULT_EXAM_DELAY.fixed),
    min: seconds(UI.settingExamDelayMin, DEFAULT_EXAM_DELAY.min),
    max: seconds(UI.settingExamDelayMax, DEFAULT_EXAM_DELAY.max),
  };
}

export function initSettingsPanel(settings) {
  UI.settingDelay.value = settings.delayMs !== undefined ? settings.delayMs : 100;
  UI.settingAutoNext.checked = settings.autoNext !== undefined ? settings.autoNext : true;
  if (UI.settingAutoSubmit)
    UI.settingAutoSubmit.checked = settings.autoSubmit !== undefined ? settings.autoSubmit : true;
  if (UI.settingLayaEndpoint) UI.settingLayaEndpoint.value = settings.layaEndpoint || '';
  if (UI.settingLayaApiKey) UI.settingLayaApiKey.value = settings.layaApiKey || '';

  const examDelay = { ...DEFAULT_EXAM_DELAY, ...(settings.examQuestionDelay || {}) };
  if (UI.settingExamDelayMode) {
    UI.settingExamDelayMode.value = examDelay.mode === 'random' ? 'random' : 'fixed';
    UI.settingExamDelayMode.addEventListener('change', () =>
      showExamDelayMode(UI.settingExamDelayMode.value),
    );
  }
  if (UI.settingExamDelayFixed) UI.settingExamDelayFixed.value = examDelay.fixed;
  if (UI.settingExamDelayMin) UI.settingExamDelayMin.value = examDelay.min;
  if (UI.settingExamDelayMax) UI.settingExamDelayMax.value = examDelay.max;
  showExamDelayMode(examDelay.mode);

  if (UI.btnTestLaya) {
    UI.btnTestLaya.addEventListener('click', async () => {
      UI.btnTestLaya.disabled = true;
      const res = await checkLayaHealth();
      UI.btnTestLaya.disabled = false;
      if (!UI.layaStatus) return;
      UI.layaStatus.style.display = 'block';
      UI.layaStatus.style.color = res.success ? '#34d399' : '#f87171';
      UI.layaStatus.textContent = res.success
        ? `✅ Đã kết nối Laya${res.loaded.length ? ` — đã nạp: ${res.loaded.join(', ')}` : ''}`
        : `❌ ${res.message}`;
    });
  }

  UI.btnSaveSettings.addEventListener('click', async () => {
    // Đang mở form cấu hình AI -> lưu luôn để không mất thay đổi
    if (UI.aiProfileEditor && UI.aiProfileEditor.style.display !== 'none')
      UI.btnSaveAiProfile.click();

    const newSettings = {
      delayMs: parseInt(UI.settingDelay.value, 10) || 100,
      autoNext: UI.settingAutoNext.checked,
      autoSubmit: UI.settingAutoSubmit ? UI.settingAutoSubmit.checked : true,
      examQuestionDelay: getExamDelayConfig(),
      slideMethod: state.currentSlideMethod,
      useAiSlide: state.currentSlideMethod === 'ai',
      useAi: state.currentSlideMethod === 'ai',
      layaEndpoint: (UI.settingLayaEndpoint?.value || '').trim(),
      layaApiKey: (UI.settingLayaApiKey?.value || '').trim(),
      cachedModelsByProvider: state.cachedModelsByProvider,
    };

    await chrome.storage.local.set(newSettings);

    const tab = await getActiveTab();
    if (tab) {
      sendTabMessage(tab.id, {
        action: 'UPDATE_SETTINGS',
        settings: newSettings,
      }).catch(() => {});
    }

    addLog(UI.slideLog, 'Đã lưu cấu hình mới!', 'success');
    addLog(UI.testLog, 'Đã cập nhật cấu hình bài tập & AI!', 'success');
  });
}
