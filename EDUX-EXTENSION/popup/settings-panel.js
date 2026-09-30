// Tab Cài đặt: delay, tự chuyển slide, tự nộp bài, Laya server, nút Lưu cài đặt
import { UI, addLog } from './ui.js';
import { state } from './state.js';
import { getActiveTab, sendTabMessage } from './edux-tab.js';
import { checkLayaHealth } from './slide-panel.js';

export function initSettingsPanel(settings) {
  UI.settingDelay.value = settings.delayMs !== undefined ? settings.delayMs : 100;
  UI.settingAutoNext.checked = settings.autoNext !== undefined ? settings.autoNext : true;
  if (UI.settingAutoSubmit)
    UI.settingAutoSubmit.checked = settings.autoSubmit !== undefined ? settings.autoSubmit : true;
  if (UI.settingLayaEndpoint) UI.settingLayaEndpoint.value = settings.layaEndpoint || '';
  if (UI.settingLayaApiKey) UI.settingLayaApiKey.value = settings.layaApiKey || '';

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
