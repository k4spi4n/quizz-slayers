/**
 * EDUX Slayers - Popup (điểm vào)
 * Nạp cài đặt đã lưu một lần rồi khởi tạo từng tab; mỗi tab nằm trong module riêng.
 */
import { UI, addLog, setStatus, initTabNavigation } from './ui.js';
import { getActiveTab, sendTabMessage, isEduxUrl, ensureInterceptor } from './edux-tab.js';
import { initAiProfiles } from './ai-profiles.js';
import { initSlidePanel, setSlideRunningUI } from './slide-panel.js';
import { initExamPanel, updateExamInfoUI } from './exam-panel.js';
import { initScoresPanel, loadExerciseScores } from './scores-panel.js';
import { initSettingsPanel } from './settings-panel.js';
import { initUpdatePanel } from './update-panel.js';

initTabNavigation((tabId) => {
  if (tabId === 'tab-scores') loadExerciseScores();
});

const settings = await chrome.storage.local.get([
  'delayMs',
  'autoNext',
  'autoSubmit',
  'useAiSlide',
  'slideMethod',
  'layaEndpoint',
  'layaApiKey',
  'savedAnswers',
  'slideStats',
  'lastExamData',
  'apiProvider',
  'apiEndpoint',
  'apiKey',
  'apiModel',
  'aiProfiles',
  'aiAssign',
  'cachedModelsByProvider',
  'testWorkflowMode',
]);

initSettingsPanel(settings);
initSlidePanel(settings);
await initAiProfiles(settings);
initExamPanel(settings);
initScoresPanel();
initUpdatePanel();
await syncWithActiveTab();

// Đọc trạng thái hiện tại từ content script của tab EDUX (slide đang chạy? bài tập đang mở?)
async function syncWithActiveTab() {
  const activeTab = await getActiveTab();
  if (activeTab && isEduxUrl(activeTab.url)) {
    ensureInterceptor(activeTab.id);

    try {
      const response = await sendTabMessage(activeTab.id, {
        action: 'GET_STATUS',
      });
      if (response) {
        if (response.isSlideRunning) {
          setSlideRunningUI(true);
          setStatus('Đang giải Slide...', 'running');
        }
        if (response.isExamOpen && response.examData) {
          updateExamInfoUI(response.examData);
        } else {
          updateExamInfoUI(null);
          chrome.storage.local.remove('lastExamData');
        }
      }
    } catch (e) {
      addLog(UI.slideLog, 'Mở slide hoặc bài tập để bắt đầu.', 'info');
    }
  } else {
    addLog(UI.slideLog, 'Vui lòng chuyển sang trang EDUX để sử dụng.', 'warn');
  }
}
