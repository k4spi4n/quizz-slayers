// Tab Slide: chọn phương pháp (AI / Laya / Bruteforce), bắt đầu / dừng, nhận log tiến độ từ trang
import { UI, addLog, setStatus } from './ui.js';
import { state } from './state.js';
import { getActiveTab, sendTabMessage } from './edux-tab.js';
import { getAssignedProfile, onProfilesChange } from './ai-profiles.js';
import { isProfileReady } from '../shared/providers.js';

function slideAiDetail() {
  const p = getAssignedProfile('slide');
  return p ? 'Chính xác nhất, chậm nhất.' : 'Chính xác nhất, chậm nhất. Cần thêm cấu hình AI.';
}

export function setSlideMethod(method, saveToStorage = true) {
  if (!['ai', 'laya', 'bruteforce'].includes(method)) method = 'ai';
  state.currentSlideMethod = method;
  const isAi = method === 'ai';
  if (UI.btnSlideMethodAi) UI.btnSlideMethodAi.classList.toggle('active', isAi);
  if (UI.btnSlideMethodLaya) UI.btnSlideMethodLaya.classList.toggle('active', method === 'laya');
  if (UI.btnSlideMethodBrute)
    UI.btnSlideMethodBrute.classList.toggle('active', method === 'bruteforce');

  if (UI.settingSlideMethod) UI.settingSlideMethod.value = method;
  if (UI.slideProfileStrip) UI.slideProfileStrip.style.display = isAi ? '' : 'none';

  if (method === 'laya') {
    if (UI.slideMethodIcon) UI.slideMethodIcon.textContent = '🎯';
    if (UI.slideMethodTitle) {
      UI.slideMethodTitle.textContent = 'Laya';
      UI.slideMethodTitle.style.color = '#5eead4';
    }
    if (UI.slideMethodDesc) {
      UI.slideMethodDesc.style.borderColor = 'rgba(45, 212, 191, 0.3)';
      UI.slideMethodDesc.style.background = 'rgba(45, 212, 191, 0.08)';
    }
    if (UI.slideMethodDetail) {
      UI.slideMethodDetail.textContent =
        '🧪 Thử nghiệm. Cân bằng tốc độ & độ chính xác, cần laya-serve.';
    }
  } else if (isAi) {
    if (UI.slideMethodIcon) UI.slideMethodIcon.textContent = '🧠';
    if (UI.slideMethodTitle) {
      UI.slideMethodTitle.textContent = 'AI';
      UI.slideMethodTitle.style.color = '#c7d2fe';
    }
    if (UI.slideMethodDesc) {
      UI.slideMethodDesc.style.borderColor = 'rgba(99, 102, 241, 0.3)';
      UI.slideMethodDesc.style.background = 'rgba(99, 102, 241, 0.12)';
    }
    if (UI.slideMethodDetail) {
      UI.slideMethodDetail.textContent = slideAiDetail();
    }
  } else {
    if (UI.slideMethodIcon) UI.slideMethodIcon.textContent = '⚡';
    if (UI.slideMethodTitle) {
      UI.slideMethodTitle.textContent = 'Bruteforce';
      UI.slideMethodTitle.style.color = '#fbbf24';
    }
    if (UI.slideMethodDesc) {
      UI.slideMethodDesc.style.borderColor = 'rgba(251, 191, 36, 0.3)';
      UI.slideMethodDesc.style.background = 'rgba(251, 191, 36, 0.08)';
    }
    if (UI.slideMethodDetail) {
      UI.slideMethodDetail.textContent = 'Nhanh nhất, thử lần lượt. Không cần gì.';
    }
  }

  if (saveToStorage) {
    chrome.storage.local.set({ slideMethod: method, useAiSlide: isAi });
  }
}

// Laya đọc endpoint/key từ storage trong background -> lưu trước rồi mới ping /health
export async function checkLayaHealth() {
  await chrome.storage.local.set({
    layaEndpoint: (UI.settingLayaEndpoint?.value || '').trim(),
    layaApiKey: (UI.settingLayaApiKey?.value || '').trim(),
  });
  try {
    const res = await chrome.runtime.sendMessage({ action: 'LAYA_HEALTH' });
    return res || { success: false, message: 'Không nhận được phản hồi' };
  } catch (err) {
    return { success: false, message: err.message };
  }
}

export function setSlideRunningUI(isRunning) {
  UI.btnStartSlide.classList.toggle('hidden', isRunning);
  UI.btnStopSlide.classList.toggle('hidden', !isRunning);
}

export function initSlidePanel(settings) {
  setSlideMethod(
    settings.slideMethod || (settings.useAiSlide === false ? 'bruteforce' : 'ai'),
    false,
  );

  if (UI.btnSlideMethodAi) {
    UI.btnSlideMethodAi.addEventListener('click', () => setSlideMethod('ai'));
  }
  if (UI.btnSlideMethodLaya) {
    UI.btnSlideMethodLaya.addEventListener('click', () => setSlideMethod('laya'));
  }
  if (UI.btnSlideMethodBrute) {
    UI.btnSlideMethodBrute.addEventListener('click', () => setSlideMethod('bruteforce'));
  }
  if (UI.settingSlideMethod) {
    UI.settingSlideMethod.addEventListener('change', () => {
      setSlideMethod(UI.settingSlideMethod.value);
    });
  }

  if (settings.slideStats) {
    UI.slideCount.textContent = settings.slideStats.solved || 0;
    UI.retryCount.textContent = settings.slideStats.retries || 0;
  }

  // Mô tả chế độ AI phụ thuộc việc đã có cấu hình AI cho Slide chưa
  onProfilesChange(() => {
    if (state.currentSlideMethod === 'ai' && UI.slideMethodDetail)
      UI.slideMethodDetail.textContent = slideAiDetail();
  });

  UI.btnStartSlide.addEventListener('click', async () => {
    const tab = await getActiveTab();
    if (!tab) return;

    let slideMethod = state.currentSlideMethod;

    if (slideMethod === 'ai' && !isProfileReady(getAssignedProfile('slide'))) {
      addLog(UI.slideLog, '⚠️ Chưa có cấu hình AI cho Slide, chuyển sang Bruteforce.', 'warn');
      setSlideMethod('bruteforce');
      slideMethod = 'bruteforce';
    }

    if (slideMethod === 'laya') {
      const health = await checkLayaHealth();
      if (!health.success) {
        addLog(
          UI.slideLog,
          `⚠️ ${health.message} Vẫn chạy nhưng sẽ thử sai tuần tự cho tới khi Laya sẵn sàng.`,
          'warn',
        );
      }
    }
    const useAi = slideMethod === 'ai';

    const origHtml = UI.btnStartSlide.innerHTML;
    UI.btnStartSlide.innerHTML = '<span class="spinner spinner-sm"></span> Đang khởi động...';
    UI.btnStartSlide.disabled = true;

    try {
      await sendTabMessage(tab.id, {
        action: 'START_SLIDE_BRUTEFORCE',
        config: {
          delayMs: parseInt(UI.settingDelay.value, 10) || 100,
          autoNext: UI.settingAutoNext.checked,
          useAi,
          slideMethod,
        },
      });
      UI.btnStartSlide.classList.add('hidden');
      UI.btnStopSlide.classList.remove('hidden');
      setStatus('Đang giải Slide...', 'running');
      addLog(
        UI.slideLog,
        {
          ai: '🧠 Bắt đầu: AI',
          laya: '🎯 Bắt đầu: Laya',
          bruteforce: '⚡ Bắt đầu: Bruteforce',
        }[slideMethod],
        'success',
      );
    } catch (err) {
      addLog(UI.slideLog, 'Lỗi kết nối với trang EDUX: ' + err.message, 'error');
    } finally {
      UI.btnStartSlide.innerHTML = origHtml;
      UI.btnStartSlide.disabled = false;
    }
  });

  UI.btnStopSlide.addEventListener('click', async () => {
    const tab = await getActiveTab();
    if (!tab) return;
    try {
      await sendTabMessage(tab.id, { action: 'STOP_SLIDE_BRUTEFORCE' });
      UI.btnStartSlide.classList.remove('hidden');
      UI.btnStopSlide.classList.add('hidden');
      setStatus('Đã dừng', 'stopped');
      addLog(UI.slideLog, 'Đã dừng giải Slide.', 'warn');
    } catch (err) {
      addLog(UI.slideLog, 'Không thể dừng tiến trình.', 'error');
    }
  });

  // Log & trạng thái gửi về từ slide-solver trên trang
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === 'SLIDE_LOG') {
      addLog(UI.slideLog, msg.message, msg.logType || 'info');
      if (msg.solvedCount !== undefined) UI.slideCount.textContent = msg.solvedCount;
      if (msg.retryCount !== undefined) UI.retryCount.textContent = msg.retryCount;
    } else if (msg.type === 'SLIDE_STATUS_CHANGE') {
      setSlideRunningUI(msg.isRunning);
      if (msg.isRunning) setStatus('Đang giải Slide...', 'running');
      else setStatus('Sẵn sàng', 'idle');
    }
  });
}
