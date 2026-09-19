/**
 * EDUX Slayers Content Script v2.2.0
 * Modular coordinator: connects Network Interceptor, Slide Solver,
 * Test Solver (Bài tập), and Score Tracker.
 */

(function () {
  window.__EDUX_SLAYERS_INJECTED__ = true;

  console.log('⚔️ EDUX Slayers Content Script v2.2.0 loaded.');

  // =========================================================================
  // 1. Network Interceptor Setup & Shared Session Cache
  // =========================================================================
  let lastCapturedExamData = null;

  // Đọc dữ liệu đề thi đã lưu từ sessionStorage (nếu trang đã tải trước khi content script chạy)
  try {
    const stored = sessionStorage.getItem('__EDUX_LAST_EXAM_DATA__');
    if (stored) {
      lastCapturedExamData = JSON.parse(stored);
      window.EduxTestSolver?.setCapturedExamData(lastCapturedExamData);
      chrome.storage.local.set({ lastExamData: lastCapturedExamData });
    }
  } catch (e) {}

  function injectNetworkInterceptor() {
    try {
      const script = document.createElement('script');
      script.src = chrome.runtime.getURL('injected.js');
      script.onload = () => script.remove();
      (document.head || document.documentElement).appendChild(script);
    } catch (e) {
      console.warn('[EDUX Slayers] Could not inject API interceptor:', e);
    }
  }
  injectNetworkInterceptor();

  window.addEventListener('message', (event) => {
    if (event.source !== window || !event.data) return;
    if (event.data.type === 'EDUX_EXAM_DATA_CAPTURED') {
      lastCapturedExamData = event.data.payload;
      window.EduxTestSolver?.setCapturedExamData(lastCapturedExamData);
      try {
        sessionStorage.setItem('__EDUX_LAST_EXAM_DATA__', JSON.stringify(lastCapturedExamData));
        chrome.storage.local.set({ lastExamData: lastCapturedExamData });
        chrome.runtime.sendMessage({
          type: 'EXAM_DATA_READY',
          payload: lastCapturedExamData
        }).catch(() => {});
      } catch (e) {}
      console.log('[EDUX Slayers] 📡 Đã tự động bắt được dữ liệu bài tập từ máy chủ EDUX!');
    } else if (event.data.type === 'EDUX_MODELS_DATA_CAPTURED') {
      const payload = event.data.payload;
      if (payload && Array.isArray(payload.data)) {
        window.EduxScoreTracker?.setCachedModels(payload.data);
      }
    } else if (event.data.type === 'EDUX_SUBJECTS_JOINED_CAPTURED') {
      window.EduxScoreTracker?.setJoinedSubjects(event.data.payload);
    } else if (event.data.type === 'EDUX_EXAM_HISTORY_CAPTURED') {
      window.EduxScoreTracker?.refresh();
    }
  });

  // =========================================================================
  // 2. Extension Message Handlers
  // =========================================================================
  chrome.storage.local.get(['delayMs', 'autoNext', 'useAiSlide'], (res) => {
    if (res && window.EduxSlideSolver) {
      window.EduxSlideSolver.setConfig({
        delayMs: res.delayMs !== undefined ? res.delayMs : 100,
        autoNext: res.autoNext !== undefined ? res.autoNext : true,
        useAi: res.useAiSlide !== undefined ? res.useAiSlide : true
      });
    }
  });

  chrome.runtime.onMessage.addListener((req, sender, sendResponse) => {
    if (req.action === 'START_SLIDE_BRUTEFORCE') {
      window.EduxSlideSolver?.start(req.config);
      sendResponse({ success: true });
    } else if (req.action === 'STOP_SLIDE_BRUTEFORCE') {
      window.EduxSlideSolver?.stop();
      sendResponse({ success: true });
    } else if (req.action === 'START_EXERCISE') {
      if (window.EduxTestSolver) {
        window.EduxTestSolver.startExercise()
          .then((res) => {
            if (res && res.questions) {
              lastCapturedExamData = lastCapturedExamData || res.questions;
            }
            sendResponse(res);
          })
          .catch((err) => sendResponse({ success: false, message: String(err) }));
        return true;
      } else {
        sendResponse({ success: false, message: 'Động cơ giải bài tập chưa sẵn sàng.' });
      }
    } else if (req.action === 'FILL_TEST_ANSWERS') {
      if (window.EduxTestSolver) {
        window.EduxTestSolver.fillTestAnswers(req.answersText, req.options)
          .then((res) => sendResponse(res))
          .catch((err) => sendResponse({ success: false, message: String(err) }));
        return true;
      } else {
        sendResponse({ success: false, message: 'Động cơ giải bài tập chưa sẵn sàng.' });
      }
    } else if (req.action === 'EXTRACT_QUESTIONS') {
      if (window.EduxTestSolver) {
        if (!lastCapturedExamData) {
          try {
            const stored = sessionStorage.getItem('__EDUX_LAST_EXAM_DATA__');
            if (stored) {
              lastCapturedExamData = JSON.parse(stored);
              window.EduxTestSolver.setCapturedExamData(lastCapturedExamData);
            }
          } catch (e) {}
        }
        const res = window.EduxTestSolver.extractQuestions(lastCapturedExamData);
        if (res && res.questions) {
          lastCapturedExamData = lastCapturedExamData || res.questions;
          chrome.storage.local.set({ lastExamData: lastCapturedExamData });
        }
        sendResponse(res);
      } else {
        sendResponse({ success: false, message: 'Động cơ giải bài tập chưa sẵn sàng.' });
      }
    } else if (req.action === 'CHECK_EXAM_OPEN') {
      const dialog = window.EduxTestSolver?.getActiveExamDialog();
      sendResponse({ isOpen: !!dialog });
    } else if (req.action === 'GET_STATUS') {
      const status = window.EduxSlideSolver?.getStatus() || { isSlideRunning: false, solvedCount: 0, retryCount: 0 };
      const isExamOpen = !!window.EduxTestSolver?.getActiveExamDialog();

      let examData = null;
      if (isExamOpen) {
        examData = lastCapturedExamData || window.EduxTestSolver?.getCapturedExamData();
        if (!examData) {
          try {
            const stored = sessionStorage.getItem('__EDUX_LAST_EXAM_DATA__');
            if (stored) {
              examData = JSON.parse(stored);
              lastCapturedExamData = examData;
              window.EduxTestSolver?.setCapturedExamData(examData);
            }
          } catch (e) {}
        }
      }

      sendResponse({
        ...status,
        isExamOpen,
        examData
      });
    } else if (req.action === 'UPDATE_SETTINGS') {
      window.EduxSlideSolver?.setConfig(req.settings);
      sendResponse({ success: true });
    } else if (req.action === 'GET_EXERCISE_SCORES') {
      if (window.EduxScoreTracker) {
        window.EduxScoreTracker.getScores()
          .then((res) => sendResponse(res))
          .catch((err) => sendResponse({ success: false, error: String(err) }));
        return true;
      } else {
        sendResponse({ success: false, message: 'Động cơ Score Tracker chưa sẵn sàng.' });
      }
    } else if (req.action === 'GET_ALL_SUBJECTS_PROGRESS') {
      if (window.EduxScoreTracker) {
        window.EduxScoreTracker.getAllSubjectsProgress()
          .then((res) => sendResponse(res))
          .catch((err) => sendResponse({ success: false, error: String(err) }));
        return true;
      } else {
        sendResponse({ success: false, message: 'Động cơ Score Tracker chưa sẵn sàng.' });
      }
    }
    return true;
  });
})();
