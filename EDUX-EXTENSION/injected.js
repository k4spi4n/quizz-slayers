/**
 * EDUX Slayers - Injected Network Interceptor v2.3.0
 * Chạy trong execution context của trang (MAIN world) để bắt toàn bộ API request/response
 * (đặc biệt là request /start và exam questions payload) tương tự Playwright page.expect_response().
 */
(function () {
  if (window.__EDUX_SLAYERS_INTERCEPTOR_ACTIVE__) return;
  window.__EDUX_SLAYERS_INTERCEPTOR_ACTIVE__ = true;

  console.log('⚔️ [EDUX Slayers Interceptor] Đã kích hoạt bộ lắng nghe gói tin mạng.');

  // =========================================================================
  // 0. Clock Drift Auto-Synchronization (Chống lỗi 'Request expired' do lệch giờ PC)
  // =========================================================================
  const OriginalDate = window.Date;
  const originalDateNow = OriginalDate.now.bind(OriginalDate);
  let serverDriftMs = 0;

  function syncServerTime(dateHeader) {
    if (!dateHeader) return;
    try {
      const serverTime = new OriginalDate(dateHeader).getTime();
      if (!isNaN(serverTime) && serverTime > 0) {
        const clientTime = originalDateNow();
        const drift = serverTime - clientTime;
        if (Math.abs(drift) > 3000) {
          serverDriftMs = drift;
          window.__EDUX_SERVER_DRIFT__ = drift;
          if (!Date.__edux_clock_patched__) {
            Date.__edux_clock_patched__ = true;
            Date.now = function () {
              return originalDateNow() + (window.__EDUX_SERVER_DRIFT__ || 0);
            };
            console.log(
              `[EDUX Slayers Interceptor] ⏱️ Đã bù trừ lệch giờ ${Math.round(drift / 1000)}s so với server để chống lỗi 'Request expired'.`
            );
          }
        }
      }
    } catch (e) {}
  }

  try {
    fetch('/api/api/v1/sessions/test?_sync=' + originalDateNow(), { cache: 'no-store' })
      .then((r) => syncServerTime(r.headers.get('date')))
      .catch(() => {});
  } catch (e) {}

  function broadcastEvent(type, data, sourceUrl) {
    try {
      window.postMessage(
        {
          type,
          url: sourceUrl,
          payload: data,
          timestamp: Date.now()
        },
        '*'
      );
      console.log(`[EDUX Slayers Interceptor] 📡 Phát sự kiện ${type} từ:`, sourceUrl);
    } catch (e) {
      // Ignore serialization issues
    }
  }

  function broadcastExamPayload(data, sourceUrl) {
    try {
      // Lưu vào sessionStorage để bất kỳ script nào (kể cả Content Script) có thể truy cập tức thì
      sessionStorage.setItem('__EDUX_LAST_EXAM_DATA__', JSON.stringify(data));
      sessionStorage.setItem('__EDUX_LAST_EXAM_URL__', sourceUrl);
      window.__EDUX_LAST_EXAM_DATA__ = data;
      window.__EDUX_LAST_EXAM_URL__ = sourceUrl;
    } catch (e) {}

    broadcastEvent('EDUX_EXAM_DATA_CAPTURED', data, sourceUrl);
  }

  function isExamPayload(data, url) {
    if (!data || typeof data !== 'object') return false;
    const urlLower = (url || '').toLowerCase();

    // Loại trừ các API lịch sử, models môn học, danh sách tham gia hoặc submit bài
    if (
      urlLower.includes('/history') ||
      urlLower.includes('/models') ||
      urlLower.includes('/joined') ||
      urlLower.includes('/submit')
    ) {
      return false;
    }

    // 1. Endpoint chứa 'start', 'take', 'begin' (Chuẩn theo Playwright EDUX-TEST-SOLVER: "start" in resp.url)
    if (urlLower.includes('start') || urlLower.includes('/take') || urlLower.includes('/begin')) {
      return true;
    }

    const d = data.data || data;
    if (!d || typeof d !== 'object') return false;

    // 2. Cấu trúc chứa exam_data
    const examData = d.exam_data || d;
    if (examData && typeof examData === 'object') {
      const hasMc = Array.isArray(examData.multiple_choice) && examData.multiple_choice.length > 0;
      const hasTf = Array.isArray(examData.true_false) && examData.true_false.length > 0;
      const hasFill = Array.isArray(examData.fill_in_blank) && examData.fill_in_blank.length > 0;
      const hasEssay = Array.isArray(examData.essay) && examData.essay.length > 0;
      const hasQuestions = Array.isArray(examData.questions) && examData.questions.length > 0;
      if (hasMc || hasTf || hasFill || hasEssay || hasQuestions) {
        return true;
      }
    }

    // 3. Mảng câu hỏi trực tiếp
    if (Array.isArray(d) && d.length > 0 && (d[0].question || d[0].so_cau || d[0].options)) {
      return true;
    }

    // 4. Chứa total_questions > 0 hoặc title bài kiểm tra
    if ((d.total_questions > 0 || data.total_questions > 0) && (d.title || data.title || d.exam_data)) {
      return true;
    }

    return false;
  }

  function checkAndBroadcast(data, url) {
    if (!data || typeof data !== 'object') return;
    const urlLower = (url || '').toLowerCase();

    // 1. Kiểm tra API danh sách môn học đã tham gia
    if (urlLower.includes('/api/subjects/joined')) {
      broadcastEvent('EDUX_SUBJECTS_JOINED_CAPTURED', data, url);
      return;
    }

    // 2. Kiểm tra API danh sách bài học / models môn học
    if (urlLower.includes('/api/subjects/') && urlLower.includes('/models')) {
      broadcastEvent('EDUX_MODELS_DATA_CAPTURED', data, url);
      return;
    }

    // 3. Kiểm tra API lịch sử làm bài
    if (urlLower.includes('/api/exam/history')) {
      broadcastEvent('EDUX_EXAM_HISTORY_CAPTURED', data, url);
      return;
    }

    // 4. Kiểm tra gói tin đề bài tập (/start hoặc payload chứa questions)
    if (isExamPayload(data, url)) {
      console.log('[EDUX Slayers Interceptor] 🎯 ĐÃ BẮT ĐƯỢC GÓI TIN ĐỀ BÀI TẬP từ:', url);
      broadcastExamPayload(data, url);
    }
  }

  // =========================================================================
  // 1. Intercept window.fetch
  // =========================================================================
  const originalFetch = window.fetch;
  if (originalFetch) {
    window.fetch = async function (...args) {
      const response = await originalFetch.apply(this, args);
      try {
        const resolvedUrl =
          (response && response.url) ||
          (typeof args[0] === 'string' ? args[0] : (args[0] && args[0].url) || '');

        if (response && response.headers) {
          syncServerTime(response.headers.get('date'));
        }

        const clone = response.clone();
        clone
          .text()
          .then((text) => {
            if (!text) return;
            try {
              const json = JSON.parse(text);
              checkAndBroadcast(json, resolvedUrl);
            } catch (e) {}
          })
          .catch(() => {});
      } catch (e) {
        // Bỏ qua lỗi clone
      }
      return response;
    };
  }

  // =========================================================================
  // 2. Intercept XMLHttpRequest (Hỗ trợ an toàn cho cả responseType = 'json')
  // =========================================================================
  const originalOpen = XMLHttpRequest.prototype.open;
  const originalSend = XMLHttpRequest.prototype.send;

  XMLHttpRequest.prototype.open = function (method, url, ...rest) {
    this._edux_url = url;
    return originalOpen.apply(this, [method, url, ...rest]);
  };

  XMLHttpRequest.prototype.send = function (...args) {
    const handleResponse = () => {
      try {
        syncServerTime(this.getResponseHeader('Date'));
        const url = this.responseURL || this._edux_url || '';
        let json = null;

        // Nếu responseType là 'json', truy cập responseText sẽ gây DOMException trong Chrome
        if (this.responseType === 'json' && this.response) {
          json = this.response;
        } else if (!this.responseType || this.responseType === 'text') {
          if (this.responseText) {
            json = JSON.parse(this.responseText);
          }
        } else if (this.response && typeof this.response === 'object') {
          json = this.response;
        }

        if (json) {
          checkAndBroadcast(json, url);
        }
      } catch (e) {
        // Parsing error or unsupported type
      }
    };

    this.addEventListener('load', handleResponse);
    return originalSend.apply(this, args);
  };
})();
