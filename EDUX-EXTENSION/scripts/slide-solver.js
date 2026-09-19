/**
 * EDUX Slayers - Slide Brute-force Engine
 * Automates lecture slides with intelligent trial & error and auto-transition
 */

(function () {
  'use strict';

  const { sleep, safeIsVisible, safeIsEnabled, safeClick, getActiveDialog, waitForHidden } = window.EduxDOM;

  // State & Stats
  let isSlideRunning = false;
  let slideLoopTimer = null;
  let solvedCount = 0;
  let retryCount = 0;
  let wrongAnswersMap = {};
  let knownCorrectAnswers = {};
  let aiAttemptedMap = {};
  let lastProgress = Date.now();
  let stallReported = false;
  const STALL_MS = 8000;

  let config = {
    delayMs: 100,
    autoNext: true,
    useAi: true
  };

  /**
   * Gửi yêu cầu giải câu hỏi slide tới Background AI Service Worker và CHỜ phản hồi
   */
  async function queryAiForSlideAnswer(questionText, choices) {
    return new Promise((resolve) => {
      let timeoutId = setTimeout(() => {
        resolve({ success: false, message: 'AI phản hồi quá lâu (hết thời gian chờ 15s)' });
      }, 15000);

      try {
        chrome.runtime.sendMessage(
          {
            action: 'AI_SOLVE_SLIDE',
            question: questionText,
            choices: choices
          },
          (response) => {
            clearTimeout(timeoutId);
            if (chrome.runtime.lastError) {
              resolve({ success: false, message: chrome.runtime.lastError.message });
            } else {
              resolve(response || { success: false, message: 'Không nhận được phản hồi' });
            }
          }
        );
      } catch (err) {
        clearTimeout(timeoutId);
        resolve({ success: false, message: err.message });
      }
    });
  }

  function notifyPopup(type, payload) {
    try {
      chrome.runtime.sendMessage({ type, ...payload });
    } catch (e) {}
  }

  function logMessage(msg, logType = 'info') {
    console.log('[EDUX Slayers Slide] ' + msg);
    notifyPopup('SLIDE_LOG', {
      message: msg,
      logType,
      solvedCount,
      retryCount
    });
  }

  function getDialogNextPageButton() {
    const dialog = getActiveDialog();
    const searchRoots = dialog ? [dialog, document] : [document];

    for (const root of searchRoots) {
      const candidates = Array.from(
        root.querySelectorAll(
          "button, a[role='button'], div[role='button'], [role='button'], div.cursor-pointer, span.cursor-pointer"
        )
      );

      // Pass 1: Buttons with green styling AND completion text
      for (const btn of candidates) {
        if (!safeIsVisible(btn) || !safeIsEnabled(btn)) continue;
        if (btn.closest('nav, aside, .sidebar, [class*="sidebar"], ul, ol')) continue;

        const txt = (btn.textContent || '').trim();
        const title = (btn.getAttribute('title') || '').trim();
        const aria = (btn.getAttribute('aria-label') || '').trim();

        // Never match slide numbers (e.g. "1", "2", "1 / 58")
        if (/^\d+(\s*\/\s*\d+)?$/.test(txt)) continue;

        // Never match navigation buttons
        if (txt.includes('Bài giảng') || title.includes('Bài giảng') || txt.includes('Khóa học')) continue;
        if (txt.includes('Thử lại') || txt.includes('Bỏ qua') || txt.includes('Phản hồi') || txt.includes('Đổi câu hỏi')) continue;

        const isGreen =
          btn.classList.contains('bg-green-600') ||
          btn.className.includes('bg-green') ||
          btn.className.includes('bg-emerald');

        const isTextMatch =
          txt === 'Trang sau' ||
          (txt.includes('Trang sau') && txt.length <= 25) ||
          title === 'Trang sau' ||
          aria === 'Trang sau' ||
          txt === 'Tiếp tục' ||
          txt === 'Hoàn thành';

        if (isGreen && isTextMatch) {
          return btn;
        }
      }

      // Pass 2: Any button matching exact completion text
      for (const btn of candidates) {
        if (!safeIsVisible(btn) || !safeIsEnabled(btn)) continue;
        if (btn.closest('nav, aside, .sidebar, [class*="sidebar"], ul, ol')) continue;

        const txt = (btn.textContent || '').trim();
        const title = (btn.getAttribute('title') || '').trim();
        const aria = (btn.getAttribute('aria-label') || '').trim();

        if (/^\d+(\s*\/\s*\d+)?$/.test(txt)) continue;
        if (txt.includes('Bài giảng') || title.includes('Bài giảng') || txt.includes('Khóa học')) continue;
        if (txt.includes('Thử lại') || txt.includes('Bỏ qua') || txt.includes('Phản hồi') || txt.includes('Đổi câu hỏi')) continue;

        const isTextMatch =
          txt === 'Trang sau' ||
          (txt.includes('Trang sau') && txt.length <= 25) ||
          title === 'Trang sau' ||
          aria === 'Trang sau' ||
          txt === 'Tiếp tục' ||
          txt === 'Hoàn thành';

        if (isTextMatch) {
          return btn;
        }
      }
    }

    return null;
  }

  /**
   * Find "Bỏ qua" button (skip countdown timer).
   * Matches Playwright: get_dialog_skip_button()
   */
  function getDialogSkipButton() {
    const selectors = ['button', 'a[role="button"]', 'div[role="button"]', '[role="button"]'];
    for (const sel of selectors) {
      try {
        const els = document.querySelectorAll(sel);
        for (const el of els) {
          if (!safeIsVisible(el)) continue;
          const txt = (el.textContent || '').trim();
          const title = el.getAttribute('title') || '';
          const aria = el.getAttribute('aria-label') || '';
          if ((txt.includes('Bỏ qua') || title.includes('Bỏ qua') || aria.includes('Bỏ qua')) && safeIsEnabled(el)) {
            return el;
          }
        }
      } catch (e) {}
    }
    return null;
  }

  /**
   * Find "Câu tiếp theo" button.
   * Matches Playwright: get_dialog_next_question_button()
   */
  function getDialogNextQuestionButton() {
    const selectors = ['button', 'a[role="button"]', 'div[role="button"]', '[role="button"]'];
    for (const sel of selectors) {
      try {
        const els = document.querySelectorAll(sel);
        for (const el of els) {
          if (!safeIsVisible(el)) continue;
          const txt = (el.textContent || '').trim();
          const title = el.getAttribute('title') || '';
          const aria = el.getAttribute('aria-label') || '';
          if (
            (txt.includes('Câu tiếp') || title.includes('Câu tiếp') || aria.includes('Câu tiếp')) &&
            safeIsEnabled(el)
          ) {
            return el;
          }
        }
      } catch (e) {}
    }
    return null;
  }

  /**
   * Find "Thử lại" button anywhere on the page or in dialog.
   * Matches Playwright: get_dialog_retry_button()
   */
  function getDialogRetryButton() {
    const selectors = [
      'button',
      'a[role="button"]',
      'div[role="button"]',
      '[role="button"]',
      'div.cursor-pointer',
      'span.cursor-pointer'
    ];
    for (const sel of selectors) {
      try {
        const els = document.querySelectorAll(sel);
        for (const el of els) {
          if (!safeIsVisible(el)) continue;
          const txt = (el.textContent || '').trim();
          const title = el.getAttribute('title') || '';
          const aria = el.getAttribute('aria-label') || '';
          if (
            (txt === 'Thử lại' || (txt.includes('Thử lại') && txt.length <= 30) ||
             title.includes('Thử lại') || aria.includes('Thử lại')) &&
            safeIsEnabled(el)
          ) {
            return el;
          }
        }
      } catch (e) {}
    }
    return null;
  }

  /**
   * Find action button by multiple target text names.
   * Prioritizes active dialog container first, checks safeIsEnabled or safeIsVisible.
   * Excludes navigation buttons like "Bài giảng".
   * Matches Playwright: find_action_button()
   */
  function findActionButton(names, mustBeEnabled = true) {
    const checkFn = mustBeEnabled ? safeIsEnabled : safeIsVisible;
    const dialog = getActiveDialog();
    const containers = dialog ? [dialog, document] : [document];

    for (const container of containers) {
      for (const name of names) {
        const buttons = Array.from(
          container.querySelectorAll('button, a[role="button"], div[role="button"], [role="button"], div.cursor-pointer')
        );
        for (const btn of buttons) {
          if (!safeIsVisible(btn)) continue;

          // Never click anything in the sidebar
          if (btn.closest('nav, aside, .sidebar, [class*="sidebar"]')) continue;

          const txt = (btn.textContent || '').trim();

          // Never click pure slide numbers
          if (/^\d+(\s*\/\s*\d+)?$/.test(txt)) continue;

          // Never click the "Bài giảng" (Lecture) or "Khóa học" navigation buttons
          if (txt.includes('Bài giảng') && !name.includes('Bài giảng')) continue;
          if (txt.includes('Khóa học') && !name.includes('Khóa học')) continue;

          const title = (btn.getAttribute('title') || '').trim();
          const aria = (btn.getAttribute('aria-label') || '').trim();

          if (
            txt === name ||
            (txt.includes(name) && txt.length <= name.length + 20) ||
            title === name ||
            title.includes(name) ||
            aria === name ||
            aria.includes(name)
          ) {
            if (checkFn(btn)) return btn;
          }
        }
      }
    }

    return null;
  }

  /**
   * Advance to the next slide via dialog button, slide bar, or ArrowRight keyboard event.
   * Matches Playwright: safe_next_slide()
   */
  function safeNextSlide() {
    const dlgBtn = getDialogNextPageButton();
    if (dlgBtn && safeClick(dlgBtn)) return true;

    const nextBtn = findActionButton(['Trang sau'], true);
    if (nextBtn && safeClick(nextBtn)) return true;

    try {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', code: 'ArrowRight', keyCode: 39, which: 39, bubbles: true }));
      window.dispatchEvent(new KeyboardEvent('keyup', { key: 'ArrowRight', code: 'ArrowRight', keyCode: 39, which: 39, bubbles: true }));
      return true;
    } catch (e) {
      return false;
    }
  }

  /**
   * Locate slide answer elements using Playwright's multi-tiered strategy.
   * Matches Playwright: get_answers_locator()
   */
  function getSlideAnswerElements() {
    const notInSidebar = (el) => !el.closest('nav, aside, .sidebar, [class*="sidebar"]');

    // Priority 1: Radiogroup children (div[role='radiogroup'] > div)
    const radiogroupChildren = Array.from(document.querySelectorAll("div[role='radiogroup'] > div")).filter((el) => {
      return safeIsVisible(el) && notInSidebar(el);
    });
    if (radiogroupChildren.length > 0) return radiogroupChildren;

    // Priority 2: Choice cards containing radio button or bold label
    const choiceCards = Array.from(document.querySelectorAll("div.rounded-xl.border-2")).filter((el) => {
      return safeIsVisible(el) && notInSidebar(el) && (el.querySelector("button[role='radio']") || el.querySelector("span.font-bold"));
    });
    if (choiceCards.length > 0) return choiceCards;

    // Priority 3: Direct radio buttons
    const radios = Array.from(document.querySelectorAll("button[role='radio']")).filter((el) => {
      return safeIsVisible(el) && notInSidebar(el);
    });
    if (radios.length > 0) return radios;

    // Priority 4: min-h-[80px] cards
    const minHCards = Array.from(document.querySelectorAll("div.border-2.rounded-xl.min-h-\\[80px\\]")).filter((el) => {
      return safeIsVisible(el) && notInSidebar(el);
    });
    if (minHCards.length > 0) return minHCards;

    // Priority 5: Generic border-2 cursor-pointer
    const pointerCards = Array.from(document.querySelectorAll("div.border-2.cursor-pointer")).filter((el) => {
      if (!safeIsVisible(el) || !notInSidebar(el)) return false;
      const txt = (el.textContent || '').trim();
      const ignore = ['Không có câu hỏi', 'Trả lời trên lớp', 'Kiểm tra', 'Câu tiếp theo', 'Thử lại', 'Trang sau'];
      return !ignore.includes(txt) && txt.length > 0 && txt.length < 500;
    });
    if (pointerCards.length > 0) return pointerCards;

    return [];
  }

  /**
   * Fingerprint answer options when question text cannot be determined.
   * Matches Playwright: answers_fingerprint()
   */
  function answersFingerprint(answerEls) {
    if (!answerEls || answerEls.length === 0) return '';
    try {
      return Array.from(answerEls)
        .map((el) => (el.textContent || '').trim())
        .filter(Boolean)
        .join(' | ')
        .substring(0, 200);
    } catch (e) {
      return '';
    }
  }

  /**
   * Extract question text using Playwright's multi-candidate list.
   * Matches Playwright: get_question_text()
   */
  function getSlideQuestionText(answerEls) {
    const candidates = [
      document.querySelector("div.bg-blue-50.border-blue-500"),
      document.querySelector("[class*='text-blue-800']"),
      document.querySelector("div.bg-blue-50"),
      document.querySelector("p.my-3.text-gray-800.leading-relaxed"),
      document.querySelector("div[role='dialog'] h3"),
      document.querySelector("div[role='dialog'] .font-semibold"),
      document.querySelector("div[role='dialog'] h2")
    ];

    for (const el of candidates) {
      if (el && safeIsVisible(el)) {
        const txt = (el.textContent || '').trim();
        if (txt && txt.length > 3) return txt;
      }
    }

    return answersFingerprint(answerEls) || '?';
  }

  /**
   * Helper to determine if an element has green styling indicating correct answer.
   */
  function isGreenIndicator(el) {
    if (!el) return false;
    try {
      const cls = (el.className && typeof el.className === 'string') ? el.className : '';
      if (
        cls.includes('border-green') ||
        cls.includes('bg-green') ||
        cls.includes('text-green') ||
        cls.includes('border-emerald') ||
        cls.includes('bg-emerald')
      ) {
        if (!cls.includes('border-red') && !cls.includes('bg-red') && !cls.includes('text-red')) {
          return true;
        }
      }

      const style = window.getComputedStyle(el);
      for (const colorStr of [style.borderColor, style.backgroundColor, style.color]) {
        if (!colorStr) continue;
        const match = colorStr.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
        if (match) {
          const r = parseInt(match[1], 10);
          const g = parseInt(match[2], 10);
          const b = parseInt(match[3], 10);
          if (g >= 120 && g > r * 1.25 && g > b * 1.1) {
            return true;
          }
        }
      }
    } catch (e) {}
    return false;
  }

  function isCardMarkedRed(card) {
    if (!card) return false;
    const cls = (card.className && typeof card.className === 'string') ? card.className : '';
    if (cls.includes('border-red') || cls.includes('bg-red') || cls.includes('text-red')) return true;
    const redChild = card.querySelector("[class*='border-red'], [class*='bg-red'], [class*='text-red']");
    if (redChild && safeIsVisible(redChild)) return true;
    try {
      const style = window.getComputedStyle(card);
      for (const colorStr of [style.borderColor, style.backgroundColor]) {
        if (!colorStr) continue;
        const match = colorStr.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
        if (match) {
          const r = parseInt(match[1], 10);
          const g = parseInt(match[2], 10);
          const b = parseInt(match[3], 10);
          if (r >= 150 && r > g * 1.4 && r > b * 1.4) return true;
        }
      }
    } catch (e) {}
    return false;
  }

  function isCardMarkedGreen(card) {
    if (!card) return false;
    if (isGreenIndicator(card)) return true;

    const greenDescendant = card.querySelector(
      "[class*='border-green'], [class*='bg-green'], [class*='text-green'], [class*='border-emerald'], [class*='bg-emerald'], svg.text-green-500, svg[class*='text-green']"
    );
    if (greenDescendant && safeIsVisible(greenDescendant)) {
      const cls = greenDescendant.className || '';
      if (typeof cls === 'string' && !cls.includes('red')) return true;
    }

    const radio = card.querySelector("button[role='radio'], div[role='radio'], input[type='radio']");
    if (radio && isGreenIndicator(radio)) return true;

    return false;
  }

  /**
   * Extract revealed correct answer letter when EDUX displays "Đáp án đúng: X."
   * OR when EDUX highlights the correct answer card in green.
   * Returns zero-based option index (0 for A, 1 for B, etc.)
   * Matches Playwright: extract_revealed_correct_index()
   */
  function extractRevealedCorrectIndex(answerEls = null) {
    // 1. Text search: "Đáp án đúng: X"
    const selectors = [
      "div.text-red-700",
      "[class*='text-red']",
      "div[role='dialog'] div",
      "div[role='dialog'] p",
      "div",
      "p"
    ];

    for (const sel of selectors) {
      try {
        const els = document.querySelectorAll(sel);
        for (const el of els) {
          if (!safeIsVisible(el)) continue;
          const text = el.textContent || '';
          if (text.includes('Đáp án đúng:')) {
            const match = text.match(/Đáp án đúng:\s*([A-Za-z])\b/);
            if (match) {
              const letter = match[1].toUpperCase();
              return letter.charCodeAt(0) - 'A'.charCodeAt(0);
            }
          }
        }
      } catch (e) {}
    }

    // 2. Visual card detection: Green card that is not red
    const cards = answerEls && answerEls.length > 0 ? answerEls : getSlideAnswerElements();
    if (cards && cards.length > 0) {
      for (let i = 0; i < cards.length; i++) {
        const card = cards[i];
        if (isCardMarkedRed(card)) continue;
        if (isCardMarkedGreen(card)) {
          return i;
        }
      }
    }

    return null;
  }

  // =========================================================================
  // 4. Slide Brute-force Engine (Ported from quizz_bruteforce.py)
  // =========================================================================

  async function runSlideBruteforceStep() {
    if (!isSlideRunning) return;

    try {
      // =====================================================================
      // STEP 1: Handle Completion / Transitions (High Priority)
      // Matches Playwright BƯỚC 1 (lines 460-547)
      // =====================================================================

      // 1.1: "Trang sau" button in Dialog (quiz completed, green button)
      const dialogNextBtn = getDialogNextPageButton();
      if (dialogNextBtn) {
        logMessage("[Done] 🎉 Phát hiện nút 'Trang sau' hoàn thành quiz, đang chuyển slide...", 'success');
        safeClick(dialogNextBtn);
        await waitForHidden(dialogNextBtn, 800);
        await sleep(50);
        lastProgress = Date.now();
        stallReported = false;
        return scheduleNextStep(config.delayMs);
      }

      // 1.2: "Bỏ qua" button (skip countdown timer 3-5s)
      const skipBtn = getDialogSkipButton();
      if (skipBtn) {
        logMessage("[Done] Bấm nút 'Bỏ qua' (Skip Countdown)...", 'info');
        safeClick(skipBtn);
        await waitForHidden(skipBtn, 600);
        await sleep(40);

        const dNext = getDialogNextPageButton();
        if (dNext) {
          logMessage("[Done] Bấm tiếp 'Trang sau' sau khi bỏ qua...", 'success');
          safeClick(dNext);
          await waitForHidden(dNext, 800);
        }
        lastProgress = Date.now();
        stallReported = false;
        return scheduleNextStep(config.delayMs);
      }

      // 1.3: "Câu tiếp theo" button
      const nextQBtn = getDialogNextQuestionButton();
      if (nextQBtn) {
        logMessage("[Done] Chuyển 'Câu tiếp theo'...", 'info');
        safeClick(nextQBtn);
        await waitForHidden(nextQBtn, 800);
        await sleep(50);
        lastProgress = Date.now();
        stallReported = false;
        return scheduleNextStep(config.delayMs);
      }

      // 1.4: "Thử lại" button (extract revealed answer first!)
      const retryBtn = getDialogRetryButton();
      if (retryBtn) {
        logMessage("[INFO] Phát hiện nút 'Thử lại', chuẩn bị thử lại câu hỏi...", 'warn');
        const currentAnswerList = getSlideAnswerElements();
        const revealedIdx = extractRevealedCorrectIndex(currentAnswerList);
        if (revealedIdx !== null) {
          const qText = getSlideQuestionText(currentAnswerList);
          knownCorrectAnswers[qText] = revealedIdx;
          logMessage(`[Revealed] 🎯 Ghi nhớ đáp án đúng: #${revealedIdx + 1}`, 'success');
        }
        safeClick(retryBtn);
        await waitForHidden(retryBtn, 1000);
        await sleep(50);
        lastProgress = Date.now();
        stallReported = false;
        return scheduleNextStep(config.delayMs);
      }

      // 1.5: Slide has no question ("Không có câu hỏi")
      const noQuestionBtn = findActionButton(['Không có câu hỏi'], false);
      if (noQuestionBtn) {
        logMessage('[INFO] Slide không có câu hỏi -> Chuyển slide tiếp theo', 'info');
        safeNextSlide();
        await waitForHidden(noQuestionBtn, 800);
        await sleep(50);
        lastProgress = Date.now();
        stallReported = false;
        return scheduleNextStep(config.delayMs);
      }

      // =====================================================================
      // STEP 2: Check Slide Status / Open Quiz Popup
      // Matches Playwright BƯỚC 2 (lines 549-593)
      // =====================================================================
      const answerList = getSlideAnswerElements();
      const answersVisible = answerList.length > 0;

      if (!answersVisible) {
        // 2.1: Open quiz popup if "Trả lời trên lớp" or "Hỏi trên lớp" button exists
        const openBtn = findActionButton(['Trả lời trên lớp', 'Hỏi trên lớp'], true);
        if (openBtn) {
          logMessage('[INFO] Bấm mở popup câu hỏi...', 'info');
          safeClick(openBtn);
          await sleep(150);
          lastProgress = Date.now();
          stallReported = false;
          return scheduleNextStep(config.delayMs);
        }

        // 2.2: Slide is checking ("Đang kiểm tra...")
        const isChecking = Array.from(document.querySelectorAll('span, div, p')).some((el) => {
          return safeIsVisible(el) && (el.textContent || '').includes('Đang kiểm tra...');
        });
        if (isChecking) {
          await sleep(100);
          return scheduleNextStep(80);
        }

        // 2.3: Slide completed or no question: "Trang sau" on slide bar is ENABLED
        if (!getActiveDialog()) {
          const slideNextBtn = findActionButton(['Trang sau'], true);
          if (slideNextBtn) {
            logMessage("[INFO] Bấm 'Trang sau' trên thanh điều khiển slide...", 'info');
            safeClick(slideNextBtn);
            await sleep(150);
            lastProgress = Date.now();
            stallReported = false;
            return scheduleNextStep(config.delayMs);
          }
        }

        // 2.4: Stall Watchdog & Keyboard Recovery
        if (!stallReported && Date.now() - lastProgress > STALL_MS) {
          if (!getActiveDialog()) {
            logMessage('[RECOVERY] Không thấy nút khả dụng, nhấn phím ArrowRight để chuyển slide...', 'warn');
            window.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', code: 'ArrowRight', keyCode: 39, which: 39, bubbles: true }));
            window.dispatchEvent(new KeyboardEvent('keyup', { key: 'ArrowRight', code: 'ArrowRight', keyCode: 39, which: 39, bubbles: true }));
          } else {
            logMessage('[STALL] Đang chờ câu hỏi hoặc kết quả...', 'warn');
          }
          stallReported = true;
          lastProgress = Date.now();
        }

        return scheduleNextStep(100);
      }

      // =====================================================================
      // STEP 3: Answer Question (Answers ARE Visible)
      // Matches Playwright BƯỚC 3 (lines 595-637)
      // =====================================================================
      lastProgress = Date.now();
      stallReported = false;

      const answerCount = answerList.length;
      if (answerCount === 0) return scheduleNextStep(100);

      const questionText = getSlideQuestionText(answerList);
      logMessage(`[Q] ${questionText.substring(0, 60)}...`, 'info');

      // Decide which index to pick
      let nextIndex = 0;
      let pickedByAi = false;

      if (questionText in knownCorrectAnswers && knownCorrectAnswers[questionText] < answerCount) {
        nextIndex = knownCorrectAnswers[questionText];
        logMessage(`[Pick Known Correct] 🎯 #${nextIndex + 1}/${answerCount}`, 'success');
      } else {
        if (!wrongAnswersMap[questionText]) {
          wrongAnswersMap[questionText] = new Set();
        }
        const triedIndices = wrongAnswersMap[questionText];

        // Ưu tiên 1: Dùng AI giải câu hỏi nếu được bật và chưa từng hỏi AI cho câu hỏi này
        const shouldQueryAi = (config.useAi !== false) && !aiAttemptedMap[questionText];
        if (shouldQueryAi) {
          aiAttemptedMap[questionText] = true;
          logMessage(`[AI] 🧠 Đang gửi câu hỏi tới AI và CHỜ phản hồi để chọn đáp án chính xác nhất...`, 'info');

          const choiceTexts = answerList.map((el) => (el.textContent || '').trim().replace(/\s+/g, ' '));

          // BẮT BUỘC CHỜ AI trả về đáp án trước khi thực hiện click để đảm bảo độ chính xác
          const aiRes = await queryAiForSlideAnswer(questionText, choiceTexts);

          if (!isSlideRunning) return; // Người dùng bấm Dừng trong lúc chờ AI

          if (aiRes && aiRes.success && typeof aiRes.index === 'number' && aiRes.index >= 0 && aiRes.index < answerCount) {
            if (!triedIndices.has(aiRes.index)) {
              nextIndex = aiRes.index;
              pickedByAi = true;
              logMessage(`[AI Pick] 🎯 AI đã phản hồi! Chọn đáp án #${nextIndex + 1}: "${choiceTexts[nextIndex].substring(0, 45)}..."`, 'success');
            } else {
              logMessage(`[AI Pick] AI chọn #${aiRes.index + 1} nhưng đáp án này đã thử trước đó và bị sai.`, 'warn');
            }
          } else {
            logMessage(`[AI Note] ${aiRes?.message || 'Không có phản hồi AI'}, chuyển sang tự động thử các đáp án...`, 'warn');
          }
        }

        // Ưu tiên 2: Fallback thử sai nếu không dùng AI hoặc AI chưa chọn được đáp án hợp lệ
        if (!pickedByAi) {
          if (triedIndices.size >= answerCount) {
            triedIndices.clear();
          }

          for (let i = 0; i < answerCount; i++) {
            if (!triedIndices.has(i)) {
              nextIndex = i;
              break;
            }
          }
          logMessage(`[Pick Fallback] #${nextIndex + 1}/${answerCount}`, 'info');
        }
      }

      // Đã có đáp án (sau khi chờ AI hoặc fallback) -> Thực hiện Click
      const optionCard = answerList[nextIndex];
      const radioInside = optionCard.querySelector("button[role='radio'], div[role='radio'], input[type='radio']");

      safeClick(optionCard);
      if (radioInside) {
        safeClick(radioInside);
      }

      await sleep(60);

      // Check if clicking the option card already triggered instant submission / completion
      let instantAction = getDialogNextPageButton() || getDialogNextQuestionButton() || getDialogRetryButton() || getDialogSkipButton();
      let checkBtn = null;

      if (!instantAction) {
        // Fast auto-wait for "Kiểm tra" button to become ENABLED (up to 180ms)
        for (let w = 0; w < 3; w++) {
          checkBtn = findActionButton(['Kiểm tra'], true);
          if (checkBtn) break;
          instantAction = getDialogNextPageButton() || getDialogNextQuestionButton() || getDialogRetryButton() || getDialogSkipButton();
          if (instantAction) break;
          await sleep(60);
        }

        if (!checkBtn && !instantAction) {
          checkBtn = findActionButton(['Kiểm tra'], false);
        }
      }

      if (checkBtn && !instantAction) {
        logMessage("[INFO] Bấm nút 'Kiểm tra'...", 'info');
        safeClick(checkBtn);
      } else if (!instantAction) {
        // No "Kiểm tra" button found: this quiz submits instantly upon option selection!
        logMessage('[INFO] Trắc nghiệm nộp tức thì, đang xử lý kết quả...', 'info');
      }

      // =====================================================================
      // STEP 4: Immediate Result Handling After Submission
      // Matches Playwright BƯỚC 4 (lines 639-733)
      // =====================================================================
      const waitStart = Date.now();
      let handled = false;

      while (Date.now() - waitStart < 2500 && isSlideRunning) {
        // 4.1: Completion button ("Trang sau" / "Tiếp tục" / "Hoàn thành")
        const curDialogNext = getDialogNextPageButton();
        if (curDialogNext) {
          solvedCount++;
          chrome.storage.local.set({ slideStats: { solved: solvedCount, retries: retryCount } });
          logMessage("[Done] 🎉 Đã hoàn thành quiz trên slide! Bấm 'Trang sau' chuyển tiếp", 'success');
          safeClick(curDialogNext);
          await waitForHidden(curDialogNext, 800);
          await sleep(50);
          lastProgress = Date.now();
          stallReported = false;
          handled = true;
          break;
        }

        // 4.2: Next Question button
        const curNextQ = getDialogNextQuestionButton();
        if (curNextQ) {
          solvedCount++;
          chrome.storage.local.set({ slideStats: { solved: solvedCount, retries: retryCount } });
          logMessage("[Done] Đã giải đúng! Chuyển 'Câu tiếp theo'", 'success');
          safeClick(curNextQ);
          await waitForHidden(curNextQ, 800);
          await sleep(50);
          lastProgress = Date.now();
          stallReported = false;
          handled = true;
          break;
        }

        // 4.3: If wrong answer -> "Thử lại" appears
        const curRetry = getDialogRetryButton();
        if (curRetry) {
          retryCount++;
          const curCards = getSlideAnswerElements();
          const revealed = extractRevealedCorrectIndex(curCards);
          if (revealed !== null && revealed < answerCount) {
            knownCorrectAnswers[questionText] = revealed;
            logMessage(`[Revealed] 🎯 Đáp án đúng được hiển thị: #${revealed + 1}`, 'success');
          } else {
            if (!wrongAnswersMap[questionText]) wrongAnswersMap[questionText] = new Set();
            wrongAnswersMap[questionText].add(nextIndex);
            logMessage(`[Wrong] Đã loại đáp án #${nextIndex + 1}`, 'warn');
          }

          chrome.storage.local.set({ slideStats: { solved: solvedCount, retries: retryCount } });
          safeClick(curRetry);
          await waitForHidden(curRetry, 1000);
          await sleep(50);
          lastProgress = Date.now();
          stallReported = false;
          handled = true;
          break;
        }

        // 4.4: Skip Countdown button
        const curSkip = getDialogSkipButton();
        if (curSkip) {
          safeClick(curSkip);
          logMessage("[Done] Đã bấm 'Bỏ qua' đếm ngược", 'info');
          await waitForHidden(curSkip, 500);
        }

        // 4.5: Next Page button on Slide Bar
        if (!getActiveDialog()) {
          const curSlideNext = findActionButton(['Trang sau'], true);
          if (curSlideNext) {
            solvedCount++;
            chrome.storage.local.set({ slideStats: { solved: solvedCount, retries: retryCount } });
            logMessage("[Done] Bấm 'Trang sau' trên Slide", 'success');
            safeClick(curSlideNext);
            await waitForHidden(curSlideNext, 800);
            await sleep(50);
            lastProgress = Date.now();
            stallReported = false;
            handled = true;
            break;
          }
        }

        await sleep(35);
      }

      if (!handled) {
        logMessage('[WARN] Chưa thấy nút phản hồi sau Kiểm tra (có thể do lag mạng), thử lại...', 'warn');
      }

      scheduleNextStep(config.delayMs);
    } catch (err) {
      console.error('[EDUX Slayers] Loop Error:', err);
      logMessage(`[WARN] Lỗi tạm thời, tự hồi phục: ${String(err).substring(0, 80)}`, 'warn');
      scheduleNextStep(1000);
    }
  }

  function scheduleNextStep(delay) {
    if (isSlideRunning) {
      slideLoopTimer = setTimeout(runSlideBruteforceStep, delay);
    }
  }

  function startSlideBruteforce(customConfig) {
    if (customConfig) config = { ...config, ...customConfig };
    if (isSlideRunning) return;

    isSlideRunning = true;
    lastProgress = Date.now();
    stallReported = false;
    wrongAnswersMap = {};
    knownCorrectAnswers = {};
    aiAttemptedMap = {};
    solvedCount = 0;
    retryCount = 0;

    const modeText = config.useAi !== false ? '🧠 Chế độ: AI Siêu Chuẩn Xác (Chờ AI -> Click)' : '⚡ Chế độ: Thử sai nhanh';
    logMessage(`▶️ Bắt đầu tự động giải Slide! [${modeText}]`, 'success');
    notifyPopup('SLIDE_STATUS_CHANGE', { isRunning: true });
    runSlideBruteforceStep();
  }

  function stopSlideBruteforce() {
    isSlideRunning = false;
    if (slideLoopTimer) clearTimeout(slideLoopTimer);
    slideLoopTimer = null;

    logMessage('⏹️ Đã dừng tự động giải Slide.', 'warn');
    notifyPopup('SLIDE_STATUS_CHANGE', { isRunning: false });
  }

  // =========================================================================
  // =========================================================================


  window.EduxSlideSolver = {
    start: startSlideBruteforce,
    stop: stopSlideBruteforce,
    getStatus: () => ({ isSlideRunning, solvedCount, retryCount }),
    setConfig: (newCfg) => { config = { ...config, ...newCfg }; }
  };
})();
