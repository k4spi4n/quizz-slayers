/**
 * EDUX Slayers - Exam Solver (Bài tập)
 * Bắt đề bài tập (từ dữ liệu mạng hoặc cào DOM) và tự động điền đáp án từng bước trên giao diện EDUX.
 * Dùng answer-parser.js, exam-dom.js, exam-prompt.js (nạp trước file này trong manifest).
 */

(function () {
  'use strict';

  const { sleep, safeIsVisible, safeClick, getActiveDialog, createLogger } = window.EduxDOM;
  const {
    QUESTION_LABEL_RE,
    normalizeText,
    parseQuestionIndex,
    sanitizeAiResponse,
    parseTrueFalseAnswers,
    loadAnswersFromInput,
  } = window.EduxAnswerParser;
  const {
    setNativeValue,
    extractOptionsFromEls,
    findTrueFalseBlocks,
    getActiveExamDialog,
    findStartButton,
    findQuestionLabel,
    findButtonByText,
    findPaginationButton,
  } = window.EduxExamDOM;
  const { buildCompactPromptPayload, generateStandardPromptText } = window.EduxExamPrompt;
  const logMessage = createLogger('[EDUX Slayers Bài Tập]', 'TEST_LOG');

  let currentCapturedExamData = null;

  function setCapturedExamData(data) {
    if (data && typeof data === 'object') {
      currentCapturedExamData = data;
      logMessage('📡 Đã ghi nhận dữ liệu đề bài tập từ hệ thống EDUX.', 'info');
    }
  }

  function getCapturedExamData() {
    return currentCapturedExamData;
  }

  /**
   * Trích xuất câu hỏi từ dữ liệu intercepted hoặc cào từ DOM
   */
  function extractQuestions(overrideData) {
    let sourceData = overrideData || currentCapturedExamData;

    // Nếu chưa có trong RAM, thử đọc từ sessionStorage (chia sẻ giữa MAIN world & content script)
    if (!sourceData) {
      try {
        const stored = sessionStorage.getItem('__EDUX_LAST_EXAM_DATA__');
        if (stored) {
          sourceData = JSON.parse(stored);
          currentCapturedExamData = sourceData;
        }
      } catch (e) {}
    }

    if (sourceData) {
      const compact = buildCompactPromptPayload(sourceData, document.title);
      if (compact.total_questions > 0) {
        const promptText = generateStandardPromptText(compact);
        logMessage(`✓ Đã trích xuất ${compact.total_questions} câu từ dữ liệu bài tập!`, 'success');
        return { questions: compact, promptText, fromApi: true };
      }
    }

    // Fallback: Quét trực tiếp từ DOM nếu dialog bài tập đang mở
    const searchRoot = getActiveExamDialog() || getActiveDialog() || document;
    const questionLabels = Array.from(searchRoot.querySelectorAll('p, div, span, h3, h4')).filter(
      (el) => {
        if (el.closest('nav, aside, header, footer')) return false;
        return safeIsVisible(el) && QUESTION_LABEL_RE.test((el.textContent || '').trim());
      },
    );

    if (questionLabels.length > 0) {
      const compact = {
        title: document.title || 'Bài tập EDUX',
        total_questions: 0,
        multiple_choice: [],
        fill_in_blank: [],
        essay: [],
        true_false: [],
      };

      questionLabels.forEach((labelEl) => {
        if (labelEl.closest('nav, aside, header, footer')) return;
        const qNum = parseQuestionIndex(labelEl.textContent);
        if (qNum === null) return;

        let container =
          labelEl.closest(
            'div.bg-white, div.rounded-lg, div.rounded-xl, div.border, div.shadow, section, article',
          ) ||
          labelEl.parentElement?.parentElement?.parentElement ||
          labelEl.parentElement;
        if (!container) return;

        const questionText =
          (container.querySelector('div.prose p, p.text-gray-800') || {}).textContent?.trim() ||
          labelEl.textContent.trim();

        const tfBlocks = findTrueFalseBlocks(container);

        const textarea = container.querySelector('textarea');
        const input = container.querySelector(
          "input:not([type='hidden']):not([type='checkbox']):not([type='radio'])",
        );
        const optionEls = Array.from(
          container.querySelectorAll(
            'div.relative.flex.items-center.space-x-2.p-2.border.rounded-lg.cursor-pointer, div.border.rounded-lg.cursor-pointer',
          ),
        ).filter(safeIsVisible);

        if (tfBlocks.length > 0) {
          const statements = tfBlocks.map((b) => {
            const textEl = b.querySelector('div.prose p, p, div.text-sm');
            return (textEl || b).textContent?.trim().replace(/^\d+\.\s*/, '') || '';
          });
          compact.true_false.push({ id: qNum, question: questionText, statements });
        } else if (textarea) {
          compact.essay.push({ id: qNum, question: questionText });
        } else if (input) {
          compact.fill_in_blank.push({ id: qNum, question: questionText });
        } else if (optionEls.length > 0) {
          const options = {};
          optionEls.forEach((opt, idx) => {
            const letter =
              (opt.querySelector('span.flex-shrink-0') || {}).textContent
                ?.trim()
                .replace(/\.$/, '') || String.fromCharCode(65 + idx);
            const text = (opt.querySelector('div.prose p, p') || opt).textContent?.trim() || '';
            options[letter] = text;
          });
          compact.multiple_choice.push({ id: qNum, question: questionText, options });
        }
      });

      compact.total_questions =
        compact.multiple_choice.length +
        compact.fill_in_blank.length +
        compact.essay.length +
        compact.true_false.length;

      if (compact.total_questions > 0) {
        const promptText = generateStandardPromptText(compact);
        logMessage(`✓ Đã quét ${compact.total_questions} câu hỏi từ giao diện bài tập.`, 'info');
        return { questions: compact, promptText, fromApi: false };
      }
    }

    logMessage(
      '⚠️ Chưa bắt được gói tin đề bài. Hãy bấm nút "🚀 Mở bài" hoặc F5 tải lại trang để bắt đề!',
      'warn',
    );
    return { questions: null, promptText: '', message: 'Chưa bắt được gói tin đề bài tập.' };
  }

  /**
   * Bấm nút "Làm bài tập" hoặc "Bài tập AI" trên trang web
   */
  async function startExercise() {
    // 1. Nếu trên trang đang thấy nút "Làm bài tập", "Làm lại", "Bài tập AI" thì luôn ưu tiên bấm nút
    // để mở bài (kể cả khi đã có kết quả trước đó trên trang)
    const match = findStartButton();

    if (match && safeIsVisible(match.element)) {
      // Xóa cache đề cũ để bắt buộc đợi đề mới của bài tập hiện tại
      currentCapturedExamData = null;
      try {
        sessionStorage.removeItem('__EDUX_LAST_EXAM_DATA__');
        sessionStorage.removeItem('__EDUX_LAST_EXAM_URL__');
        chrome.storage.local.remove('lastExamData');
      } catch (e) {}

      if (match.type === 'start_quiz') {
        const btnText = (match.element.textContent || 'Làm bài tập').trim();
        logMessage(`Đã tìm thấy nút '${btnText}'. Đang bấm để mở bài...`, 'info');
        safeClick(match.element);
        try {
          if (typeof match.element.click === 'function') match.element.click();
        } catch (e) {}

        // Chờ đề bài tập được bắt hoặc dialog xuất hiện (tối đa 12 giây)
        for (let i = 0; i < 48; i++) {
          await sleep(250);

          // Kiểm tra nếu xuất hiện hộp thoại xác nhận làm bài (khi đã làm bài 1 lần trước đó):
          const confirmBtn = findButtonByText(
            ['bắt đầu làm bài', 'làm lại', 'xác nhận', 'đồng ý', 'bắt đầu'],
            document,
            true,
            true,
          );
          if (confirmBtn && confirmBtn !== match.element && safeIsVisible(confirmBtn)) {
            logMessage('Đã phát hiện hộp thoại xác nhận làm bài. Bấm xác nhận...', 'info');
            safeClick(confirmBtn);
          }

          // Kiểm tra xem đã bắt được packet chưa
          const captured = getCapturedExamData();
          if (captured) {
            const compact = buildCompactPromptPayload(captured, document.title);
            logMessage(
              `🎉 Đã mở bài và bắt được gói tin đề (${compact.total_questions} câu)!`,
              'success',
            );
            return { success: true, opened: true, questions: compact };
          }

          // Hoặc kiểm tra dialog câu hỏi đã xuất hiện
          const dialog = getActiveExamDialog();
          if (dialog) {
            logMessage('🎉 Cửa sổ làm bài tập đã mở thành công!', 'success');
            const extracted = extractQuestions();
            return { success: true, opened: true, questions: extracted.questions };
          }

          // Sau 2 giây nếu vẫn chưa mở, thử kích hoạt lại nút bấm
          if (i === 8 || i === 20) {
            logMessage('Đang thử kích hoạt lại nút mở bài tập...', 'info');
            safeClick(match.element);
            try {
              if (typeof match.element.click === 'function') match.element.click();
            } catch (e) {}
          }
        }

        return {
          success: true,
          opened: false,
          message: "Đã bấm 'Làm bài tập', đang chờ hệ thống tải câu hỏi...",
        };
      }

      if (match.type === 'open_lesson_exercise') {
        logMessage("Đã tìm thấy bài học. Bấm 'Bài tập AI' để mở...", 'info');
        safeClick(match.element);
        try {
          if (typeof match.element.click === 'function') match.element.click();
        } catch (e) {}

        // Chờ màn hình có nút "Làm bài tập" xuất hiện (tối đa 6 giây)
        for (let i = 0; i < 30; i++) {
          await sleep(200);
          const nextMatch = findStartButton();
          if (nextMatch && nextMatch.type === 'start_quiz') {
            logMessage("Đã mở bài tập! Tiếp tục bấm nút 'Làm bài tập'...", 'info');
            safeClick(nextMatch.element);
            try {
              if (typeof nextMatch.element.click === 'function') nextMatch.element.click();
            } catch (e) {}

            // Chờ dialog làm bài xuất hiện hoặc bắt được gói tin
            for (let j = 0; j < 48; j++) {
              await sleep(250);

              const confirmBtn = findButtonByText(
                ['bắt đầu làm bài', 'làm lại', 'xác nhận', 'đồng ý', 'bắt đầu'],
                document,
                true,
                true,
              );
              if (confirmBtn && confirmBtn !== nextMatch.element && safeIsVisible(confirmBtn)) {
                logMessage('Đã phát hiện hộp thoại xác nhận làm bài. Bấm xác nhận...', 'info');
                safeClick(confirmBtn);
              }

              const captured = getCapturedExamData();
              if (captured) {
                const compact = buildCompactPromptPayload(captured, document.title);
                logMessage(
                  `🎉 Đã mở bài và bắt được gói tin đề (${compact.total_questions} câu)!`,
                  'success',
                );
                return { success: true, opened: true, questions: compact };
              }
              const dialog = getActiveExamDialog();
              if (dialog) {
                logMessage('🎉 Cửa sổ làm bài tập đã mở thành công!', 'success');
                const extracted = extractQuestions();
                return { success: true, opened: true, questions: extracted.questions };
              }
            }
            return { success: true, opened: true };
          }
        }
        return {
          success: true,
          opened: false,
          message: "Đã mở màn hình bài tập. Hãy bấm 'Làm bài tập' trên trang.",
        };
      }
    }

    // 2. Nếu không thấy nút bấm trên trang, kiểm tra nếu dialog câu hỏi ĐÃ thực sự mở sẵn
    const existingDialog = getActiveExamDialog();
    if (existingDialog) {
      const qLabel = findQuestionLabel(existingDialog);
      logMessage(
        `Cửa sổ bài tập đã được mở sẵn sàng${qLabel ? ' (' + qLabel.textContent.trim() + ')' : ''}.`,
        'success',
      );
      const extracted = extractQuestions();
      return { success: true, opened: true, questions: extracted.questions };
    }

    return {
      success: false,
      message: "Không tìm thấy nút 'Làm bài tập' hoặc 'Bài tập AI' trên trang.",
    };
  }

  /**
   * Bắt đầu điền đáp án bài tập
   */
  async function fillTestAnswers(rawText, options = {}) {
    const answers = loadAnswersFromInput(rawText);
    const questionIndices = Object.keys(answers);
    if (questionIndices.length === 0) {
      return {
        success: false,
        message: 'Không thể phân tích bất kỳ đáp án hợp lệ nào từ nội dung đã nhập!',
      };
    }

    logMessage(`🚀 Bắt đầu điền ${questionIndices.length} câu trả lời cho bài tập...`, 'info');

    // Chờ hoặc lấy dialog bài tập
    let dialog = getActiveExamDialog();
    if (!dialog) {
      logMessage('Cửa sổ bài tập chưa mở. Đang tự động mở bài tập để điền...', 'info');
      const startRes = await startExercise();
      if (startRes && startRes.opened) {
        await sleep(350);
        dialog = getActiveExamDialog();
      }
    }

    if (!dialog) {
      for (let wait = 0; wait < 15; wait++) {
        await sleep(200);
        dialog = getActiveExamDialog() || getActiveDialog();
        if (dialog && safeIsVisible(dialog)) break;
      }
    }

    let result;
    if (dialog && safeIsVisible(dialog)) {
      result = await fillTestDialog(dialog, answers, options);
    } else {
      result = fillTestFullPage(answers);
    }

    return result;
  }

  /**
   * Vòng lặp điền bài tập từng bước mô phỏng chính xác test_solver.py lines 483-623
   */
  async function fillTestDialog(dialog, answers, options = {}) {
    let filledCount = 0;
    const autoSubmit = options.autoSubmit !== false;
    const maxIterations = Object.keys(answers).length + 15;
    let iterations = 0;

    // Đảm bảo bắt đầu từ câu 1 nếu hiện tại đang đứng ở câu khác
    let firstLabel = findQuestionLabel(dialog);
    let startIdx = firstLabel ? parseQuestionIndex(firstLabel.textContent) : null;
    if (startIdx && startIdx > 1) {
      logMessage(
        `[INFO] Đang ở câu ${startIdx}. Tự động quay lại câu 1 để giải toàn bộ bài tập...`,
        'info',
      );
      const btn1 = findPaginationButton(1, dialog);
      if (btn1) {
        safeClick(btn1);
        await sleep(350);
      } else {
        for (let b = 0; b < startIdx; b++) {
          const prevBtn = findButtonByText(['Câu trước'], dialog, true, true);
          if (!prevBtn) break;
          safeClick(prevBtn);
          await sleep(150);
          const cur = findQuestionLabel(dialog);
          if (cur && parseQuestionIndex(cur.textContent) === 1) break;
        }
      }
    }

    while (iterations < maxIterations) {
      iterations++;

      // 1. Chờ label "Câu X" xuất hiện (tối đa 4 giây mỗi câu)
      let labelEl = null;
      for (let wait = 0; wait < 20; wait++) {
        labelEl = findQuestionLabel(dialog);
        if (labelEl) break;
        await sleep(200);
      }

      if (!labelEl) {
        logMessage('[WARN] Không tìm thấy nhãn câu hỏi. Dừng tiến trình.', 'warn');
        break;
      }

      const labelText = labelEl.textContent.trim();
      const questionIndex = parseQuestionIndex(labelText);
      if (questionIndex === null) {
        logMessage(`[WARN] Không parse được số câu từ: "${labelText}"`, 'warn');
        break;
      }

      const answerValue = (answers[questionIndex] || '').trim();

      if (!answerValue) {
        logMessage(`[WARN] Không có đáp án cho câu ${questionIndex}, bỏ qua.`, 'warn');
      } else {
        // Chờ ít nhất 1 phần tử tương tác của câu hỏi xuất hiện (True/False, Textarea, Input, ContentEditable, Options)
        let trueFalseBlocks = [];
        let textareaEl = null;
        let inputEls = [];
        let contentEditableEl = null;
        let optionEls = [];

        for (let wait = 0; wait < 15; wait++) {
          trueFalseBlocks = findTrueFalseBlocks(dialog);

          textareaEl = dialog.querySelector('textarea');
          if (textareaEl && !safeIsVisible(textareaEl)) textareaEl = null;

          inputEls = Array.from(
            dialog.querySelectorAll(
              "input:not([type='hidden']):not([type='checkbox']):not([type='radio'])",
            ),
          ).filter(safeIsVisible);

          contentEditableEl = dialog.querySelector('[contenteditable="true"], [role="textbox"]');
          if (contentEditableEl && !safeIsVisible(contentEditableEl)) contentEditableEl = null;

          optionEls = Array.from(
            dialog.querySelectorAll(
              "div.relative.flex.items-center.space-x-2.p-2.border.rounded-lg.cursor-pointer, div.border.rounded-lg.cursor-pointer, [role='radio']",
            ),
          ).filter(safeIsVisible);

          if (
            trueFalseBlocks.length > 0 ||
            textareaEl ||
            inputEls.length > 0 ||
            contentEditableEl ||
            optionEls.length > 0
          ) {
            break;
          }
          await sleep(200);
        }

        if (trueFalseBlocks.length > 0) {
          const tfAnswers = parseTrueFalseAnswers(answerValue, trueFalseBlocks.length);
          logMessage(
            `[INFO] Câu ${questionIndex}: điền Đúng/Sai (${tfAnswers.length}/${trueFalseBlocks.length} mệnh đề)`,
            'info',
          );
          for (let i = 0; i < trueFalseBlocks.length; i++) {
            const block = trueFalseBlocks[i];
            const shouldBeTrue = i < tfAnswers.length ? tfAnswers[i] : true;
            const targetName = shouldBeTrue ? 'Đúng' : 'Sai';
            const btn = Array.from(block.querySelectorAll('button')).find(
              (b) => (b.textContent || '').trim() === targetName,
            );
            if (btn) {
              const isAlreadyActive =
                btn.classList.contains('bg-green-500') ||
                btn.classList.contains('bg-red-500') ||
                btn.getAttribute('data-state') === 'active' ||
                btn.getAttribute('aria-pressed') === 'true';
              if (!isAlreadyActive) {
                safeClick(btn);
                await sleep(120);
              }
            }
          }
          filledCount++;
        } else if (textareaEl) {
          logMessage(`[INFO] Câu ${questionIndex}: điền tự luận`, 'info');
          setNativeValue(textareaEl, answerValue);
          filledCount++;
        } else if (inputEls.length > 0) {
          logMessage(`[INFO] Câu ${questionIndex}: điền ô trống (${inputEls.length} ô)`, 'info');
          if (inputEls.length === 1) {
            setNativeValue(inputEls[0], answerValue);
          } else {
            let parts = [];
            try {
              const parsed = JSON.parse(answerValue);
              if (Array.isArray(parsed)) parts = parsed.map(String);
            } catch (e) {}
            if (parts.length === 0) {
              parts = answerValue
                .split(/[,;\n]/)
                .map((s) => s.trim())
                .filter(Boolean);
            }
            for (let i = 0; i < inputEls.length; i++) {
              const val = i < parts.length ? parts[i] : answerValue;
              setNativeValue(inputEls[i], val);
            }
          }
          filledCount++;
        } else if (contentEditableEl) {
          logMessage(
            `[INFO] Câu ${questionIndex}: điền vùng nhập văn bản (contenteditable)`,
            'info',
          );
          setNativeValue(contentEditableEl, answerValue);
          filledCount++;
        } else {
          // Trắc nghiệm nhiều lựa chọn
          let currentOptionEls = optionEls;

          if (currentOptionEls.length === 0) {
            logMessage(`[WARN] Câu ${questionIndex}: không tìm thấy lựa chọn đáp án.`, 'warn');
          } else {
            logMessage(`[INFO] Câu ${questionIndex}: chọn '${answerValue}'`, 'info');
            const optionsList = extractOptionsFromEls(currentOptionEls);
            let chosenIndex = -1;
            const trimmedAns = answerValue.trim();

            // 1. Khớp theo ký tự đầu A, B, C, D (hỗ trợ "A", "A.", "A: ", "(A)")
            const letterMatch =
              trimmedAns.match(/^[\(\[]?([A-D])[\.\)\:\s]/i) ||
              (trimmedAns.length === 1 && trimmedAns.match(/^([A-D])$/i));

            if (letterMatch) {
              const targetLetter = letterMatch[1].toUpperCase();
              for (let i = 0; i < optionsList.length; i++) {
                const optL = optionsList[i].letter.toUpperCase().replace(/[^A-D]/g, '');
                if (
                  optL === targetLetter ||
                  optionsList[i].letter.toUpperCase().startsWith(targetLetter)
                ) {
                  chosenIndex = i;
                  break;
                }
              }
            }

            // 2. Khớp theo nội dung text nếu chưa tìm thấy bằng ký tự
            if (chosenIndex === -1) {
              const textWithoutLetter = trimmedAns
                .replace(/^[\(\[]?[A-D][\.\)\:\s\-]+/i, '')
                .trim();
              const target = normalizeText(textWithoutLetter || trimmedAns);
              if (target) {
                for (let i = 0; i < optionsList.length; i++) {
                  const optText = normalizeText(optionsList[i].text);
                  if (optText && (optText.includes(target) || target.includes(optText))) {
                    chosenIndex = i;
                    break;
                  }
                }
              }
            }

            // 3. Khớp theo số thứ tự (1..4)
            if (chosenIndex === -1 && /^[1-4]$/.test(trimmedAns)) {
              const idx = parseInt(trimmedAns, 10) - 1;
              if (optionsList[idx]) chosenIndex = idx;
            }

            if (chosenIndex === -1) {
              logMessage(
                `[WARN] Câu ${questionIndex}: Không khớp được lựa chọn nào cho '${answerValue}'.`,
                'warn',
              );
            } else {
              safeClick(optionsList[chosenIndex].node);
              filledCount++;
            }
          }
        }
      }

      await sleep(250);

      // Kiểm tra nút "Nộp bài" và "Câu tiếp"
      const submitBtn =
        findButtonByText(['Nộp bài', 'Nộp'], dialog, true, true) ||
        findButtonByText(['Nộp bài', 'Nộp'], document, true, true);
      const nextBtn = findButtonByText(['Câu tiếp', 'Câu tiếp theo'], dialog, true, true);

      // Nếu chỉ có nút Nộp bài hoặc không còn Câu tiếp
      if (!nextBtn && submitBtn) {
        if (autoSubmit) {
          logMessage("🎉 Đã đến câu cuối. Tự động bấm nút 'Nộp bài'...", 'success');
          safeClick(submitBtn);
          await sleep(500);
          const confirmBtn = findButtonByText(
            ['Xác nhận', 'Đồng ý', 'Chắc chắn', 'Nộp bài'],
            document,
            true,
            true,
          );
          if (confirmBtn && confirmBtn !== submitBtn) {
            logMessage('✓ Bấm xác nhận nộp bài...', 'info');
            safeClick(confirmBtn);
          }
        } else {
          logMessage("✓ Đã hoàn thành điền câu cuối. Bạn có thể bấm 'Nộp bài'.", 'success');
        }
        break;
      }

      if (nextBtn) {
        const currentLabel = labelText;
        const progressEl = dialog.querySelector('span.text-gray-700');
        const currentProgress = progressEl ? progressEl.textContent.trim() : '';

        safeClick(nextBtn);

        // Chờ câu tiếp theo xuất hiện (label đổi HOẶC progress đổi - mô phỏng test_solver.py lines 602-618)
        let changed = false;
        for (let i = 0; i < 35; i++) {
          await sleep(150);
          const newLabelEl = findQuestionLabel(dialog);
          const newLabel = newLabelEl ? newLabelEl.textContent.trim() : '';
          const newProgressEl = dialog.querySelector('span.text-gray-700');
          const newProgress = newProgressEl ? newProgressEl.textContent.trim() : '';

          if (
            (newLabel && newLabel !== currentLabel) ||
            (newProgress && newProgress !== currentProgress)
          ) {
            changed = true;
            break;
          }
        }

        // Fallback: nếu bấm nextBtn không đổi, thử bấm pagination button kế tiếp
        if (!changed) {
          const nextIndex = questionIndex + 1;
          const paginationBtn = findPaginationButton(nextIndex, dialog);
          if (paginationBtn) {
            logMessage(`[INFO] Thử chuyển câu bằng nút số ${nextIndex}...`, 'info');
            safeClick(paginationBtn);
            for (let i = 0; i < 20; i++) {
              await sleep(150);
              const newLabelEl = findQuestionLabel(dialog);
              if (newLabelEl && newLabelEl.textContent.trim() !== currentLabel) {
                changed = true;
                break;
              }
            }
          }
        }

        if (!changed) {
          // Khi bấm "Câu tiếp" mà không đổi câu, có thể đã đến câu cuối cùng
          let endSubmitBtn =
            findButtonByText(['Nộp bài', 'Nộp'], dialog, true, true) ||
            findButtonByText(['Nộp bài', 'Nộp'], document, true, true);

          if (endSubmitBtn && autoSubmit) {
            logMessage("🎉 Không còn câu tiếp theo. Tự động bấm nút 'Nộp bài'...", 'success');
            safeClick(endSubmitBtn);
            await sleep(500);
            const confirmBtn = findButtonByText(
              ['Xác nhận', 'Đồng ý', 'Chắc chắn', 'Nộp bài', 'Nộp'],
              document,
              true,
              true,
            );
            if (confirmBtn && confirmBtn !== endSubmitBtn) {
              logMessage('✓ Bấm xác nhận nộp bài...', 'info');
              safeClick(confirmBtn);
            }
          } else {
            // Kiểm tra xem sau khi bấm "Câu tiếp" có xuất hiện modal xác nhận nộp bài không
            let confirmModalBtn = null;
            for (let c = 0; c < 8; c++) {
              await sleep(150);
              confirmModalBtn = findButtonByText(
                ['Xác nhận', 'Đồng ý', 'Chắc chắn', 'Nộp bài', 'Nộp'],
                document,
                true,
                true,
              );
              if (confirmModalBtn && confirmModalBtn !== nextBtn) break;
            }
            if (confirmModalBtn && autoSubmit) {
              logMessage('✓ Đã phát hiện hộp thoại xác nhận nộp bài. Bấm xác nhận...', 'info');
              safeClick(confirmModalBtn);
            } else {
              logMessage('[WARN] Câu tiếp theo chưa hiển thị kịp hoặc đã đến cuối bài.', 'warn');
            }
          }
          break;
        }
      } else if (submitBtn) {
        if (autoSubmit) {
          logMessage("🎉 Tự động bấm nút 'Nộp bài'...", 'success');
          safeClick(submitBtn);
          await sleep(500);
          const confirmBtn = findButtonByText(
            ['Xác nhận', 'Đồng ý', 'Chắc chắn', 'Nộp bài', 'Nộp'],
            document,
            true,
            true,
          );
          if (confirmBtn && confirmBtn !== submitBtn) safeClick(confirmBtn);
        }
        break;
      } else {
        // Không còn nút Câu tiếp và không có nút Nộp bài (bài tập tự lưu)
        logMessage('🎉 Đã hoàn thành câu cuối cùng của bài tập!', 'success');
        break;
      }
    }

    logMessage(`🎉 Hoàn tất! Đã điền xong ${filledCount} câu trong bài tập.`, 'success');

    // Dọn dẹp cache đề đã nộp và thông báo cho Popup cập nhật UI
    currentCapturedExamData = null;
    try {
      sessionStorage.removeItem('__EDUX_LAST_EXAM_DATA__');
      sessionStorage.removeItem('__EDUX_LAST_EXAM_URL__');
      chrome.storage.local.remove('lastExamData');
      chrome.runtime.sendMessage({ type: 'EXAM_SUBMITTED' });
    } catch (e) {}

    return { success: true, filledCount };
  }

  /**
   * Fallback khi bài tập hiển thị cả trang (không phải modal)
   */
  function fillTestFullPage(answers) {
    let filledCount = 0;
    const questionLabels = Array.from(document.querySelectorAll('p, div, span, h3, h4')).filter(
      (el) => {
        return safeIsVisible(el) && QUESTION_LABEL_RE.test((el.textContent || '').trim());
      },
    );

    questionLabels.forEach((labelEl) => {
      const qNum = parseQuestionIndex(labelEl.textContent);
      if (qNum === null) return;

      const targetAns = answers[qNum];
      if (!targetAns) return;

      let container =
        labelEl.closest(
          'div.bg-white, div.rounded-lg, div.rounded-xl, div.border, div.shadow, section, article',
        ) ||
        labelEl.parentElement?.parentElement?.parentElement ||
        labelEl.parentElement;
      if (!container) return;

      const tfBlocks = findTrueFalseBlocks(container);

      if (tfBlocks.length > 0) {
        const tfAnswers = parseTrueFalseAnswers(targetAns, tfBlocks.length);
        tfBlocks.forEach((block, i) => {
          const shouldBeTrue = i < tfAnswers.length ? tfAnswers[i] : true;
          const btn = Array.from(block.querySelectorAll('button')).find(
            (b) => (b.textContent || '').trim() === (shouldBeTrue ? 'Đúng' : 'Sai'),
          );
          if (btn) safeClick(btn);
        });
        filledCount++;
        logMessage(`✓ Câu ${qNum}: đã chọn Đúng/Sai`, 'success');
        return;
      }

      const textarea = container.querySelector('textarea');
      if (textarea && safeIsVisible(textarea)) {
        setNativeValue(textarea, targetAns);
        filledCount++;
        logMessage(`✓ Câu ${qNum}: đã điền tự luận`, 'success');
        return;
      }

      const input = container.querySelector(
        "input:not([type='hidden']):not([type='checkbox']):not([type='radio'])",
      );
      if (input && safeIsVisible(input)) {
        setNativeValue(input, targetAns);
        filledCount++;
        logMessage(`✓ Câu ${qNum}: đã điền ô trống`, 'success');
        return;
      }

      const contentEditable = container.querySelector('[contenteditable="true"], [role="textbox"]');
      if (contentEditable && safeIsVisible(contentEditable)) {
        setNativeValue(contentEditable, targetAns);
        filledCount++;
        logMessage(`✓ Câu ${qNum}: đã điền vùng nhập văn bản`, 'success');
        return;
      }

      const optionEls = Array.from(
        container.querySelectorAll(
          'div.relative.flex.items-center.space-x-2.p-2.border.rounded-lg.cursor-pointer, div.border.rounded-lg.cursor-pointer, label',
        ),
      ).filter(safeIsVisible);

      for (const opt of optionEls) {
        const optText = (opt.textContent || '').trim();
        const letterSpan =
          (opt.querySelector('span.flex-shrink-0') || {}).textContent?.trim() || '';
        const targetUpper = targetAns.toUpperCase();

        if (
          letterSpan === targetUpper ||
          letterSpan.startsWith(targetUpper + '.') ||
          optText.startsWith(targetUpper + '.') ||
          (targetAns.length > 1 && normalizeText(optText).includes(normalizeText(targetAns)))
        ) {
          safeClick(opt);
          filledCount++;
          logMessage(`✓ Câu ${qNum}: đã chọn ${targetAns}`, 'success');
          break;
        }
      }
    });

    logMessage(`🎉 Đã tự động điền xong ${filledCount} câu hỏi bài tập.`, 'success');
    return { success: true, filledCount };
  }

  // =========================================================================
  // Xuất API toàn cục cho Extension
  // =========================================================================
  window.EduxTestSolver = {
    fillTestAnswers,
    extractQuestions,
    startExercise,
    setCapturedExamData,
    getCapturedExamData,
    getActiveExamDialog,
    findStartButton,
    loadAnswersFromInput,
    sanitizeAiResponse,
    buildCompactPromptPayload,
    generateStandardPromptText,
  };
})();
