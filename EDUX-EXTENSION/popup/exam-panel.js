// Tab Bài tập: chế độ Tự động (API) và Chatbot — mở bài, bắt đề, gọi AI, dán & điền đáp án
import { UI, addLog, setStatus, showTab } from './ui.js';
import { getActiveTab, sendTabMessage } from './edux-tab.js';
import { getAssignedProfile } from './ai-profiles.js';
import { displayModel, isProfileReady } from '../shared/providers.js';

export function updateExamInfoUI(data) {
  if (!UI.examInfoBox || !UI.examInfoText) return;
  if (data && typeof data === 'object') {
    const payloadData = data.data || data;
    const title = payloadData.title || data.title || 'Bài tập phát hiện';
    const qCount =
      payloadData.total_questions ||
      data.total_questions ||
      payloadData.exam_data?.multiple_choice?.length ||
      data.exam_data?.multiple_choice?.length ||
      '?';
    UI.examInfoBox.classList.add('active');
    if (UI.examStatusDot) UI.examStatusDot.style.display = 'block';
    UI.examInfoText.style.display = 'block';
    UI.examInfoText.textContent = `🎯 ${title} (${qCount} câu)`;
    UI.examInfoText.title = title;
    if (UI.btnStartExercise) {
      UI.btnStartExercise.style.width = 'auto';
      UI.btnStartExercise.innerHTML = '<span>🚀</span> Mở bài';
    }
  } else {
    UI.examInfoBox.classList.remove('active');
    if (UI.examStatusDot) UI.examStatusDot.style.display = 'none';
    UI.examInfoText.style.display = 'none';
    UI.examInfoText.textContent = '';
    UI.examInfoText.title = '';
    if (UI.btnStartExercise) {
      UI.btnStartExercise.style.width = '100%';
      UI.btnStartExercise.style.justifyContent = 'center';
      UI.btnStartExercise.innerHTML = '<span>🚀</span> Mở bài tập';
    }
  }
}

// Stepper State: 0 = idle, 1 = bắt đề, 2 = AI giải, 3 = điền bài, 4 = xong
function setStepperState(step = 0) {
  const steps = [UI.step1, UI.step2, UI.step3];
  const lines = [UI.stepLine1, UI.stepLine2];

  steps.forEach((s) => {
    if (s) {
      s.classList.remove('active', 'done');
    }
  });
  lines.forEach((l) => {
    if (l) {
      l.classList.remove('done');
    }
  });

  if (step === 1) {
    if (UI.step1) UI.step1.classList.add('active');
  } else if (step === 2) {
    if (UI.step1) UI.step1.classList.add('done');
    if (UI.stepLine1) UI.stepLine1.classList.add('done');
    if (UI.step2) UI.step2.classList.add('active');
  } else if (step === 3) {
    if (UI.step1) UI.step1.classList.add('done');
    if (UI.stepLine1) UI.stepLine1.classList.add('done');
    if (UI.step2) UI.step2.classList.add('done');
    if (UI.stepLine2) UI.stepLine2.classList.add('done');
    if (UI.step3) UI.step3.classList.add('active');
  } else if (step >= 4) {
    steps.forEach((s) => s && s.classList.add('done'));
    lines.forEach((l) => l && l.classList.add('done'));
  }
}

function switchTestMode(mode) {
  const isAuto = mode === 'auto';
  if (UI.btnModeAuto) UI.btnModeAuto.classList.toggle('active', isAuto);
  if (UI.btnModeManual) UI.btnModeManual.classList.toggle('active', !isAuto);
  if (UI.testAutoSection) UI.testAutoSection.style.display = isAuto ? 'flex' : 'none';
  if (UI.testManualSection) UI.testManualSection.style.display = isAuto ? 'none' : 'flex';
  chrome.storage.local.set({ testWorkflowMode: mode });
}

/**
 * Làm mới / Bắt đầu phiên giải bài mới:
 * Xóa sạch ô đáp án, prompt xem trước, và reset stepper.
 */
async function resetSolveSession(keepExamInfo = true) {
  if (UI.answerInput) UI.answerInput.value = '';
  await chrome.storage.local.remove('savedAnswers');

  if (UI.promptPreviewBox) UI.promptPreviewBox.value = '';
  if (UI.promptPreviewCard) UI.promptPreviewCard.style.display = 'none';

  if (UI.autoAnswersBox) UI.autoAnswersBox.value = '';
  if (UI.autoAnswersContainer) UI.autoAnswersContainer.style.display = 'none';
  if (UI.btnToggleAutoAnswers) UI.btnToggleAutoAnswers.style.display = 'none';

  setStepperState(0);
  setHeroBtnLoading(false);

  if (!keepExamInfo) {
    updateExamInfoUI(null);
    await chrome.storage.local.remove('lastExamData');
  }
}

function setHeroBtnLoading(isLoading, title, subtitle) {
  if (!UI.btnSolveAI) return;
  const iconEl = UI.btnSolveAI.querySelector('.hero-btn-icon');
  const titleEl = UI.btnSolveAI.querySelector('.hero-btn-title');
  const subtitleEl = UI.btnSolveAI.querySelector('.hero-btn-subtitle');

  if (isLoading) {
    UI.btnSolveAI.classList.add('loading');
    if (iconEl) iconEl.innerHTML = '<span class="spinner"></span>';
    if (titleEl && title) titleEl.textContent = title;
    if (subtitleEl && subtitle) subtitleEl.textContent = subtitle;
  } else {
    UI.btnSolveAI.classList.remove('loading');
    if (iconEl) iconEl.textContent = '⚡';
    if (titleEl) titleEl.textContent = 'GIẢI BÀI TẬP BẰNG AI (API)';
    if (subtitleEl) subtitleEl.textContent = 'Tự mở bài ➔ AI giải ➔ Điền đáp án ➔ Nộp bài';
  }
}

export function initExamPanel(settings) {
  switchTestMode(settings.testWorkflowMode || 'auto');

  if (settings.savedAnswers) {
    UI.answerInput.value = settings.savedAnswers;
    if (UI.autoAnswersBox) UI.autoAnswersBox.value = settings.savedAnswers;
    if (UI.btnToggleAutoAnswers) UI.btnToggleAutoAnswers.style.display = 'block';
  }

  // Khởi tạo trạng thái ban đầu: chỉ hiển thị nút "Mở bài tập"
  updateExamInfoUI(null);

  // Chuyển đổi chế độ: Tự động (API) vs Chatbot
  if (UI.btnModeAuto) {
    UI.btnModeAuto.addEventListener('click', () => switchTestMode('auto'));
  }
  if (UI.btnModeManual) {
    UI.btnModeManual.addEventListener('click', () => switchTestMode('manual'));
  }

  // Nút mở nhanh Chatbot AI (ChatGPT / Gemini / Claude) trong tab mới
  document.querySelectorAll('.ai-link-btn[data-ai-url]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const url = btn.getAttribute('data-ai-url');
      if (!url) return;
      if (window.chrome?.tabs?.create) {
        chrome.tabs.create({ url });
      } else {
        window.open(url, '_blank');
      }
      addLog(UI.testLog, `🔗 Đã mở ${btn.textContent.trim()} — hãy dán đề vào đó.`, 'info');
    });
  });

  // Nút liên kết chuyển sang tab Cài đặt đổi Model
  [UI.btnGoToSettings, UI.btnSlideGoToSettings].forEach((btn) => {
    if (!btn) return;
    btn.addEventListener('click', () => {
      showTab('tab-settings');
    });
  });

  // Nút Phiên mới: Xóa trắng ô đáp án, prompt xem trước và reset tiến trình
  if (UI.btnNewSession) {
    UI.btnNewSession.addEventListener('click', async () => {
      await resetSolveSession(true);
      addLog(UI.testLog, '🔄 Đã bắt đầu phiên làm việc mới (ô đáp án đã được xóa trắng).', 'info');
    });
  }

  // Nút Xóa nhanh ô đáp án trong chế độ chatbot
  if (UI.btnClearAnswers) {
    UI.btnClearAnswers.addEventListener('click', async () => {
      await resetSolveSession(true);
      addLog(UI.testLog, '🗑️ Đã xóa trắng ô đáp án.', 'info');
    });
  }

  // Xem/ẩn đáp án vừa giải ở chế độ Tự động
  if (UI.btnToggleAutoAnswers) {
    UI.btnToggleAutoAnswers.addEventListener('click', () => {
      if (!UI.autoAnswersContainer) return;
      const isHidden = UI.autoAnswersContainer.style.display === 'none';
      UI.autoAnswersContainer.style.display = isHidden ? 'block' : 'none';
    });
  }
  if (UI.btnHideAutoAnswers) {
    UI.btnHideAutoAnswers.addEventListener('click', () => {
      if (UI.autoAnswersContainer) UI.autoAnswersContainer.style.display = 'none';
    });
  }

  // Nút Mở bài tập (Bấm "Làm bài tập" trên trang)
  if (UI.btnStartExercise) {
    UI.btnStartExercise.addEventListener('click', async () => {
      // Khi bắt đầu một phiên bài tập mới -> Xóa trắng ô đáp án cũ
      await resetSolveSession(true);
      setStepperState(1);

      const origHtml = UI.btnStartExercise.innerHTML;
      UI.btnStartExercise.innerHTML = '<span class="spinner spinner-sm"></span> Đang mở...';
      UI.btnStartExercise.disabled = true;

      const tab = await getActiveTab();
      if (!tab) {
        UI.btnStartExercise.innerHTML = origHtml;
        UI.btnStartExercise.disabled = false;
        return;
      }

      try {
        addLog(UI.testLog, "Đang tìm nút 'Làm bài tập' trên trang...", 'info');
        const res = await sendTabMessage(tab.id, { action: 'START_EXERCISE' });
        if (res && res.success) {
          if (res.questions) {
            updateExamInfoUI(res.questions);
            addLog(
              UI.testLog,
              `🎉 Cửa sổ bài tập đã sẵn sàng (${res.questions.total_questions || 0} câu)!`,
              'success',
            );
          } else {
            addLog(
              UI.testLog,
              res.opened ? 'Cửa sổ bài tập đã sẵn sàng!' : 'Đã bấm nút làm bài tập.',
              'success',
            );
          }
        } else {
          addLog(UI.testLog, res?.message || 'Không tìm thấy nút làm bài tập.', 'warn');
          setStepperState(0);
        }
      } catch (err) {
        addLog(UI.testLog, 'Lỗi: ' + err.message, 'error');
        setStepperState(0);
      } finally {
        UI.btnStartExercise.innerHTML = origHtml;
        UI.btnStartExercise.disabled = false;
      }
    });
  }

  // Nút: Copy Prompt câu hỏi chuẩn theo EDUX-TEST-SOLVER (Chế độ chatbot)
  if (UI.btnExtractQuestions) {
    UI.btnExtractQuestions.addEventListener('click', async () => {
      const tab = await getActiveTab();
      if (!tab) return;

      const origHtml = UI.btnExtractQuestions.innerHTML;
      UI.btnExtractQuestions.innerHTML =
        '<span class="spinner spinner-sm"></span> Đang trích xuất...';
      UI.btnExtractQuestions.disabled = true;

      try {
        // 1. Kiểm tra xem bài tập đã mở trên trang chưa
        const checkRes = await sendTabMessage(tab.id, {
          action: 'CHECK_EXAM_OPEN',
        });
        let res = null;

        if (!checkRes || !checkRes.isOpen) {
          addLog(UI.testLog, "Bài tập chưa mở. Đang bấm 'Làm bài tập' và bắt đề...", 'info');
          const startRes = await sendTabMessage(tab.id, {
            action: 'START_EXERCISE',
          });
          if (startRes && startRes.questions) {
            res = startRes;
            updateExamInfoUI(startRes.questions);
          } else if (!startRes || !startRes.opened) {
            addLog(UI.testLog, startRes?.message || 'Không thể mở bài tập trên trang.', 'warn');
            return;
          }
        }

        // 2. Trích xuất đề nếu chưa có
        if (!res || !res.promptText) {
          addLog(UI.testLog, 'Đang trích xuất đề bài tập...', 'info');
          res = await sendTabMessage(tab.id, { action: 'EXTRACT_QUESTIONS' });
        }

        if (res && res.promptText) {
          await navigator.clipboard.writeText(res.promptText);
          if (UI.promptPreviewBox) UI.promptPreviewBox.value = res.promptText;
          if (UI.promptPreviewCard) UI.promptPreviewCard.style.display = 'flex';
          updateExamInfoUI(res.questions);
          addLog(
            UI.testLog,
            `Thành công! Đã copy Prompt (${res.questions?.total_questions || 0} câu) vào Clipboard.`,
            'success',
          );
        } else {
          addLog(UI.testLog, 'Chưa tìm thấy câu hỏi bài tập nào trên trang.', 'warn');
        }
      } catch (err) {
        addLog(UI.testLog, 'Lỗi trích xuất câu hỏi: ' + err.message, 'error');
      } finally {
        UI.btnExtractQuestions.innerHTML = origHtml;
        UI.btnExtractQuestions.disabled = false;
      }
    });
  }

  // Nút Hero CTA: Giải tự động bằng AI (API)
  if (UI.btnSolveAI) {
    UI.btnSolveAI.addEventListener('click', async () => {
      const tab = await getActiveTab();
      if (!tab) return;

      const examProfile = getAssignedProfile('exam');

      if (!isProfileReady(examProfile)) {
        addLog(UI.testLog, '⚠️ Chưa có cấu hình AI cho Bài tập.', 'warn');
        addLog(UI.testLog, '💬 Đang tự động chuyển sang chế độ Chatbot để bạn tự giải...', 'info');
        switchTestMode('manual');

        // Hỗ trợ người dùng: tự động mở bài và copy prompt sang chế độ thủ công
        try {
          const checkRes = await sendTabMessage(tab.id, {
            action: 'CHECK_EXAM_OPEN',
          });
          let extRes = null;
          if (!checkRes || !checkRes.isOpen) {
            const startRes = await sendTabMessage(tab.id, {
              action: 'START_EXERCISE',
            });
            if (startRes && startRes.questions) extRes = startRes;
          }
          if (!extRes || !extRes.promptText) {
            extRes = await sendTabMessage(tab.id, {
              action: 'EXTRACT_QUESTIONS',
            });
          }
          if (extRes && extRes.promptText) {
            await navigator.clipboard.writeText(extRes.promptText);
            if (UI.promptPreviewBox) UI.promptPreviewBox.value = extRes.promptText;
            if (UI.promptPreviewCard) UI.promptPreviewCard.style.display = 'flex';
            updateExamInfoUI(extRes.questions);
            addLog(
              UI.testLog,
              `✓ Đã tự động copy Prompt (${extRes.questions?.total_questions || 0} câu) vào Clipboard! Hãy dán vào ChatGPT/Claude.`,
              'success',
            );
          }
        } catch (e) {}
        return;
      }

      // XÓA TRẮNG Ô ĐÁP ÁN KHI BẮT ĐẦU PHIÊN GIẢI MỚI & BẬT BƯỚC 1 (BẮT ĐỀ) VỚI SPINNER
      await resetSolveSession(true);
      setStepperState(1);
      setHeroBtnLoading(true, 'ĐANG BẮT ĐỀ BÀI TẬP...', 'Đang mở bài tập và trích xuất câu hỏi...');

      let extRes = null;

      try {
        // BƯỚC 1: Đảm bảo bài tập được mở trên trang trước khi giải
        const checkRes = await sendTabMessage(tab.id, {
          action: 'CHECK_EXAM_OPEN',
        });

        if (!checkRes || !checkRes.isOpen) {
          addLog(UI.testLog, "Bài tập chưa mở. Đang bấm 'Làm bài tập' và bắt đề...", 'info');
          const startRes = await sendTabMessage(tab.id, {
            action: 'START_EXERCISE',
          });
          if (startRes && startRes.questions) {
            extRes = startRes;
            updateExamInfoUI(startRes.questions);
          } else if (!startRes || !startRes.opened) {
            addLog(UI.testLog, startRes?.message || 'Không thể mở bài tập trên trang.', 'warn');
            setStepperState(0);
            setHeroBtnLoading(false);
            switchTestMode('manual');
            addLog(
              UI.testLog,
              '💬 Đã tự động chuyển sang chế độ Chatbot để bạn tự thao tác.',
              'info',
            );
            return;
          }
        }

        // BƯỚC 2: Trích xuất đề bài tập (nếu chưa có từ startRes)
        if (!extRes || !extRes.promptText) {
          addLog(UI.testLog, 'Đang trích xuất đề bài tập...', 'info');
          extRes = await sendTabMessage(tab.id, {
            action: 'EXTRACT_QUESTIONS',
          });
        }

        if (!extRes || !extRes.promptText) {
          addLog(UI.testLog, 'Không tìm thấy đề bài tập. Hãy kiểm tra giao diện bài tập!', 'warn');
          setStepperState(0);
          setHeroBtnLoading(false);
          switchTestMode('manual');
          addLog(
            UI.testLog,
            '💬 Đã tự động chuyển sang chế độ Chatbot để bạn tự thao tác.',
            'info',
          );
          return;
        }

        const qCount = extRes.questions?.total_questions || 0;
        updateExamInfoUI(extRes.questions);
        const modelName = displayModel(examProfile.model, examProfile.provider);
        addLog(UI.testLog, `Đang gửi ${qCount} câu tới AI (${modelName})...`, 'info');
        setStatus('AI đang giải bài...', 'running');

        // BƯỚC 2: AI giải câu hỏi
        setStepperState(2);
        setHeroBtnLoading(
          true,
          'AI ĐANG GIẢI BÀI...',
          `Đang gửi ${qCount} câu tới AI (${modelName})...`,
        );

        // Background dùng cấu hình được gán cho "exam" (cùng examProfile ở trên)
        const aiRes = await chrome.runtime.sendMessage({
          action: 'AI_SOLVE_EXAM',
          promptText: extRes.promptText,
        });
        if (!aiRes?.success) {
          throw new Error(aiRes?.message || 'Không nhận được phản hồi từ AI.');
        }
        const aiAnswers = aiRes.answersText;
        UI.answerInput.value = aiAnswers;
        if (UI.autoAnswersBox) UI.autoAnswersBox.value = aiAnswers;
        if (UI.btnToggleAutoAnswers) UI.btnToggleAutoAnswers.style.display = 'block';
        await chrome.storage.local.set({ savedAnswers: aiAnswers });

        addLog(UI.testLog, '✓ AI đã giải xong! Bắt đầu tự động điền đáp án...', 'success');

        // BƯỚC 3: Điền & nộp
        setStepperState(3);
        setHeroBtnLoading(true, 'ĐANG ĐIỀN ĐÁP ÁN...', 'Đang tự động chọn đáp án và nộp bài...');

        const fillRes = await sendTabMessage(tab.id, {
          action: 'FILL_TEST_ANSWERS',
          answersText: aiAnswers,
          options: {
            autoSubmit: UI.settingAutoSubmit ? UI.settingAutoSubmit.checked : true,
          },
        });

        setStatus('Sẵn sàng', 'idle');
        setHeroBtnLoading(false);

        if (fillRes && fillRes.success) {
          setStepperState(4); // Hoàn tất cả 3 bước
          addLog(
            UI.testLog,
            `🎉 Hoàn tất! Đã điền xong ${fillRes.filledCount} câu bài tập.`,
            'success',
          );
          // Sau khi nộp thành công đề: ẩn tiêu đề, chỉ còn nút "Mở bài"
          updateExamInfoUI(null);
          await chrome.storage.local.remove('lastExamData');
        } else {
          setStepperState(0);
          addLog(UI.testLog, `Thông báo: ${fillRes?.message || 'Không thể điền bài.'}`, 'warn');
          switchTestMode('manual');
          addLog(
            UI.testLog,
            '💬 Đã chuyển sang chế độ Chatbot để bạn kiểm tra lại đáp án và điền lại.',
            'info',
          );
        }
      } catch (err) {
        setStatus('Lỗi giải bài', 'stopped');
        setStepperState(0);
        setHeroBtnLoading(false);
        addLog(UI.testLog, '❌ Lỗi giải tự động: ' + err.message, 'error');

        // TỰ ĐỘNG CHUYỂN SANG CHẾ ĐỘ CHATBOT KHI GẶP LỖI
        switchTestMode('manual');
        addLog(
          UI.testLog,
          '💬 Đã tự động chuyển sang chế độ Chatbot. Bạn có thể tự dán đáp án vào ô bên dưới.',
          'warn',
        );

        // Nếu đã trích xuất được prompt trước khi lỗi, hiển thị ngay vào khung prompt chatbot
        if (extRes && extRes.promptText) {
          if (UI.promptPreviewBox) UI.promptPreviewBox.value = extRes.promptText;
          if (UI.promptPreviewCard) UI.promptPreviewCard.style.display = 'flex';
          navigator.clipboard.writeText(extRes.promptText).catch(() => {});
          addLog(
            UI.testLog,
            `✓ Đã sao chép sẵn đề (${extRes.questions?.total_questions || 0} câu) vào Clipboard và khung Prompt.`,
            'info',
          );
        }
      }
    });
  }

  // Nút Dán đáp án từ Clipboard (Chế độ chatbot)
  // Gọi readText() trong user gesture để trình duyệt hiện hộp xin quyền clipboard
  if (UI.btnPasteClipboard) {
    UI.btnPasteClipboard.addEventListener('click', async () => {
      const origHtml = UI.btnPasteClipboard.innerHTML;
      UI.btnPasteClipboard.innerHTML = '⏳ Đang xin quyền...';
      try {
        // Kích hoạt hộp thoại xin quyền của trình duyệt (nếu chưa cấp)
        try {
          await navigator.permissions.query({
            name: 'clipboard-read',
          });
        } catch (_) {
          // Bỏ qua: một số trình duyệt không hỗ trợ query, readText() vẫn sẽ hỏi quyền
        }
        // Lệnh này buộc trình duyệt hiện prompt "Cho phép đọc clipboard?" khi cần
        const text = await navigator.clipboard.readText();
        if (!text || !text.trim()) {
          addLog(UI.testLog, 'Clipboard đang trống! Hãy copy đáp án từ chatbot trước.', 'warn');
          if (UI.answerInput) UI.answerInput.focus();
          return;
        }
        UI.answerInput.value = text.trim();
        await chrome.storage.local.set({ savedAnswers: text.trim() });
        addLog(UI.testLog, '✓ Đã dán đáp án từ Clipboard.', 'success');
      } catch (e) {
        if (e && e.name === 'NotAllowedError') {
          addLog(
            UI.testLog,
            '⚠️ Trình duyệt chặn đọc Clipboard. Hãy bấm “Cho phép / Allow” khi được hỏi, rồi bấm lại nút này (hoặc dán bằng Ctrl+V).',
            'warn',
          );
        } else {
          addLog(UI.testLog, 'Không thể đọc Clipboard: ' + e.message, 'error');
        }
        if (UI.answerInput) UI.answerInput.focus();
      } finally {
        UI.btnPasteClipboard.innerHTML = origHtml;
      }
    });
  }

  // Nút Xem/ẩn Prompt xem trước — đã gộp vào nút "Copy đề & xem trước",
  // giữ lại để tương thích nếu phần tử cũ còn tồn tại
  if (UI.btnTogglePrompt) {
    UI.btnTogglePrompt.addEventListener('click', async () => {
      if (!UI.promptPreviewCard) return;
      if (UI.promptPreviewCard.style.display === 'none') {
        if (!UI.promptPreviewBox.value.trim()) {
          const tab = await getActiveTab();
          if (tab) {
            const res = await sendTabMessage(tab.id, {
              action: 'EXTRACT_QUESTIONS',
            });
            if (res && res.promptText) {
              UI.promptPreviewBox.value = res.promptText;
              updateExamInfoUI(res.questions);
            }
          }
        }
        UI.promptPreviewCard.style.display = 'flex';
      } else {
        UI.promptPreviewCard.style.display = 'none';
      }
    });
  }

  if (UI.btnHidePrompt) {
    UI.btnHidePrompt.addEventListener('click', () => {
      if (UI.promptPreviewCard) UI.promptPreviewCard.style.display = 'none';
    });
  }

  // Nút Bắt đầu điền bài tập (Chế độ chatbot)
  if (UI.btnFillAnswers) {
    UI.btnFillAnswers.addEventListener('click', async () => {
      const rawAnswers = UI.answerInput.value.trim();
      if (!rawAnswers) {
        addLog(UI.testLog, 'Vui lòng nhập hoặc dán danh sách đáp án trước!', 'warn');
        return;
      }

      await chrome.storage.local.set({ savedAnswers: rawAnswers });

      const tab = await getActiveTab();
      if (!tab) return;

      const origHtml = UI.btnFillAnswers.innerHTML;
      UI.btnFillAnswers.innerHTML = '<span class="spinner spinner-sm"></span> Đang điền bài...';
      UI.btnFillAnswers.disabled = true;

      try {
        // Đảm bảo bài tập đang mở trước khi điền
        const checkRes = await sendTabMessage(tab.id, {
          action: 'CHECK_EXAM_OPEN',
        });
        if (!checkRes || !checkRes.isOpen) {
          addLog(UI.testLog, 'Đang mở bài tập trên trang để chuẩn bị điền...', 'info');
          const startRes = await sendTabMessage(tab.id, {
            action: 'START_EXERCISE',
          });
          if (!startRes || !startRes.opened) {
            addLog(UI.testLog, startRes?.message || 'Không thể mở bài tập trên trang.', 'warn');
            return;
          }
          await new Promise((r) => setTimeout(r, 400));
        }

        addLog(UI.testLog, 'Đang gửi đáp án tới trang bài tập...', 'info');
        const res = await sendTabMessage(tab.id, {
          action: 'FILL_TEST_ANSWERS',
          answersText: rawAnswers,
          options: {
            autoSubmit: UI.settingAutoSubmit ? UI.settingAutoSubmit.checked : true,
          },
        });
        if (res && res.success) {
          addLog(UI.testLog, `Hoàn tất! Đã điền ${res.filledCount} câu bài tập.`, 'success');
          // Sau khi nộp thành công đề: ẩn tiêu đề, chỉ còn nút "Mở bài"
          updateExamInfoUI(null);
          await chrome.storage.local.remove('lastExamData');
        } else {
          addLog(UI.testLog, `Thông báo: ${res?.message || 'Không thể điền bài tập.'}`, 'warn');
        }
      } catch (err) {
        addLog(UI.testLog, 'Lỗi: Không tìm thấy trang bài tập EDUX.', 'error');
      } finally {
        UI.btnFillAnswers.innerHTML = origHtml;
        UI.btnFillAnswers.disabled = false;
      }
    });
  }

  // Log & sự kiện gửi về từ exam-solver trên trang
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === 'TEST_LOG') {
      addLog(UI.testLog, msg.message, msg.logType || 'info');
    } else if (msg.type === 'EXAM_DATA_READY') {
      updateExamInfoUI(msg.payload);
      addLog(UI.testLog, '📡 Đã bắt được đề bài tập từ hệ thống!', 'success');
    } else if (msg.type === 'EXAM_SUBMITTED') {
      updateExamInfoUI(null);
      chrome.storage.local.remove('lastExamData');
    }
  });
}
