// Phần tử giao diện popup và các helper hiển thị dùng chung giữa các tab
export const UI = {
  tabs: document.querySelectorAll(".tab-btn"),
  tabContents: document.querySelectorAll(".tab-content"),
  globalStatus: document.getElementById("globalStatus"),

  // Slide Solver UI
  btnStartSlide: document.getElementById("btnStartSlide"),
  btnStopSlide: document.getElementById("btnStopSlide"),
  slideCount: document.getElementById("slideCount"),
  retryCount: document.getElementById("retryCount"),
  slideLog: document.getElementById("slideLog"),
  btnSlideMethodAi: document.getElementById("btnSlideMethodAi"),
  btnSlideMethodLaya: document.getElementById("btnSlideMethodLaya"),
  btnSlideMethodBrute: document.getElementById("btnSlideMethodBrute"),
  slideMethodDesc: document.getElementById("slideMethodDesc"),
  slideMethodIcon: document.getElementById("slideMethodIcon"),
  slideMethodTitle: document.getElementById("slideMethodTitle"),
  slideMethodDetail: document.getElementById("slideMethodDetail"),

  // Test Solver (Bài tập) UI
  examInfoBox: document.getElementById("examInfoBox"),
  examInfoText: document.getElementById("examInfoText"),
  examStatusDot: document.getElementById("examStatusDot"),
  btnStartExercise: document.getElementById("btnStartExercise"),
  btnNewSession: document.getElementById("btnNewSession"),
  btnModeAuto: document.getElementById("btnModeAuto"),
  btnModeManual: document.getElementById("btnModeManual"),
  testAutoSection: document.getElementById("testAutoSection"),
  testManualSection: document.getElementById("testManualSection"),
  autoStepper: document.getElementById("autoStepper"),
  step1: document.getElementById("step1"),
  step2: document.getElementById("step2"),
  step3: document.getElementById("step3"),
  stepLine1: document.getElementById("stepLine1"),
  stepLine2: document.getElementById("stepLine2"),
  btnExtractQuestions: document.getElementById("btnExtractQuestions"),
  btnSolveAI: document.getElementById("btnSolveAI"),
  btnGoToSettings: document.getElementById("btnGoToSettings"),
  autoAnswersContainer: document.getElementById("autoAnswersContainer"),
  autoAnswersBox: document.getElementById("autoAnswersBox"),
  btnHideAutoAnswers: document.getElementById("btnHideAutoAnswers"),
  btnToggleAutoAnswers: document.getElementById("btnToggleAutoAnswers"),
  promptPreviewCard: document.getElementById("promptPreviewCard"),
  promptPreviewBox: document.getElementById("promptPreviewBox"),
  btnHidePrompt: document.getElementById("btnHidePrompt"),
  btnTogglePrompt: document.getElementById("btnTogglePrompt"),
  btnPasteClipboard: document.getElementById("btnPasteClipboard"),
  btnClearAnswers: document.getElementById("btnClearAnswers"),
  answerInput: document.getElementById("answerInput"),
  btnFillAnswers: document.getElementById("btnFillAnswers"),
  testLog: document.getElementById("testLog"),

  // Exercise Scores UI
  scoresSubjectTitle: document.getElementById("scoresSubjectTitle"),
  scoresCompleted: document.getElementById("scoresCompleted"),
  scoresHighest: document.getElementById("scoresHighest"),
  scoresAlertBox: document.getElementById("scoresAlertBox"),
  btnRefreshScores: document.getElementById("btnRefreshScores"),
  scoresList: document.getElementById("scoresList"),

  // Settings UI
  settingDelay: document.getElementById("settingDelay"),
  settingAutoNext: document.getElementById("settingAutoNext"),
  settingAutoSubmit: document.getElementById("settingAutoSubmit"),
  settingSlideMethod: document.getElementById("settingSlideMethod"),
  settingLayaEndpoint: document.getElementById("settingLayaEndpoint"),
  settingLayaApiKey: document.getElementById("settingLayaApiKey"),
  btnTestLaya: document.getElementById("btnTestLaya"),
  layaStatus: document.getElementById("layaStatus"),
  aiProfileList: document.getElementById("aiProfileList"),
  aiProfileEditor: document.getElementById("aiProfileEditor"),
  btnAddAiProfile: document.getElementById("btnAddAiProfile"),
  btnSaveAiProfile: document.getElementById("btnSaveAiProfile"),
  btnCancelAiProfile: document.getElementById("btnCancelAiProfile"),
  assignSlide: document.getElementById("assignSlide"),
  assignExam: document.getElementById("assignExam"),
  examProfileSelect: document.getElementById("examProfileSelect"),
  settingReasoningGroup: document.getElementById("settingReasoningGroup"),
  settingReasoning: document.getElementById("settingReasoning"),
  slideProfileStrip: document.getElementById("slideProfileStrip"),
  slideProfileSelect: document.getElementById("slideProfileSelect"),
  btnSlideGoToSettings: document.getElementById("btnSlideGoToSettings"),
  settingApiProvider: document.getElementById("settingApiProvider"),
  settingApiEndpointGroup: document.getElementById("settingApiEndpointGroup"),
  settingApiEndpoint: document.getElementById("settingApiEndpoint"),
  settingApiKey: document.getElementById("settingApiKey"),
  settingModelSelect: document.getElementById("settingModelSelect"),
  settingModelCustom: document.getElementById("settingModelCustom"),
  customModelGroup: document.getElementById("customModelGroup"),
  btnFetchModels: document.getElementById("btnFetchModels"),
  modelDatalist: document.getElementById("modelDatalist"),
  fetchModelsStatus: document.getElementById("fetchModelsStatus"),
  btnSaveSettings: document.getElementById("btnSaveSettings"),

  // Update & Backup UI
  appVersion: document.getElementById("appVersion"),
  updateBanner: document.getElementById("updateBanner"),
  updateBannerTitle: document.getElementById("updateBannerTitle"),
  updateBannerDesc: document.getElementById("updateBannerDesc"),
  btnUpdateAction: document.getElementById("btnUpdateAction"),
  updateCard: document.getElementById("updateCard"),
  currentVersion: document.getElementById("currentVersion"),
  updateStatusText: document.getElementById("updateStatusText"),
  btnCheckUpdate: document.getElementById("btnCheckUpdate"),
  btnDownloadUpdate: document.getElementById("btnDownloadUpdate"),
  btnReleaseNotes: document.getElementById("btnReleaseNotes"),
  btnExportSettings: document.getElementById("btnExportSettings"),
  btnImportSettings: document.getElementById("btnImportSettings"),
  importSettingsFile: document.getElementById("importSettingsFile"),
  backupStatus: document.getElementById("backupStatus"),
};

export function addLog(container, message, type = "info") {
  if (!container) return;
  const entry = document.createElement("div");
  entry.className = `log-entry ${type}`;
  const timeStr = new Date().toLocaleTimeString("vi-VN", { hour12: false });
  entry.textContent = `[${timeStr}] ${message}`;
  container.appendChild(entry);
  container.scrollTop = container.scrollHeight;
}

export function setStatus(text, state = "idle") {
  if (!UI.globalStatus) return;
  UI.globalStatus.className = `status-indicator ${state}`;
  const textEl = UI.globalStatus.querySelector(".status-text");
  if (textEl) textEl.textContent = text;
}

// Chuyển tab giao diện; onShow(tabId) chạy sau khi tab được hiện
export function initTabNavigation(onShow) {
  UI.tabs.forEach((btn) => {
    btn.addEventListener("click", () => {
      const tabId = btn.getAttribute("data-tab");
      UI.tabs.forEach((b) => b.classList.remove("active"));
      UI.tabContents.forEach((c) => c.classList.remove("active"));

      btn.classList.add("active");
      const targetContent = document.getElementById(tabId);
      if (targetContent) targetContent.classList.add("active");

      onShow?.(tabId);
    });
  });
}

export function showTab(tabId) {
  const tabBtn = document.querySelector(`.tab-btn[data-tab="${tabId}"]`);
  if (tabBtn) tabBtn.click();
}
