/**
 * EDUX Slayers - Popup Controller v2.2.0
 * Điều khiển giao diện Extension, giải Slide, giải Bài tập (mô phỏng EDUX-TEST-SOLVER)
 * và theo dõi điểm số môn học.
 */

document.addEventListener('DOMContentLoaded', async () => {
  // Ordered content scripts for tab re-injection
  const CONTENT_SCRIPTS = [
    'scripts/dom-utils.js',
    'scripts/slide-solver.js',
    'scripts/test-solver.js',
    'scripts/score-tracker.js',
    'content.js'
  ];

  // =========================================================================
  // 1. UI Elements Mapping
  // =========================================================================
  const UI = {
    tabs: document.querySelectorAll('.tab-btn'),
    tabContents: document.querySelectorAll('.tab-content'),
    globalStatus: document.getElementById('globalStatus'),

    // Slide Solver UI
    btnStartSlide: document.getElementById('btnStartSlide'),
    btnStopSlide: document.getElementById('btnStopSlide'),
    slideCount: document.getElementById('slideCount'),
    retryCount: document.getElementById('retryCount'),
    slideLog: document.getElementById('slideLog'),

    // Test Solver (Bài tập) UI
    examInfoBox: document.getElementById('examInfoBox'),
    examInfoText: document.getElementById('examInfoText'),
    examStatusDot: document.getElementById('examStatusDot'),
    btnStartExercise: document.getElementById('btnStartExercise'),
    btnExtractQuestions: document.getElementById('btnExtractQuestions'),
    btnSolveAI: document.getElementById('btnSolveAI'),
    promptPreviewCard: document.getElementById('promptPreviewCard'),
    promptPreviewBox: document.getElementById('promptPreviewBox'),
    btnHidePrompt: document.getElementById('btnHidePrompt'),
    btnTogglePrompt: document.getElementById('btnTogglePrompt'),
    btnPasteClipboard: document.getElementById('btnPasteClipboard'),
    answerInput: document.getElementById('answerInput'),
    btnFillAnswers: document.getElementById('btnFillAnswers'),
    testLog: document.getElementById('testLog'),

    // Exercise Scores UI
    scoresSubjectTitle: document.getElementById('scoresSubjectTitle'),
    scoresCompleted: document.getElementById('scoresCompleted'),
    scoresHighest: document.getElementById('scoresHighest'),
    scoresAlertBox: document.getElementById('scoresAlertBox'),
    btnRefreshScores: document.getElementById('btnRefreshScores'),
    scoresList: document.getElementById('scoresList'),

    // Settings UI
    settingDelay: document.getElementById('settingDelay'),
    settingAutoNext: document.getElementById('settingAutoNext'),
    settingAutoSubmit: document.getElementById('settingAutoSubmit'),
    settingUseAiSlide: document.getElementById('settingUseAiSlide'),
    settingApiProvider: document.getElementById('settingApiProvider'),
    settingApiEndpointGroup: document.getElementById('settingApiEndpointGroup'),
    settingApiEndpoint: document.getElementById('settingApiEndpoint'),
    settingApiKey: document.getElementById('settingApiKey'),
    settingModel: document.getElementById('settingModel'),
    settingModelSelect: document.getElementById('settingModelSelect'),
    settingModelCustom: document.getElementById('settingModelCustom'),
    customModelGroup: document.getElementById('customModelGroup'),
    btnFetchModels: document.getElementById('btnFetchModels'),
    modelDatalist: document.getElementById('modelDatalist'),
    fetchModelsStatus: document.getElementById('fetchModelsStatus'),
    btnSaveSettings: document.getElementById('btnSaveSettings')
  };

  // =========================================================================
  // 2. Logging & Status Helpers
  // =========================================================================
  function addLog(container, message, type = 'info') {
    if (!container) return;
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    const timeStr = new Date().toLocaleTimeString('vi-VN', { hour12: false });
    entry.textContent = `[${timeStr}] ${message}`;
    container.appendChild(entry);
    container.scrollTop = container.scrollHeight;
  }

  function setStatus(text, state = 'idle') {
    if (!UI.globalStatus) return;
    UI.globalStatus.className = `status-indicator ${state}`;
    const textEl = UI.globalStatus.querySelector('.status-text');
    if (textEl) textEl.textContent = text;
  }

  function updateExamInfoUI(data) {
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
      UI.examInfoText.textContent = `🎯 ${title} (${qCount} câu)`;
      UI.examInfoText.title = title;
    } else {
      UI.examInfoBox.classList.remove('active');
      UI.examInfoText.textContent = "Chưa bắt được đề. Mở hoặc bấm 'Làm bài tập'.";
      UI.examInfoText.title = '';
    }
  }

  // =========================================================================
  // 3. Tab Communication
  // =========================================================================
  async function getActiveTab() {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab && tab.url && (tab.url.includes('cmcu.edu.vn') || tab.url.includes('edux'))) {
      return tab;
    }
    const allTabs = await chrome.tabs.query({});
    const eduxTab = allTabs.find((t) => t.url && (t.url.includes('cmcu.edu.vn') || t.url.includes('edux')));
    return eduxTab || tab;
  }

  async function sendTabMessage(tabId, message) {
    try {
      return await chrome.tabs.sendMessage(tabId, message);
    } catch (err) {
      // Content script not loaded or tab disconnected: inject required scripts
      try {
        await chrome.scripting.executeScript({
          target: { tabId },
          files: ['injected.js'],
          world: 'MAIN'
        }).catch(() => {});

        await chrome.scripting.executeScript({
          target: { tabId },
          files: CONTENT_SCRIPTS
        });
        await chrome.scripting.insertCSS({
          target: { tabId },
          files: ['content.css']
        });
        await new Promise((r) => setTimeout(r, 200));
        return await chrome.tabs.sendMessage(tabId, message);
      } catch (injectErr) {
        throw err;
      }
    }
  }

  // =========================================================================
  // 4. Tab Navigation
  // =========================================================================
  UI.tabs.forEach((btn) => {
    btn.addEventListener('click', () => {
      const tabId = btn.getAttribute('data-tab');
      UI.tabs.forEach((b) => b.classList.remove('active'));
      UI.tabContents.forEach((c) => c.classList.remove('active'));

      btn.classList.add('active');
      const targetContent = document.getElementById(tabId);
      if (targetContent) targetContent.classList.add('active');

      if (tabId === 'tab-scores') {
        loadExerciseScores();
      }
    });
  });

  // =========================================================================
  // 5. Load Stored Configuration & Initial State
  // =========================================================================
  const API_PRESETS = {
    gemini: {
      endpoint: '',
      model: 'gemini-2.0-flash',
      placeholderEndpoint: 'Mặc định: https://generativelanguage.googleapis.com',
      keyPlaceholder: 'Nhập Google Gemini API Key (AIza...)'
    },
    openai: {
      endpoint: 'https://api.openai.com/v1',
      model: 'gpt-4o-mini',
      placeholderEndpoint: 'https://api.openai.com/v1',
      keyPlaceholder: 'Nhập OpenAI API Key (sk-...)'
    },
    deepseek: {
      endpoint: 'https://api.deepseek.com/v1',
      model: 'deepseek-chat',
      placeholderEndpoint: 'https://api.deepseek.com/v1',
      keyPlaceholder: 'Nhập DeepSeek API Key (sk-...)'
    },
    openrouter: {
      endpoint: 'https://openrouter.ai/api/v1',
      model: 'google/gemini-2.0-flash-001',
      placeholderEndpoint: 'https://openrouter.ai/api/v1',
      keyPlaceholder: 'Nhập OpenRouter API Key (sk-or-v1-...)'
    },
    ollama: {
      endpoint: 'http://localhost:11434/v1',
      model: 'llama3.2',
      placeholderEndpoint: 'http://localhost:11434/v1',
      keyPlaceholder: 'Không cần API Key đối với Ollama (để trống)'
    },
    custom: {
      endpoint: 'http://localhost:20128/v1',
      model: '',
      placeholderEndpoint: 'http://localhost:20128/v1',
      keyPlaceholder: 'Nhập API Key nếu có (hoặc để trống)...'
    }
  };

  // =========================================================================
  // 5.1 Model Dropdown & Preset Models
  // =========================================================================
  const DEFAULT_MODELS_BY_PROVIDER = {
    gemini: [
      { id: 'gemini-2.0-flash', label: 'gemini-2.0-flash (Khuyên dùng - Nhanh & Chuẩn)' },
      { id: 'gemini-2.0-pro-exp-02-05', label: 'gemini-2.0-pro-exp-02-05 (Suy luận sâu)' },
      { id: 'gemini-1.5-flash', label: 'gemini-1.5-flash' },
      { id: 'gemini-1.5-pro', label: 'gemini-1.5-pro' }
    ],
    openai: [
      { id: 'gpt-4o-mini', label: 'gpt-4o-mini (Khuyên dùng - Nhanh & Rẻ)' },
      { id: 'gpt-4o', label: 'gpt-4o (Toàn diện nhất)' },
      { id: 'o3-mini', label: 'o3-mini (Lý luận cao cấp)' },
      { id: 'gpt-4-turbo', label: 'gpt-4-turbo' }
    ],
    deepseek: [
      { id: 'deepseek-chat', label: 'deepseek-chat (DeepSeek-V3)' },
      { id: 'deepseek-reasoner', label: 'deepseek-reasoner (DeepSeek-R1)' }
    ],
    openrouter: [
      { id: 'google/gemini-2.0-flash-001', label: 'google/gemini-2.0-flash-001' },
      { id: 'deepseek/deepseek-r1', label: 'deepseek/deepseek-r1' },
      { id: 'meta-llama/llama-3.3-70b-instruct', label: 'meta-llama/llama-3.3-70b-instruct' },
      { id: 'anthropic/claude-3.5-sonnet', label: 'anthropic/claude-3.5-sonnet' }
    ],
    ollama: [
      { id: 'llama3.2', label: 'llama3.2' },
      { id: 'qwen2.5:7b', label: 'qwen2.5:7b' },
      { id: 'deepseek-r1:7b', label: 'deepseek-r1:7b' },
      { id: 'mistral', label: 'mistral' }
    ],
    custom: []
  };

  let cachedModelsByProvider = {};

  function getActiveModel() {
    if (!UI.settingModelSelect) return '';
    if (UI.settingModelSelect.value === '__custom__') {
      return (UI.settingModelCustom?.value || '').trim();
    }
    return (UI.settingModelSelect.value || '').trim() || (UI.settingModelCustom?.value || '').trim();
  }

  function renderModelDropdown(provider, targetModel = '') {
    if (!UI.settingModelSelect) return;
    const currentVal = targetModel || getActiveModel() || '';
    const presets = DEFAULT_MODELS_BY_PROVIDER[provider] || [];
    const cached = cachedModelsByProvider[provider] || [];

    const seen = new Set();
    const options = [];

    // 1. Thêm models đã tải từ server API
    cached.forEach(m => {
      const id = typeof m === 'string' ? m : m.id;
      const label = typeof m === 'string' ? m : (m.label || m.name || m.id);
      if (id && !seen.has(id)) {
        seen.add(id);
        options.push({ id, label: `🌐 ${label}` });
      }
    });

    // 2. Thêm models preset mặc định
    presets.forEach(m => {
      if (m.id && !seen.has(m.id)) {
        seen.add(m.id);
        options.push(m);
      }
    });

    // 3. Nếu model hiện tại chưa có trong list, thêm vào đầu
    if (currentVal && currentVal !== '__custom__' && !seen.has(currentVal)) {
      seen.add(currentVal);
      options.unshift({ id: currentVal, label: `⭐ ${currentVal} (Đang dùng)` });
    }

    UI.settingModelSelect.innerHTML = '';
    options.forEach(opt => {
      const el = document.createElement('option');
      el.value = opt.id;
      el.textContent = opt.label;
      UI.settingModelSelect.appendChild(el);
    });

    // Option nhập tùy chỉnh thủ công
    const customOpt = document.createElement('option');
    customOpt.value = '__custom__';
    customOpt.textContent = '✏️ Nhập model tùy chỉnh khác...';
    UI.settingModelSelect.appendChild(customOpt);

    if (currentVal && seen.has(currentVal)) {
      UI.settingModelSelect.value = currentVal;
      if (UI.customModelGroup) UI.customModelGroup.style.display = 'none';
      if (UI.settingModelCustom) UI.settingModelCustom.value = currentVal;
    } else if (currentVal) {
      UI.settingModelSelect.value = '__custom__';
      if (UI.customModelGroup) UI.customModelGroup.style.display = 'block';
      if (UI.settingModelCustom) UI.settingModelCustom.value = currentVal;
    } else {
      if (options.length > 0) {
        UI.settingModelSelect.value = options[0].id;
        if (UI.customModelGroup) UI.customModelGroup.style.display = 'none';
        if (UI.settingModelCustom) UI.settingModelCustom.value = options[0].id;
      } else {
        UI.settingModelSelect.value = '__custom__';
        if (UI.customModelGroup) UI.customModelGroup.style.display = 'block';
      }
    }
  }

  function setActiveModel(val) {
    const provider = UI.settingApiProvider ? UI.settingApiProvider.value : 'gemini';
    renderModelDropdown(provider, val);
  }

  // Khởi tạo proxy UI.settingModel để tương thích 100% với các hàm khác
  UI.settingModel = {
    get value() {
      return getActiveModel();
    },
    set value(v) {
      setActiveModel(v);
    }
  };

  if (UI.settingModelSelect) {
    UI.settingModelSelect.addEventListener('change', () => {
      if (UI.settingModelSelect.value === '__custom__') {
        if (UI.customModelGroup) UI.customModelGroup.style.display = 'block';
        if (UI.settingModelCustom) UI.settingModelCustom.focus();
      } else {
        if (UI.customModelGroup) UI.customModelGroup.style.display = 'none';
        if (UI.settingModelCustom) UI.settingModelCustom.value = UI.settingModelSelect.value;
      }
    });
  }

  if (UI.settingModelCustom) {
    UI.settingModelCustom.addEventListener('input', () => {
      if (UI.settingModelSelect && UI.settingModelSelect.value !== '__custom__') {
        UI.settingModelSelect.value = '__custom__';
      }
    });
  }

  function updateEndpointVisibility() {
    const provider = UI.settingApiProvider ? UI.settingApiProvider.value : 'gemini';
    if (UI.settingApiEndpointGroup) {
      if (provider === 'custom') {
        UI.settingApiEndpointGroup.style.display = 'flex';
      } else {
        UI.settingApiEndpointGroup.style.display = 'none';
      }
    }
  }

  if (UI.settingApiProvider) {
    UI.settingApiProvider.addEventListener('change', () => {
      const selected = UI.settingApiProvider.value;
      const preset = API_PRESETS[selected];
      if (preset) {
        if (selected !== 'custom') {
          if (UI.settingApiEndpoint) UI.settingApiEndpoint.value = preset.endpoint;
          if (UI.settingModel) UI.settingModel.value = preset.model;
          renderModelDropdown(selected, preset.model);
        } else {
          if (UI.settingApiEndpoint && !UI.settingApiEndpoint.value.trim()) {
            UI.settingApiEndpoint.value = preset.endpoint;
          }
          renderModelDropdown(selected, getActiveModel() || '');
        }
        if (UI.settingApiEndpoint) UI.settingApiEndpoint.placeholder = preset.placeholderEndpoint;
        if (UI.settingApiKey) UI.settingApiKey.placeholder = preset.keyPlaceholder;
      }
      updateEndpointVisibility();
    });
  }

  // Nút lấy danh sách models (OpenAI-compatible /models)
  if (UI.btnFetchModels) {
    UI.btnFetchModels.addEventListener('click', async () => {
      const provider = UI.settingApiProvider ? UI.settingApiProvider.value : 'gemini';
      const key = (UI.settingApiKey?.value || '').trim();
      let rawEndpoint = (UI.settingApiEndpoint?.value || '').trim();

      if (provider === 'custom' && !rawEndpoint) {
        rawEndpoint = 'http://localhost:20128/v1';
        if (UI.settingApiEndpoint) UI.settingApiEndpoint.value = rawEndpoint;
      } else if (!rawEndpoint) {
        if (provider === 'openai') rawEndpoint = 'https://api.openai.com/v1';
        else if (provider === 'deepseek') rawEndpoint = 'https://api.deepseek.com/v1';
        else if (provider === 'openrouter') rawEndpoint = 'https://openrouter.ai/api/v1';
        else if (provider === 'ollama') rawEndpoint = 'http://localhost:11434/v1';
      }

      const showStatus = (text, isError = false) => {
        if (!UI.fetchModelsStatus) return;
        UI.fetchModelsStatus.style.display = 'block';
        UI.fetchModelsStatus.style.color = isError ? '#f87171' : '#34d399';
        UI.fetchModelsStatus.textContent = text;
      };

      showStatus('⏳ Đang tải danh sách model...');

      try {
        let modelIds = [];

        if (provider === 'gemini') {
          if (!key) throw new Error('Cần nhập API Key để lấy danh sách model Gemini.');
          const url = `https://generativelanguage.googleapis.com/v1beta/models?key=${encodeURIComponent(key)}`;
          const res = await fetch(url);
          if (!res.ok) {
            const errJson = await res.json().catch(() => ({}));
            throw new Error(`Gemini API (${res.status}): ${errJson.error?.message || res.statusText}`);
          }
          const data = await res.json();
          if (Array.isArray(data.models)) {
            modelIds = data.models
              .filter(m => !m.supportedGenerationMethods || m.supportedGenerationMethods.includes('generateContent'))
              .map(m => m.name.replace(/^models\//, ''));
          }
        } else {
          // Chuẩn OpenAI-compatible /models
          const cleanEndpoint = (rawEndpoint || 'http://localhost:20128/v1').replace(/\/+$/, '');
          let modelsUrl = '';
          if (cleanEndpoint.endsWith('/chat/completions')) {
            modelsUrl = cleanEndpoint.replace(/\/chat\/completions$/, '/models');
          } else if (cleanEndpoint.endsWith('/models')) {
            modelsUrl = cleanEndpoint;
          } else if (cleanEndpoint.endsWith('/v1')) {
            modelsUrl = `${cleanEndpoint}/models`;
          } else {
            modelsUrl = `${cleanEndpoint}/v1/models`;
          }

          const headers = {};
          if (key) {
            headers['Authorization'] = `Bearer ${key}`;
          }
          if (modelsUrl.includes('openrouter.ai')) {
            headers['HTTP-Referer'] = 'https://edux.cmcu.edu.vn';
            headers['X-Title'] = 'EDUX Slayers';
          }

          const res = await fetch(modelsUrl, { method: 'GET', headers });
          if (!res.ok) {
            const errJson = await res.json().catch(() => ({}));
            throw new Error(`API (${res.status}): ${errJson.error?.message || errJson.message || res.statusText}`);
          }

          const resJson = await res.json();
          if (Array.isArray(resJson)) {
            modelIds = resJson.map(m => typeof m === 'string' ? m : (m.id || m.name)).filter(Boolean);
          } else if (Array.isArray(resJson?.data)) {
            modelIds = resJson.data.map(m => typeof m === 'string' ? m : (m.id || m.name)).filter(Boolean);
          } else if (Array.isArray(resJson?.models)) {
            modelIds = resJson.models.map(m => typeof m === 'string' ? m : (m.name || m.id)).filter(Boolean);
          }
        }

        if (modelIds.length === 0) {
          showStatus('⚠️ Server không trả về danh sách model.', true);
          return;
        }

        // Cập nhật datalist cho ô input Model
        if (UI.modelDatalist) {
          UI.modelDatalist.innerHTML = '';
          modelIds.forEach(id => {
            const opt = document.createElement('option');
            opt.value = id;
            UI.modelDatalist.appendChild(opt);
          });
        }
        // Lưu vào cache theo provider và cập nhật Dropdown
        cachedModelsByProvider[provider] = modelIds;
        chrome.storage.local.set({ cachedModelsByProvider });
        renderModelDropdown(provider, modelIds[0]);

        // Tự động gán model đầu tiên nếu ô nhập trống
        if (UI.settingModel && !UI.settingModel.value.trim()) {
          UI.settingModel.value = modelIds[0];
        }

        showStatus(`✓ Đã tải ${modelIds.length} models! (Bấm đúp ô nhập để chọn)`);
        addLog(UI.testLog, `✓ Đã cập nhật danh sách ${modelIds.length} models từ server API.`, 'success');
        showStatus(`✓ Đã tải ${modelIds.length} models vào menu dropdown!`);
        addLog(UI.testLog, `✓ Đã cập nhật ${modelIds.length} models từ server API vào menu dropdown.`, 'success');
      } catch (err) {
        showStatus(`❌ Lỗi: ${err.message}`, true);
        addLog(UI.testLog, `Không thể lấy danh sách model: ${err.message}`, 'error');
      }
    });
  }

  const settings = await chrome.storage.local.get([
    'delayMs',
    'autoNext',
    'autoSubmit',
    'useAiSlide',
    'savedAnswers',
    'slideStats',
    'lastExamData',
    'apiProvider',
    'apiEndpoint',
    'apiKey',
    'apiModel',
    'cachedModelsByProvider'
  ]);

  if (settings.cachedModelsByProvider && typeof settings.cachedModelsByProvider === 'object') {
    cachedModelsByProvider = settings.cachedModelsByProvider;
  }

  UI.settingDelay.value = settings.delayMs !== undefined ? settings.delayMs : 100;
  UI.settingAutoNext.checked = settings.autoNext !== undefined ? settings.autoNext : true;
  if (UI.settingAutoSubmit) UI.settingAutoSubmit.checked = settings.autoSubmit !== undefined ? settings.autoSubmit : true;
  if (UI.settingUseAiSlide) UI.settingUseAiSlide.checked = settings.useAiSlide !== undefined ? settings.useAiSlide : true;
  if (UI.settingApiProvider && settings.apiProvider) {
    UI.settingApiProvider.value = settings.apiProvider;
    const preset = API_PRESETS[settings.apiProvider];
    if (preset) {
      if (UI.settingApiEndpoint) UI.settingApiEndpoint.placeholder = preset.placeholderEndpoint;
      if (UI.settingApiKey) UI.settingApiKey.placeholder = preset.keyPlaceholder;
    }
  }
  if (UI.settingApiEndpoint) {
    if (settings.apiEndpoint !== undefined && settings.apiEndpoint !== '') {
      UI.settingApiEndpoint.value = settings.apiEndpoint;
    } else if (UI.settingApiProvider && UI.settingApiProvider.value === 'custom') {
      UI.settingApiEndpoint.value = 'http://localhost:20128/v1';
    }
  }
  updateEndpointVisibility();

  if (UI.settingApiKey && settings.apiKey) UI.settingApiKey.value = settings.apiKey;
  if (UI.settingModel && settings.apiModel) UI.settingModel.value = settings.apiModel;
  const activeProvider = settings.apiProvider || 'gemini';
  renderModelDropdown(activeProvider, settings.apiModel || '');

  if (settings.savedAnswers) UI.answerInput.value = settings.savedAnswers;
  if (settings.slideStats) {
    UI.slideCount.textContent = settings.slideStats.solved || 0;
    UI.retryCount.textContent = settings.slideStats.retries || 0;
  }

  if (settings.lastExamData) {
    updateExamInfoUI(settings.lastExamData);
  }

  // Check state from content script on popup open
  const activeTab = await getActiveTab();
  if (activeTab && activeTab.url && (activeTab.url.includes('cmcu.edu.vn') || activeTab.url.includes('edux'))) {
    // Đảm bảo network interceptor luôn hoạt động trong MAIN world
    chrome.scripting.executeScript({
      target: { tabId: activeTab.id },
      files: ['injected.js'],
      world: 'MAIN'
    }).catch(() => {});

    try {
      const response = await sendTabMessage(activeTab.id, { action: 'GET_STATUS' });
      if (response) {
        if (response.isSlideRunning) {
          UI.btnStartSlide.classList.add('hidden');
          UI.btnStopSlide.classList.remove('hidden');
          setStatus('Đang giải Slide...', 'running');
        }
        if (response.examData) {
          updateExamInfoUI(response.examData);
        }
      }
    } catch (e) {
      addLog(UI.slideLog, 'Mở slide hoặc bài tập để bắt đầu.', 'info');
    }
  } else {
    addLog(UI.slideLog, 'Vui lòng chuyển sang trang EDUX để sử dụng.', 'warn');
  }

  // Listen for progress updates from content script
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === 'SLIDE_LOG') {
      addLog(UI.slideLog, msg.message, msg.logType || 'info');
      if (msg.solvedCount !== undefined) UI.slideCount.textContent = msg.solvedCount;
      if (msg.retryCount !== undefined) UI.retryCount.textContent = msg.retryCount;
    } else if (msg.type === 'TEST_LOG') {
      addLog(UI.testLog, msg.message, msg.logType || 'info');
    } else if (msg.type === 'EXAM_DATA_READY') {
      updateExamInfoUI(msg.payload);
      addLog(UI.testLog, '📡 Đã bắt được đề bài tập từ hệ thống!', 'success');
    } else if (msg.type === 'SLIDE_STATUS_CHANGE') {
      if (msg.isRunning) {
        UI.btnStartSlide.classList.add('hidden');
        UI.btnStopSlide.classList.remove('hidden');
        setStatus('Đang giải Slide...', 'running');
      } else {
        UI.btnStartSlide.classList.remove('hidden');
        UI.btnStopSlide.classList.add('hidden');
        setStatus('Sẵn sàng', 'idle');
      }
    }
  });

  // =========================================================================
  // 6. Slide Brute-force Actions
  // =========================================================================
  UI.btnStartSlide.addEventListener('click', async () => {
    const tab = await getActiveTab();
    if (!tab) return;
    try {
      const useAi = UI.settingUseAiSlide ? UI.settingUseAiSlide.checked : true;
      await sendTabMessage(tab.id, {
        action: 'START_SLIDE_BRUTEFORCE',
        config: {
          delayMs: parseInt(UI.settingDelay.value, 10) || 100,
          autoNext: UI.settingAutoNext.checked,
          useAi
        }
      });
      UI.btnStartSlide.classList.add('hidden');
      UI.btnStopSlide.classList.remove('hidden');
      setStatus('Đang giải Slide...', 'running');
      addLog(
        UI.slideLog,
        useAi
          ? '🧠 Đã bật giải Slide bằng AI (Chờ AI phân tích câu hỏi -> Click)'
          : '⚡ Đã bật giải Slide chế độ thử sai nhanh.',
        'success'
      );
    } catch (err) {
      addLog(UI.slideLog, 'Lỗi kết nối với trang EDUX: ' + err.message, 'error');
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

  // =========================================================================
  // 7. AI Solver Service (Hỗ trợ tùy chỉnh nguồn API: Gemini, OpenAI, DeepSeek, OpenRouter, Ollama...)
  // =========================================================================
  async function solveWithAI(promptContent, apiKey, model, apiEndpoint, apiProvider) {
    const key = (apiKey || '').trim();
    const rawModel = (model || '').trim();
    let customEndpoint = (apiEndpoint || '').trim();
    const provider = (apiProvider || 'gemini').toLowerCase().trim();

    if (provider === 'custom' && !customEndpoint) {
      customEndpoint = 'http://localhost:20128/v1';
    }

    const isLocal = provider === 'ollama' || customEndpoint.includes('localhost') || customEndpoint.includes('127.0.0.1');

    if (!key && !isLocal) {
      throw new Error('Chưa cấu hình API Key. Vui lòng vào tab Cài đặt để nhập key.');
    }

    // Xác định giao thức API: Gemini format hay OpenAI Chat Completions format
    let isGemini = false;
    if (provider === 'gemini') {
      isGemini = true;
    } else if (provider === 'openai' || provider === 'deepseek' || provider === 'openrouter' || provider === 'ollama') {
      isGemini = false;
    } else {
      // provider là 'custom' hoặc auto
      if (customEndpoint) {
        if (customEndpoint.includes('googleapis.com') || customEndpoint.includes(':generateContent')) {
          isGemini = true;
        } else {
          isGemini = false;
        }
      } else if (key.startsWith('AIza') || rawModel.toLowerCase().includes('gemini') || !rawModel) {
        isGemini = true;
      }
    }

    const systemPrompt =
      'Bạn là chuyên gia khảo thí và học thuật cao cấp hàng đầu, có độ chính xác tuyệt đối 100% trong việc giải quyết các bài kiểm tra trắc nghiệm, đúng/sai, điền khuyết và tự luận.\n' +
      'Yêu cầu:\n' +
      '1. Phân tích cẩn thận từng câu hỏi và các lựa chọn loại trừ để chọn phương án đúng tuyệt đối.\n' +
      '2. Trả về JSONL một dòng duy nhất (hoặc JSON Array các object);\n' +
      '3. Mỗi phần tử có "so_cau" và "dap_an";\n' +
      '4. "dap_an" là A/B/C/D hoặc từ/cụm từ cần điền;\n' +
      '5. Với câu đúng/sai, "dap_an" là mảng giá trị Đúng/Sai theo thứ tự từng mệnh đề (ví dụ: [true, false, true, true]);\n' +
      '6. Tuyệt đối không thêm lời dẫn hay giải thích.';

    let responseText = '';

    if (isGemini) {
      const geminiModel = rawModel ? rawModel.replace(/^gemini\//i, '') : 'gemini-2.0-flash';
      let endpoint = '';

      if (customEndpoint) {
        if (customEndpoint.includes(':generateContent')) {
          endpoint = customEndpoint;
          if (key && !endpoint.includes('key=')) {
            endpoint += (endpoint.includes('?') ? '&' : '?') + `key=${encodeURIComponent(key)}`;
          }
        } else {
          const base = customEndpoint.replace(/\/+$/, '');
          if (base.endsWith('/v1beta') || base.endsWith('/v1')) {
            endpoint = `${base}/models/${geminiModel}:generateContent?key=${encodeURIComponent(key)}`;
          } else {
            endpoint = `${base}/v1beta/models/${geminiModel}:generateContent?key=${encodeURIComponent(key)}`;
          }
        }
      } else {
        endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${geminiModel}:generateContent?key=${encodeURIComponent(key)}`;
      }

      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ parts: [{ text: promptContent }] }],
          systemInstruction: { parts: [{ text: systemPrompt }] },
          generationConfig: { temperature: 0 }
        })
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(`Lỗi Gemini API (${res.status}): ${errJson.error?.message || res.statusText}`);
      }

      const resJson = await res.json();
      responseText = resJson?.candidates?.[0]?.content?.parts?.[0]?.text || '';
    } else {
      // OpenAI / OpenAI-Compatible (DeepSeek, OpenRouter, Ollama, Groq, custom proxy...)
      let openAiModel = rawModel;
      if (!openAiModel) {
        if (provider === 'deepseek') openAiModel = 'deepseek-chat';
        else if (provider === 'ollama') openAiModel = 'llama3.2';
        else openAiModel = 'gpt-4o-mini';
      }

      let endpoint = '';
      if (customEndpoint) {
        const base = customEndpoint.replace(/\/+$/, '');
        if (base.endsWith('/chat/completions')) {
          endpoint = base;
        } else if (base.endsWith('/v1')) {
          endpoint = `${base}/chat/completions`;
        } else {
          endpoint = `${base}/v1/chat/completions`;
        }
      } else {
        if (provider === 'deepseek') {
          endpoint = 'https://api.deepseek.com/v1/chat/completions';
        } else if (provider === 'openrouter') {
          endpoint = 'https://openrouter.ai/api/v1/chat/completions';
        } else if (provider === 'ollama') {
          endpoint = 'http://localhost:11434/v1/chat/completions';
        } else {
          endpoint = 'https://api.openai.com/v1/chat/completions';
        }
      }

      const headers = { 'Content-Type': 'application/json' };
      if (key) {
        headers['Authorization'] = `Bearer ${key}`;
      }
      if (endpoint.includes('openrouter.ai')) {
        headers['HTTP-Referer'] = 'https://edux.cmcu.edu.vn';
        headers['X-Title'] = 'EDUX Slayers';
      }

      const res = await fetch(endpoint, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          model: openAiModel,
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: promptContent }
          ],
          temperature: 0,
          stream: false
        })
      });

      const rawText = await res.text();
      if (!res.ok) {
        let errMessage = res.statusText;
        try {
          const errJson = JSON.parse(rawText);
          errMessage = errJson.error?.message || errJson.message || errMessage;
        } catch (e) {}
        throw new Error(`Lỗi API (${res.status}): ${errMessage}`);
      }

      if (rawText.trim().startsWith('data:') || rawText.includes('\ndata:')) {
        let accumulated = '';
        const lines = rawText.split('\n');
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;
          const dataStr = trimmed.replace(/^data:\s*/, '').trim();
          if (dataStr === '[DONE]') break;
          try {
            const chunk = JSON.parse(dataStr);
            const delta = chunk.choices?.[0]?.delta?.content || chunk.choices?.[0]?.message?.content || '';
            accumulated += delta;
          } catch (e) {}
        }
        responseText = accumulated;
      } else {
        try {
          const resJson = JSON.parse(rawText);
          const choiceMessage = resJson?.choices?.[0]?.message;
          responseText = choiceMessage?.content || choiceMessage?.reasoning_content || '';
        } catch (e) {
          responseText = rawText;
        }
      }
    }

    // Mô phỏng sanitize_ai_response()
    let cleaned = responseText.trim();
    if (cleaned.startsWith('```')) {
      const match = cleaned.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
      if (match) {
        cleaned = match[1].trim();
      }
    }
    return cleaned;
  }

  // =========================================================================
  // 8. Test Solver Actions (Bài tập)
  // =========================================================================

  // Nút 1: Mở bài tập (Bấm "Làm bài tập" trên trang)
  if (UI.btnStartExercise) {
    UI.btnStartExercise.addEventListener('click', async () => {
      const tab = await getActiveTab();
      if (!tab) return;
      try {
        addLog(UI.testLog, "Đang tìm nút 'Làm bài tập' trên trang...", 'info');
        const res = await sendTabMessage(tab.id, { action: 'START_EXERCISE' });
        if (res && res.success) {
          if (res.questions) {
            updateExamInfoUI(res.questions);
            addLog(UI.testLog, `🎉 Cửa sổ bài tập đã sẵn sàng (${res.questions.total_questions || 0} câu)!`, 'success');
          } else {
            addLog(UI.testLog, res.opened ? 'Cửa sổ bài tập đã sẵn sàng!' : 'Đã bấm nút làm bài tập.', 'success');
          }
        } else {
          addLog(UI.testLog, res?.message || 'Không tìm thấy nút làm bài tập.', 'warn');
        }
      } catch (err) {
        addLog(UI.testLog, 'Lỗi: ' + err.message, 'error');
      }
    });
  }

  // Nút 2: Copy Prompt câu hỏi chuẩn theo EDUX-TEST-SOLVER
  UI.btnExtractQuestions.addEventListener('click', async () => {
    const tab = await getActiveTab();
    if (!tab) return;

    try {
      // 1. Kiểm tra xem bài tập đã mở trên trang chưa
      const checkRes = await sendTabMessage(tab.id, { action: 'CHECK_EXAM_OPEN' });
      let res = null;

      if (!checkRes || !checkRes.isOpen) {
        addLog(UI.testLog, "Bài tập chưa mở. Đang bấm 'Làm bài tập' và bắt đề...", 'info');
        const startRes = await sendTabMessage(tab.id, { action: 'START_EXERCISE' });
        if (startRes && startRes.questions) {
          res = startRes;
          updateExamInfoUI(startRes.questions);
        } else if (!startRes || !startRes.opened) {
          addLog(UI.testLog, startRes?.message || "Không thể mở bài tập trên trang.", 'warn');
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
          'success'
        );
      } else {
        addLog(UI.testLog, 'Chưa tìm thấy câu hỏi bài tập nào trên trang.', 'warn');
      }
    } catch (err) {
      addLog(UI.testLog, 'Lỗi trích xuất câu hỏi: ' + err.message, 'error');
    }
  });

  // Nút 3: Giải tự động bằng AI (API Key)
  if (UI.btnSolveAI) {
    UI.btnSolveAI.addEventListener('click', async () => {
      const tab = await getActiveTab();
      if (!tab) return;

      const apiKey = (UI.settingApiKey?.value || '').trim();
      const model = (UI.settingModel?.value || '').trim();
      const apiEndpoint = (UI.settingApiEndpoint?.value || '').trim();
      const apiProvider = (UI.settingApiProvider?.value || 'gemini').trim();

      const isLocal = apiProvider === 'ollama' || apiEndpoint.includes('localhost') || apiEndpoint.includes('127.0.0.1');

      if (!apiKey && !isLocal) {
        addLog(UI.testLog, '⚠️ Chưa có API Key! Đang chuyển sang tab Cài đặt để nhập key...', 'warn');
        const settingsTabBtn = document.querySelector('.tab-btn[data-tab="tab-settings"]');
        if (settingsTabBtn) settingsTabBtn.click();
        return;
      }

      try {
        // BƯỚC 1: Đảm bảo bài tập được mở trên trang trước khi giải
        const checkRes = await sendTabMessage(tab.id, { action: 'CHECK_EXAM_OPEN' });
        let extRes = null;

        if (!checkRes || !checkRes.isOpen) {
          addLog(UI.testLog, "Bài tập chưa mở. Đang bấm 'Làm bài tập' và bắt đề...", 'info');
          const startRes = await sendTabMessage(tab.id, { action: 'START_EXERCISE' });
          if (startRes && startRes.questions) {
            extRes = startRes;
            updateExamInfoUI(startRes.questions);
          } else if (!startRes || !startRes.opened) {
            addLog(UI.testLog, startRes?.message || "Không thể mở bài tập trên trang.", 'warn');
            return;
          }
        }

        // BƯỚC 2: Trích xuất đề bài tập (nếu chưa có từ startRes)
        if (!extRes || !extRes.promptText) {
          addLog(UI.testLog, 'Đang trích xuất đề bài tập...', 'info');
          extRes = await sendTabMessage(tab.id, { action: 'EXTRACT_QUESTIONS' });
        }

        if (!extRes || !extRes.promptText) {
          addLog(UI.testLog, "Không tìm thấy đề bài tập. Hãy kiểm tra giao diện bài tập!", 'warn');
          return;
        }

        const qCount = extRes.questions?.total_questions || 0;
        updateExamInfoUI(extRes.questions);
        const displayModel = model || (apiProvider === 'gemini' ? 'Gemini' : apiProvider === 'deepseek' ? 'DeepSeek' : apiProvider === 'ollama' ? 'Ollama' : 'AI');
        addLog(UI.testLog, `Đang gửi ${qCount} câu tới AI (${displayModel})...`, 'info');
        setStatus('AI đang giải bài...', 'running');

        const aiAnswers = await solveWithAI(extRes.promptText, apiKey, model, apiEndpoint, apiProvider);
        UI.answerInput.value = aiAnswers;
        await chrome.storage.local.set({ savedAnswers: aiAnswers });

        addLog(UI.testLog, '✓ AI đã giải xong! Bắt đầu tự động điền đáp án...', 'success');

        const fillRes = await sendTabMessage(tab.id, {
          action: 'FILL_TEST_ANSWERS',
          answersText: aiAnswers,
          options: { autoSubmit: UI.settingAutoSubmit ? UI.settingAutoSubmit.checked : true }
        });

        setStatus('Sẵn sàng', 'idle');
        if (fillRes && fillRes.success) {
          addLog(UI.testLog, `🎉 Hoàn tất! Đã điền xong ${fillRes.filledCount} câu bài tập.`, 'success');
        } else {
          addLog(UI.testLog, `Thông báo: ${fillRes?.message || 'Không thể điền bài.'}`, 'warn');
        }
      } catch (err) {
        setStatus('Sẵn sàng', 'idle');
        addLog(UI.testLog, 'Lỗi giải AI: ' + err.message, 'error');
      }
    });
  }

  // Nút 4: Dán đáp án từ Clipboard
  if (UI.btnPasteClipboard) {
    UI.btnPasteClipboard.addEventListener('click', async () => {
      try {
        const text = await navigator.clipboard.readText();
        if (!text || !text.trim()) {
          addLog(UI.testLog, 'Clipboard đang trống!', 'warn');
          return;
        }
        UI.answerInput.value = text.trim();
        await chrome.storage.local.set({ savedAnswers: text.trim() });
        addLog(UI.testLog, '✓ Đã dán đáp án từ Clipboard.', 'success');
      } catch (e) {
        addLog(UI.testLog, 'Không thể đọc Clipboard: ' + e.message, 'error');
      }
    });
  }

  // Nút 5: Xem/ẩn Prompt xem trước
  if (UI.btnTogglePrompt) {
    UI.btnTogglePrompt.addEventListener('click', async () => {
      if (!UI.promptPreviewCard) return;
      if (UI.promptPreviewCard.style.display === 'none') {
        if (!UI.promptPreviewBox.value.trim()) {
          const tab = await getActiveTab();
          if (tab) {
            const res = await sendTabMessage(tab.id, { action: 'EXTRACT_QUESTIONS' });
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

  // Nút 6: Bắt đầu điền bài tập
  UI.btnFillAnswers.addEventListener('click', async () => {
    const rawAnswers = UI.answerInput.value.trim();
    if (!rawAnswers) {
      addLog(UI.testLog, 'Vui lòng nhập hoặc dán danh sách đáp án trước!', 'warn');
      return;
    }

    await chrome.storage.local.set({ savedAnswers: rawAnswers });

    const tab = await getActiveTab();
    if (!tab) return;

    try {
      // Đảm bảo bài tập đang mở trước khi điền
      const checkRes = await sendTabMessage(tab.id, { action: 'CHECK_EXAM_OPEN' });
      if (!checkRes || !checkRes.isOpen) {
        addLog(UI.testLog, "Đang mở bài tập trên trang để chuẩn bị điền...", 'info');
        const startRes = await sendTabMessage(tab.id, { action: 'START_EXERCISE' });
        if (!startRes || !startRes.opened) {
          addLog(UI.testLog, startRes?.message || "Không thể mở bài tập trên trang.", 'warn');
          return;
        }
        await new Promise((r) => setTimeout(r, 400));
      }

      addLog(UI.testLog, 'Đang gửi đáp án tới trang bài tập...', 'info');
      const res = await sendTabMessage(tab.id, {
        action: 'FILL_TEST_ANSWERS',
        answersText: rawAnswers,
        options: { autoSubmit: UI.settingAutoSubmit ? UI.settingAutoSubmit.checked : true }
      });
      if (res && res.success) {
        addLog(UI.testLog, `Hoàn tất! Đã điền ${res.filledCount} câu bài tập.`, 'success');
      } else {
        addLog(UI.testLog, `Thông báo: ${res?.message || 'Không thể điền bài tập.'}`, 'warn');
      }
    } catch (err) {
      addLog(UI.testLog, 'Lỗi: Không tìm thấy trang bài tập EDUX.', 'error');
    }
  });

  // =========================================================================
  // 9. Settings Actions
  // =========================================================================
  UI.btnSaveSettings.addEventListener('click', async () => {
    const providerVal = UI.settingApiProvider ? UI.settingApiProvider.value : 'gemini';
    let endpointVal = UI.settingApiEndpoint ? UI.settingApiEndpoint.value.trim() : '';
    if (providerVal === 'custom' && !endpointVal) {
      endpointVal = 'http://localhost:20128/v1';
      if (UI.settingApiEndpoint) UI.settingApiEndpoint.value = endpointVal;
    }

    const newSettings = {
      delayMs: parseInt(UI.settingDelay.value, 10) || 100,
      autoNext: UI.settingAutoNext.checked,
      autoSubmit: UI.settingAutoSubmit ? UI.settingAutoSubmit.checked : true,
      useAiSlide: UI.settingUseAiSlide ? UI.settingUseAiSlide.checked : true,
      useAi: UI.settingUseAiSlide ? UI.settingUseAiSlide.checked : true,
      apiProvider: providerVal,
      apiEndpoint: endpointVal,
      apiKey: UI.settingApiKey ? UI.settingApiKey.value.trim() : '',
      apiModel: UI.settingModel ? UI.settingModel.value.trim() : 'gemini-2.0-flash',
      cachedModelsByProvider
    };

    await chrome.storage.local.set(newSettings);

    const tab = await getActiveTab();
    if (tab) {
      sendTabMessage(tab.id, {
        action: 'UPDATE_SETTINGS',
        settings: newSettings
      }).catch(() => {});
    }

    addLog(UI.slideLog, 'Đã lưu cấu hình mới!', 'success');
    addLog(UI.testLog, 'Đã cập nhật cấu hình bài tập & AI!', 'success');
  });

  // =========================================================================
  // 10. Exercise Scores Actions
  // =========================================================================
  async function loadExerciseScores() {
    const tab = await getActiveTab();
    if (!tab || !tab.url || !tab.url.includes('cmcu.edu.vn')) {
      if (UI.scoresSubjectTitle) UI.scoresSubjectTitle.textContent = 'Vui lòng mở trang EDUX';
      return;
    }

    const isStudentDashboard = tab.url.includes('/student') && !tab.url.includes('id=');

    try {
      if (UI.scoresSubjectTitle) UI.scoresSubjectTitle.textContent = 'Đang quét dữ liệu tiến độ...';

      if (isStudentDashboard) {
        // Load all subjects progress
        const res = await sendTabMessage(tab.id, { action: 'GET_ALL_SUBJECTS_PROGRESS' });
        if (!res || !res.success || !Array.isArray(res.subjects)) {
          if (UI.scoresSubjectTitle) UI.scoresSubjectTitle.textContent = 'Không thể lấy dữ liệu học phần.';
          return;
        }

        const subjects = res.subjects;
        let totalDoneExams = 0;
        let totalExams = 0;
        let totalDoneSlides = 0;
        let totalSlides = 0;
        let totalPendingExams = 0;

        subjects.forEach((s) => {
          totalDoneExams += s.doneExams || 0;
          totalExams += s.totalExams || 0;
          totalDoneSlides += s.doneSlides || 0;
          totalSlides += s.totalSlides || 0;
          totalPendingExams += s.pendingExams || 0;
        });

        if (UI.scoresSubjectTitle) {
          UI.scoresSubjectTitle.textContent = `Tổng quan: ${subjects.length} Học phần`;
        }
        if (UI.scoresCompleted) {
          UI.scoresCompleted.textContent = `${totalDoneExams}/${totalExams} bài`;
        }
        if (UI.scoresHighest) {
          UI.scoresHighest.textContent = `${totalDoneSlides}/${totalSlides} slide`;
        }

        if (UI.scoresAlertBox) {
          if (totalPendingExams > 0) {
            UI.scoresAlertBox.style.display = 'block';
            UI.scoresAlertBox.className = 'log-entry warn';
            UI.scoresAlertBox.innerHTML = `⚠️ Toàn bộ học phần: Còn <strong>${totalPendingExams}</strong> bài tập AI chưa làm!`;
          } else {
            UI.scoresAlertBox.style.display = 'block';
            UI.scoresAlertBox.className = 'log-entry success';
            UI.scoresAlertBox.innerHTML = `🎉 Tuyệt vời! Bạn đã hoàn thành 100% bài tập của tất cả môn học!`;
          }
        }

        if (UI.scoresList) {
          UI.scoresList.innerHTML = '';
          subjects.forEach((s) => {
            const item = document.createElement('div');
            const isDone = s.isAllDone;
            item.className = `log-entry ${isDone ? 'success' : s.pendingExams > 0 ? 'warn' : 'info'}`;
            item.style.display = 'flex';
            item.style.flexDirection = 'column';
            item.style.gap = '4px';
            item.style.padding = '8px 10px';

            const badgeText = isDone
              ? '<span style="color: #059669; font-weight: bold;">✓ 100%</span>'
              : s.pendingExams > 0
              ? `<span style="color: #e11d48; font-weight: bold;">⚠️ Còn ${s.pendingExams} bài</span>`
              : `<span style="color: #d97706; font-weight: bold;">📖 Còn ${s.pendingSlides} slide</span>`;

            item.innerHTML = `
              <div style="display: flex; justify-content: space-between; align-items: center;">
                <strong style="color: #0f172a; font-size: 12px;" title="${s.name}">
                  ${s.code ? `[${s.code}] ` : ''}${s.name}
                </strong>
                ${badgeText}
              </div>
              <div style="display: flex; gap: 12px; font-size: 11px; color: #475569;">
                <span>🖥️ Slide: <strong>${s.doneSlides}/${s.totalSlides}</strong> (${s.slidePercent}%)</span>
                <span>📝 Bài tập: <strong>${s.doneExams}/${s.totalExams}</strong> (${s.examPercent}%)</span>
              </div>
            `;
            UI.scoresList.appendChild(item);
          });
        }
        return;
      }

      // Single subject page logic
      const res = await sendTabMessage(tab.id, { action: 'GET_EXERCISE_SCORES' });
      if (!res || !res.success || !Array.isArray(res.models)) {
        if (UI.scoresSubjectTitle) UI.scoresSubjectTitle.textContent = res?.message || 'Không tìm thấy dữ liệu bài tập.';
        return;
      }

      const models = res.models;
      const examModels = models.filter((m) => m.exist_exam);
      const totalExams = examModels.length;
      const completedExams = examModels.filter((m) => m.highest_score !== null && m.highest_score !== undefined);
      const pendingExams = examModels.filter((m) => m.highest_score === null || m.highest_score === undefined);

      const scores = completedExams
        .map((m) => parseFloat(m.highest_score))
        .filter((s) => !isNaN(s));
      const maxScore = scores.length ? Math.max(...scores).toFixed(2).replace(/\.00$/, '') : '--';

      if (UI.scoresSubjectTitle) {
        UI.scoresSubjectTitle.textContent = `Môn học: ${res.subjectId ? res.subjectId.slice(0, 8) + '...' : 'Hiện tại'}`;
      }
      if (UI.scoresCompleted) UI.scoresCompleted.textContent = `${completedExams.length}/${totalExams}`;
      if (UI.scoresHighest) UI.scoresHighest.textContent = maxScore !== '--' ? `${maxScore}/10` : '--';

      // Alert box
      if (UI.scoresAlertBox) {
        if (pendingExams.length > 0) {
          UI.scoresAlertBox.style.display = 'block';
          UI.scoresAlertBox.className = 'log-entry warn';
          UI.scoresAlertBox.innerHTML = `⚠️ Cảnh báo: Bạn còn <strong>${pendingExams.length}</strong> bài tập chưa có điểm!`;
        } else if (totalExams > 0) {
          UI.scoresAlertBox.style.display = 'block';
          UI.scoresAlertBox.className = 'log-entry success';
          UI.scoresAlertBox.innerHTML = `🎉 Xuất sắc! Đã hoàn thành 100% bài tập môn này!`;
        } else {
          UI.scoresAlertBox.style.display = 'none';
        }
      }

      // Render exercise list
      if (UI.scoresList) {
        UI.scoresList.innerHTML = '';
        examModels.forEach((m) => {
          const item = document.createElement('div');
          const hasScore = m.highest_score !== null && m.highest_score !== undefined;
          item.className = `log-entry ${hasScore ? 'success' : 'warn'}`;
          item.style.display = 'flex';
          item.style.justifyContent = 'space-between';
          item.style.alignItems = 'center';
          item.style.gap = '8px';

          const scoreVal = hasScore ? parseFloat(m.highest_score) : null;
          const scoreDisplay =
            scoreVal !== null && !isNaN(scoreVal) ? scoreVal.toFixed(2).replace(/\.00$/, '') : m.highest_score;
          const scoreText = hasScore
            ? `<strong style="color: #10b981;">🏆 ${scoreDisplay}/10</strong>`
            : `<span style="color: #f59e0b; font-weight: bold;">⚠️ Chưa làm</span>`;

          item.innerHTML = `
            <span style="flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${m.title}">
              ${m.title}
            </span>
            <span>${scoreText}</span>
          `;
          UI.scoresList.appendChild(item);
        });

        if (examModels.length === 0) {
          UI.scoresList.innerHTML = '<div class="log-entry info">Môn học này không có bài tập AI.</div>';
        }
      }
    } catch (err) {
      if (UI.scoresSubjectTitle) UI.scoresSubjectTitle.textContent = 'Lỗi kết nối trang EDUX';
      if (UI.scoresList) UI.scoresList.innerHTML = `<div class="log-entry error">Không thể lấy điểm số: ${err.message}</div>`;
    }
  }

  if (UI.btnRefreshScores) {
    UI.btnRefreshScores.addEventListener('click', loadExerciseScores);
  }
});
