// Cấu hình AI (API profiles): danh sách, form thêm/sửa, chọn model, gán cấu hình cho từng chức năng.
// Storage: aiProfiles = [{ id, provider, endpoint, apiKey, model, reasoningEffort }], aiAssign = { slide, exam }
import { UI, addLog } from "./ui.js";
import { state } from "./state.js";
import {
  PROVIDERS,
  REASONING_LABELS,
  displayModel,
  reasoningEffortsFor,
  validReasoningEffort,
} from "../shared/providers.js";

const AI_FUNCTIONS = ["slide", "exam"];

// Module khác (VD tab Slide) cần cập nhật khi danh sách / gán cấu hình thay đổi
const changeListeners = [];
export function onProfilesChange(fn) {
  changeListeners.push(fn);
}

export function profileLabel(p) {
  const keyTail = p.apiKey ? ` …${p.apiKey.slice(-4)}` : "";
  const effort = validReasoningEffort(p.provider, p.reasoningEffort);
  const tier = effort ? ` (${effort})` : "";
  return `${PROVIDERS[p.provider]?.name || p.provider} · ${displayModel(p.model, p.provider)}${tier}${keyTail}`;
}

export function getAssignedProfile(fn) {
  return state.aiProfiles.find((p) => p.id === state.aiAssign[fn]) || state.aiProfiles[0] || null;
}

// =========================================================================
// Chọn model: dropdown gồm model tải từ server + model gợi ý sẵn + nhập tay
// =========================================================================
function getActiveModel() {
  if (!UI.settingModelSelect) return "";
  if (UI.settingModelSelect.value === "__custom__") {
    return (UI.settingModelCustom?.value || "").trim();
  }
  return (
    (UI.settingModelSelect.value || "").trim() ||
    (UI.settingModelCustom?.value || "").trim()
  );
}

function renderModelDropdown(provider, targetModel = "") {
  if (!UI.settingModelSelect) return;
  const currentVal = targetModel || getActiveModel() || "";
  const presets = PROVIDERS[provider]?.models || [];
  const cached = state.cachedModelsByProvider[provider] || [];

  const seen = new Set();
  const options = [];

  // 1. Thêm models đã tải từ server API
  cached.forEach((m) => {
    const id = typeof m === "string" ? m : m.id;
    const label = typeof m === "string" ? m : m.label || m.name || m.id;
    if (id && !seen.has(id)) {
      seen.add(id);
      options.push({ id, label: `🌐 ${label}` });
    }
  });

  // 2. Thêm models preset mặc định
  presets.forEach((m) => {
    if (m.id && !seen.has(m.id)) {
      seen.add(m.id);
      options.push(m);
    }
  });

  // 3. Nếu model hiện tại chưa có trong list, thêm vào đầu
  if (currentVal && currentVal !== "__custom__" && !seen.has(currentVal)) {
    seen.add(currentVal);
    options.unshift({
      id: currentVal,
      label: `⭐ ${currentVal} (Đang dùng)`,
    });
  }

  UI.settingModelSelect.innerHTML = "";
  options.forEach((opt) => {
    const el = document.createElement("option");
    el.value = opt.id;
    el.textContent = opt.label;
    UI.settingModelSelect.appendChild(el);
  });

  // Option nhập tùy chỉnh thủ công
  const customOpt = document.createElement("option");
  customOpt.value = "__custom__";
  customOpt.textContent = "✏️ Nhập model tùy chỉnh khác...";
  UI.settingModelSelect.appendChild(customOpt);

  if (currentVal && seen.has(currentVal)) {
    UI.settingModelSelect.value = currentVal;
    if (UI.customModelGroup) UI.customModelGroup.style.display = "none";
    if (UI.settingModelCustom) UI.settingModelCustom.value = currentVal;
  } else if (currentVal) {
    UI.settingModelSelect.value = "__custom__";
    if (UI.customModelGroup) UI.customModelGroup.style.display = "block";
    if (UI.settingModelCustom) UI.settingModelCustom.value = currentVal;
  } else {
    if (options.length > 0) {
      UI.settingModelSelect.value = options[0].id;
      if (UI.customModelGroup) UI.customModelGroup.style.display = "none";
      if (UI.settingModelCustom) UI.settingModelCustom.value = options[0].id;
    } else {
      UI.settingModelSelect.value = "__custom__";
      if (UI.customModelGroup) UI.customModelGroup.style.display = "block";
    }
  }
}

function setActiveModel(val) {
  const provider = UI.settingApiProvider
    ? UI.settingApiProvider.value
    : "gemini";
  renderModelDropdown(provider, val);
}


if (UI.settingModelSelect) {
  UI.settingModelSelect.addEventListener("change", () => {
    if (UI.settingModelSelect.value === "__custom__") {
      if (UI.customModelGroup) UI.customModelGroup.style.display = "block";
      if (UI.settingModelCustom) UI.settingModelCustom.focus();
    } else {
      if (UI.customModelGroup) UI.customModelGroup.style.display = "none";
      if (UI.settingModelCustom)
        UI.settingModelCustom.value = UI.settingModelSelect.value;
    }
  });
}

if (UI.settingModelCustom) {
  UI.settingModelCustom.addEventListener("input", () => {
    if (
      UI.settingModelSelect &&
      UI.settingModelSelect.value !== "__custom__"
    ) {
      UI.settingModelSelect.value = "__custom__";
    }
  });
}

function updateEndpointVisibility() {
  const provider = UI.settingApiProvider
    ? UI.settingApiProvider.value
    : "gemini";
  if (UI.settingApiEndpointGroup) {
    if (provider === "custom") {
      UI.settingApiEndpointGroup.style.display = "flex";
    } else {
      UI.settingApiEndpointGroup.style.display = "none";
    }
  }
}

if (UI.settingApiProvider) {
  UI.settingApiProvider.addEventListener("change", () => {
    const selected = UI.settingApiProvider.value;
    const preset = PROVIDERS[selected];
    if (preset) {
      if (selected !== "custom") {
        if (UI.settingApiEndpoint)
          UI.settingApiEndpoint.value = preset.endpoint;
        renderModelDropdown(selected, preset.model);
      } else {
        if (UI.settingApiEndpoint && !UI.settingApiEndpoint.value.trim()) {
          UI.settingApiEndpoint.value = preset.endpoint;
        }
        renderModelDropdown(selected, getActiveModel() || "");
      }
      if (UI.settingApiEndpoint)
        UI.settingApiEndpoint.placeholder = preset.placeholderEndpoint;
      if (UI.settingApiKey)
        UI.settingApiKey.placeholder = preset.keyPlaceholder;
    }
    updateEndpointVisibility();
  });
}

// Nút lấy danh sách models (OpenAI-compatible /models)
if (UI.btnFetchModels) {
  UI.btnFetchModels.addEventListener("click", async () => {
    const provider = UI.settingApiProvider
      ? UI.settingApiProvider.value
      : "gemini";
    const key = (UI.settingApiKey?.value || "").trim();
    let rawEndpoint = (UI.settingApiEndpoint?.value || "").trim();

    if (provider === "custom" && !rawEndpoint) {
      rawEndpoint = PROVIDERS.custom.endpoint;
      if (UI.settingApiEndpoint) UI.settingApiEndpoint.value = rawEndpoint;
    } else if (!rawEndpoint) {
      rawEndpoint = PROVIDERS[provider]?.endpoint || "";
    }

    const showStatus = (text, isError = false) => {
      if (!UI.fetchModelsStatus) return;
      UI.fetchModelsStatus.style.display = "block";
      UI.fetchModelsStatus.style.color = isError ? "#f87171" : "#34d399";
      UI.fetchModelsStatus.textContent = text;
    };

    showStatus("⏳ Đang tải danh sách model...");

    try {
      let modelIds = [];

      if (provider === "gemini") {
        if (!key)
          throw new Error("Cần nhập API Key để lấy danh sách model Gemini.");
        const url = `https://generativelanguage.googleapis.com/v1beta/models?key=${encodeURIComponent(key)}`;
        const res = await fetch(url);
        if (!res.ok) {
          const errJson = await res.json().catch(() => ({}));
          throw new Error(
            `Gemini API (${res.status}): ${errJson.error?.message || res.statusText}`,
          );
        }
        const data = await res.json();
        if (Array.isArray(data.models)) {
          modelIds = data.models
            .filter(
              (m) =>
                !m.supportedGenerationMethods ||
                m.supportedGenerationMethods.includes("generateContent"),
            )
            .map((m) => m.name.replace(/^models\//, ""));
        }
      } else {
        // Chuẩn OpenAI-compatible /models
        const cleanEndpoint = (
          rawEndpoint || PROVIDERS.custom.endpoint
        ).replace(/\/+$/, "");
        let modelsUrl = "";
        if (cleanEndpoint.endsWith("/chat/completions")) {
          modelsUrl = cleanEndpoint.replace(
            /\/chat\/completions$/,
            "/models",
          );
        } else if (cleanEndpoint.endsWith("/models")) {
          modelsUrl = cleanEndpoint;
        } else if (cleanEndpoint.endsWith("/v1")) {
          modelsUrl = `${cleanEndpoint}/models`;
        } else {
          modelsUrl = `${cleanEndpoint}/v1/models`;
        }

        const headers = {};
        if (key) {
          headers["Authorization"] = `Bearer ${key}`;
        }
        if (modelsUrl.includes("openrouter.ai")) {
          headers["HTTP-Referer"] = "https://edux.cmcu.edu.vn";
          headers["X-Title"] = "EDUX Slayers";
        }

        const res = await fetch(modelsUrl, { method: "GET", headers });
        if (!res.ok) {
          const errJson = await res.json().catch(() => ({}));
          throw new Error(
            `API (${res.status}): ${errJson.error?.message || errJson.message || res.statusText}`,
          );
        }

        const resJson = await res.json();
        if (Array.isArray(resJson)) {
          modelIds = resJson
            .map((m) => (typeof m === "string" ? m : m.id || m.name))
            .filter(Boolean);
        } else if (Array.isArray(resJson?.data)) {
          modelIds = resJson.data
            .map((m) => (typeof m === "string" ? m : m.id || m.name))
            .filter(Boolean);
        } else if (Array.isArray(resJson?.models)) {
          modelIds = resJson.models
            .map((m) => (typeof m === "string" ? m : m.name || m.id))
            .filter(Boolean);
        }
      }

      if (modelIds.length === 0) {
        showStatus("⚠️ Server không trả về danh sách model.", true);
        return;
      }

      // Cập nhật datalist cho ô input Model
      if (UI.modelDatalist) {
        UI.modelDatalist.innerHTML = "";
        modelIds.forEach((id) => {
          const opt = document.createElement("option");
          opt.value = id;
          UI.modelDatalist.appendChild(opt);
        });
      }
      // Lưu vào cache theo provider và cập nhật Dropdown
      state.cachedModelsByProvider[provider] = modelIds;
      chrome.storage.local.set({ cachedModelsByProvider: state.cachedModelsByProvider });
      renderModelDropdown(provider, modelIds[0]);

      // Tự động gán model đầu tiên nếu ô nhập trống
      if (!getActiveModel().trim()) {
        setActiveModel(modelIds[0]);
      }

      showStatus(
        `✓ Đã tải ${modelIds.length} models! (Bấm đúp ô nhập để chọn)`,
      );
      addLog(
        UI.testLog,
        `✓ Đã cập nhật danh sách ${modelIds.length} models từ server API.`,
        "success",
      );
      showStatus(`✓ Đã tải ${modelIds.length} models vào menu dropdown!`);
      addLog(
        UI.testLog,
        `✓ Đã cập nhật ${modelIds.length} models từ server API vào menu dropdown.`,
        "success",
      );
    } catch (err) {
      showStatus(`❌ Lỗi: ${err.message}`, true);
      addLog(
        UI.testLog,
        `Không thể lấy danh sách model: ${err.message}`,
        "error",
      );
    }
  });
}

// =========================================================================
// Danh sách cấu hình, form sửa, gán chức năng
// =========================================================================
async function persistAiProfiles() {
  await chrome.storage.local.set({ aiProfiles: state.aiProfiles, aiAssign: state.aiAssign });
  renderAiProfiles();
}

function fillProfileSelect(sel, fn) {
  if (!sel) return;
  sel.innerHTML = "";
  if (state.aiProfiles.length === 0) {
    const opt = document.createElement("option");
    opt.textContent = "— Chưa có cấu hình —";
    sel.appendChild(opt);
    sel.disabled = true;
    return;
  }
  sel.disabled = false;
  state.aiProfiles.forEach((p) => {
    const opt = document.createElement("option");
    opt.value = p.id;
    opt.textContent = profileLabel(p);
    sel.appendChild(opt);
  });
  sel.value = getAssignedProfile(fn).id;
}

function renderAiProfiles() {
  if (UI.aiProfileList) {
    UI.aiProfileList.innerHTML = "";
    if (state.aiProfiles.length === 0) {
      const empty = document.createElement("div");
      empty.className = "ai-profile-empty";
      empty.textContent = "Chưa có cấu hình. Bấm + Thêm để nhập API key.";
      UI.aiProfileList.appendChild(empty);
    }
    state.aiProfiles.forEach((p) => {
      const row = document.createElement("div");
      row.className = "ai-profile-row";
      row.classList.toggle("editing", p.id === state.editingProfileId);

      const name = document.createElement("span");
      name.className = "ai-profile-name";
      name.textContent = profileLabel(p);
      name.title = p.endpoint || "";

      const btnEdit = document.createElement("button");
      btnEdit.type = "button";
      btnEdit.textContent = "✏️";
      btnEdit.title = "Sửa";
      btnEdit.addEventListener("click", () => openProfileEditor(p));

      // Bấm 2 lần để xóa (tránh xóa nhầm key)
      const btnDel = document.createElement("button");
      btnDel.type = "button";
      btnDel.textContent = "🗑";
      btnDel.title = "Xóa";
      btnDel.addEventListener("click", async () => {
        if (!btnDel.dataset.armed) {
          btnDel.dataset.armed = "1";
          btnDel.textContent = "Xóa?";
          btnDel.style.color = "#f87171";
          setTimeout(() => {
            delete btnDel.dataset.armed;
            btnDel.textContent = "🗑";
            btnDel.style.color = "";
          }, 2500);
          return;
        }
        state.aiProfiles = state.aiProfiles.filter((x) => x.id !== p.id);
        AI_FUNCTIONS.forEach((fn) => {
          if (state.aiAssign[fn] === p.id) state.aiAssign[fn] = state.aiProfiles[0]?.id;
        });
        if (state.editingProfileId === p.id) closeProfileEditor();
        await persistAiProfiles();
      });

      row.append(name, btnEdit, btnDel);
      UI.aiProfileList.appendChild(row);
    });
  }

  fillProfileSelect(UI.assignSlide, "slide");
  fillProfileSelect(UI.assignExam, "exam");
  fillProfileSelect(UI.examProfileSelect, "exam");
  fillProfileSelect(UI.slideProfileSelect, "slide");
  changeListeners.forEach((fn) => fn());
}

function renderReasoningSelect(provider, value) {
  const efforts = reasoningEffortsFor(provider);
  if (UI.settingReasoningGroup)
    UI.settingReasoningGroup.style.display = efforts.length ? "flex" : "none";
  if (!UI.settingReasoning || !efforts.length) return;
  UI.settingReasoning.innerHTML = "";
  ["", ...efforts].forEach((v) => {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v ? REASONING_LABELS[v] || v : "Mặc định";
    UI.settingReasoning.appendChild(opt);
  });
  UI.settingReasoning.value = validReasoningEffort(provider, value);
}

if (UI.settingApiProvider) {
  UI.settingApiProvider.addEventListener("change", () =>
    renderReasoningSelect(UI.settingApiProvider.value, ""),
  );
}

function openProfileEditor(p) {
  state.editingProfileId = p ? p.id : null;
  const provider = p ? p.provider : "gemini";
  const preset = PROVIDERS[provider] || PROVIDERS.custom;
  if (UI.settingApiProvider) UI.settingApiProvider.value = provider;
  if (UI.settingApiEndpoint) {
    UI.settingApiEndpoint.value = p ? p.endpoint || "" : preset.endpoint;
    UI.settingApiEndpoint.placeholder = preset.placeholderEndpoint;
  }
  if (UI.settingApiKey) {
    UI.settingApiKey.value = p ? p.apiKey || "" : "";
    UI.settingApiKey.placeholder = preset.keyPlaceholder;
  }
  renderModelDropdown(provider, p ? p.model : preset.model);
  renderReasoningSelect(provider, p ? p.reasoningEffort : "");
  updateEndpointVisibility();
  if (UI.fetchModelsStatus) UI.fetchModelsStatus.style.display = "none";
  if (UI.aiProfileEditor) UI.aiProfileEditor.style.display = "block";
  if (UI.btnAddAiProfile) UI.btnAddAiProfile.style.display = "none";
  renderAiProfiles();
}

function closeProfileEditor() {
  state.editingProfileId = null;
  if (UI.aiProfileEditor) UI.aiProfileEditor.style.display = "none";
  if (UI.btnAddAiProfile) UI.btnAddAiProfile.style.display = "";
  renderAiProfiles();
}

if (UI.btnAddAiProfile)
  UI.btnAddAiProfile.addEventListener("click", () => openProfileEditor(null));
if (UI.btnCancelAiProfile)
  UI.btnCancelAiProfile.addEventListener("click", closeProfileEditor);

if (UI.btnSaveAiProfile) {
  UI.btnSaveAiProfile.addEventListener("click", async () => {
    const provider = UI.settingApiProvider?.value || "gemini";
    let endpoint = (UI.settingApiEndpoint?.value || "").trim();
    if (provider === "custom" && !endpoint) endpoint = PROVIDERS.custom.endpoint;
    const profile = {
      id: state.editingProfileId || `p${Date.now().toString(36)}`,
      provider,
      endpoint,
      apiKey: (UI.settingApiKey?.value || "").trim(),
      model: getActiveModel(),
      reasoningEffort: validReasoningEffort(provider, UI.settingReasoning?.value),
    };
    const idx = state.aiProfiles.findIndex((x) => x.id === profile.id);
    if (idx >= 0) state.aiProfiles[idx] = profile;
    else state.aiProfiles.push(profile);
    // Chức năng chưa gán (hoặc gán cấu hình đã xóa) -> dùng cấu hình vừa lưu
    AI_FUNCTIONS.forEach((fn) => {
      if (!state.aiProfiles.some((x) => x.id === state.aiAssign[fn])) state.aiAssign[fn] = profile.id;
    });
    state.editingProfileId = null;
    await persistAiProfiles();
    closeProfileEditor();
    addLog(UI.testLog, `Đã lưu cấu hình AI: ${profileLabel(profile)}`, "success");
  });
}

[
  [UI.assignSlide, "slide"],
  [UI.assignExam, "exam"],
  [UI.examProfileSelect, "exam"],
  [UI.slideProfileSelect, "slide"],
].forEach(([sel, fn]) => {
  if (!sel) return;
  sel.addEventListener("change", () => {
    state.aiAssign[fn] = sel.value;
    persistAiProfiles();
  });
});

/**
 * Nạp cấu hình đã lưu. Cấu hình đơn cũ (apiProvider/apiKey/...) được chuyển thành
 * cấu hình đầu tiên, dùng cho mọi chức năng.
 */
export async function initAiProfiles(settings) {
  if (settings.cachedModelsByProvider && typeof settings.cachedModelsByProvider === "object") {
    state.cachedModelsByProvider = settings.cachedModelsByProvider;
  }

  if (Array.isArray(settings.aiProfiles)) {
    state.aiProfiles = settings.aiProfiles;
    state.aiAssign = settings.aiAssign || {};
  } else {
    // Chuyển cấu hình đơn cũ (apiProvider/apiKey/...) thành cấu hình đầu tiên cho mọi chức năng
    const legacyProvider = settings.apiProvider || "gemini";
    if (settings.apiKey || legacyProvider !== "gemini") {
      state.aiProfiles = [
        {
          id: "p1",
          provider: legacyProvider,
          endpoint: settings.apiEndpoint || "",
          apiKey: settings.apiKey || "",
          model: settings.apiModel || "",
        },
      ];
      state.aiAssign = { slide: "p1", exam: "p1" };
    }
    await chrome.storage.local.set({ aiProfiles: state.aiProfiles, aiAssign: state.aiAssign });
    await chrome.storage.local.remove(["apiProvider", "apiEndpoint", "apiKey", "apiModel"]);
  }
  renderAiProfiles();
}
