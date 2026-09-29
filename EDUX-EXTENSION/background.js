// EDUX Slayers Background Service Worker (Manifest V3)

chrome.runtime.onInstalled.addListener(() => {
  console.log('⚔️ EDUX Slayers Extension installed successfully.');

  // Set default configuration
  chrome.storage.local.get(['delayMs', 'autoNext', 'slideStats'], (existing) => {
    const defaults = {};
    if (existing.delayMs === undefined) defaults.delayMs = 100;
    if (existing.autoNext === undefined) defaults.autoNext = true;
    if (!existing.slideStats) defaults.slideStats = { solved: 0, retries: 0 };

    if (Object.keys(defaults).length > 0) {
      chrome.storage.local.set(defaults);
    }
  });
});

// Tự động đảm bảo bộ lắng nghe mạng (injected.js) hoạt động trong MAIN world khi trang tải
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (
    changeInfo.status === 'loading' &&
    tab.url &&
    (tab.url.includes('edux.cmcu.edu.vn') || tab.url.includes('cmcu.edu.vn'))
  ) {
    chrome.scripting
      .executeScript({
        target: { tabId },
        files: ['injected.js'],
        world: 'MAIN'
      })
      .catch(() => {});
  }
});

// =========================================================================
// AI Service Worker for All Extension Features (High Accuracy)
// =========================================================================
// Mức suy luận (reasoning_effort) hợp lệ theo provider. Giữ đồng bộ với popup.js
const REASONING_EFFORTS = {
  inception: ['instant', 'low', 'medium', 'high']
};

// Cấu hình AI được gán cho chức năng (purpose: 'slide' | 'exam'). Popup chuyển cấu hình
// đơn cũ (apiKey/apiModel/...) sang aiProfiles khi mở lần đầu; trước đó vẫn đọc key cũ.
async function getAiProfile(purpose) {
  const s = await chrome.storage.local.get([
    'aiProfiles',
    'aiAssign',
    'apiKey',
    'apiModel',
    'apiEndpoint',
    'apiProvider'
  ]);
  if (!Array.isArray(s.aiProfiles)) {
    return { provider: s.apiProvider, endpoint: s.apiEndpoint, apiKey: s.apiKey, model: s.apiModel };
  }
  const id = s.aiAssign?.[purpose];
  return s.aiProfiles.find((p) => p.id === id) || s.aiProfiles[0] || null;
}

async function callAiService({ prompt, systemPrompt, temperature = 0, purpose = 'slide' }) {
  const profile = await getAiProfile(purpose);
  if (!profile) {
    throw new Error('Chưa có cấu hình AI. Vào tab Cài đặt → Cấu hình AI để thêm.');
  }

  const key = (profile.apiKey || '').trim();
  const rawModel = (profile.model || '').trim();
  let customEndpoint = (profile.endpoint || '').trim();
  const provider = (profile.provider || 'gemini').toLowerCase().trim();

  if (provider === 'custom' && !customEndpoint) {
    customEndpoint = 'http://localhost:20128/v1';
  }

  const isLocal = provider === 'ollama' || customEndpoint.includes('localhost') || customEndpoint.includes('127.0.0.1');

  if (!key && !isLocal) {
    throw new Error('Chưa cấu hình API Key. Vui lòng vào tab Cài đặt để nhập key.');
  }

  let isGemini = false;
  if (provider === 'gemini') {
    isGemini = true;
  } else if (provider === 'openai' || provider === 'deepseek' || provider === 'openrouter' || provider === 'ollama' || provider === 'inception') {
    isGemini = false;
  } else {
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

    const payload = {
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: { temperature }
    };
    if (systemPrompt) {
      payload.systemInstruction = { parts: [{ text: systemPrompt }] };
    }

    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}));
      throw new Error(`Gemini API (${res.status}): ${errJson.error?.message || res.statusText}`);
    }

    const resJson = await res.json();
    responseText = resJson?.candidates?.[0]?.content?.parts?.[0]?.text || '';
  } else {
    // OpenAI-compatible
    let openAiModel = rawModel;
    if (!openAiModel) {
      if (provider === 'deepseek') openAiModel = 'deepseek-chat';
      else if (provider === 'ollama') openAiModel = 'llama3.2';
      else if (provider === 'inception') openAiModel = 'mercury-2.5';
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
      if (provider === 'deepseek') endpoint = 'https://api.deepseek.com/v1/chat/completions';
      else if (provider === 'openrouter') endpoint = 'https://openrouter.ai/api/v1/chat/completions';
      else if (provider === 'ollama') endpoint = 'http://localhost:11434/v1/chat/completions';
      else if (provider === 'inception') endpoint = 'https://api.inceptionlabs.ai/v1/chat/completions';
      else endpoint = 'https://api.openai.com/v1/chat/completions';
    }

    const headers = { 'Content-Type': 'application/json' };
    if (key) {
      headers['Authorization'] = `Bearer ${key}`;
    }
    if (endpoint.includes('openrouter.ai')) {
      headers['HTTP-Referer'] = 'https://edux.cmcu.edu.vn';
      headers['X-Title'] = 'EDUX Slayers';
    }

    const messages = [];
    if (systemPrompt) {
      messages.push({ role: 'system', content: systemPrompt });
    }
    messages.push({ role: 'user', content: prompt });

    const body = {
      model: openAiModel,
      messages,
      // Inception (Mercury) chỉ nhận 0.5–1.0; ngoài khoảng sẽ bị đặt về mặc định 1.0
      temperature: endpoint.includes('inceptionlabs.ai') ? Math.max(temperature, 0.5) : temperature,
      stream: false
    };
    if ((REASONING_EFFORTS[provider] || []).includes(profile.reasoningEffort)) {
      body.reasoning_effort = profile.reasoningEffort;
    }

    let res = await fetch(endpoint, { method: 'POST', headers, body: JSON.stringify(body) });
    let rawText = await res.text();

    // Server từ chối reasoning_effort -> thử lại 1 lần không kèm tham số này
    if (!res.ok && body.reasoning_effort && (res.status === 400 || res.status === 422) && /reasoning/i.test(rawText)) {
      delete body.reasoning_effort;
      res = await fetch(endpoint, { method: 'POST', headers, body: JSON.stringify(body) });
      rawText = await res.text();
    }
    if (!res.ok) {
      let errMessage = res.statusText;
      try {
        const errJson = JSON.parse(rawText);
        errMessage = errJson.error?.message || errJson.message || errMessage;
      } catch (e) {}
      throw new Error(`API (${res.status}): ${errMessage}`);
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
// Laya (https://github.com/NandhaKishorM/laya) — encoder chấm điểm đáp án 1 lượt,
// chạy local qua `laya-serve` (POST /v1/systemone). Không sinh văn bản nên nhanh
// hơn LLM, và trả về xác suất cho TỪNG đáp án để thử sai theo thứ tự tốt nhất.
// =========================================================================
const LAYA_DEFAULT_ENDPOINT = 'http://localhost:8000';

async function getLayaSettings() {
  const s = await chrome.storage.local.get(['layaEndpoint', 'layaApiKey']);
  const base = ((s.layaEndpoint || '').trim() || LAYA_DEFAULT_ENDPOINT).replace(/\/+$/, '');
  const headers = { 'Content-Type': 'application/json' };
  const key = (s.layaApiKey || '').trim();
  if (key) headers['Authorization'] = `Bearer ${key}`;
  return { base, headers };
}

async function layaFetch(path, init) {
  const { base, headers } = await getLayaSettings();
  let res;
  try {
    res = await fetch(`${base}${path}`, { ...init, headers });
  } catch (e) {
    throw new Error(`Không kết nối được Laya tại ${base}. Hãy chạy "laya-serve" trước.`);
  }
  const rawText = await res.text();
  let json = null;
  try {
    json = JSON.parse(rawText);
  } catch (e) {}
  if (!res.ok) {
    throw new Error(`Laya (${res.status}): ${json?.detail || res.statusText}`);
  }
  return { json, res };
}

// Dùng chính nội dung đáp án làm nhãn (đo trên bộ câu hỏi tiếng Việt: top-1 50%
// so với 38% khi dùng nhãn "A: ..."). Laya từ chối nhãn trùng -> thêm hậu tố.
function layaLabels(choices) {
  const seen = new Set();
  return choices.map((c, i) => {
    let label = (c || '').trim() || `Lựa chọn ${i + 1}`;
    if (seen.has(label)) label = `${label} (${i + 1})`;
    seen.add(label);
    return label;
  });
}

// Lắng nghe các yêu cầu giải AI từ Slide Solver & Test Solver
chrome.runtime.onMessage.addListener((req, sender, sendResponse) => {
  if (req.action === 'LAYA_HEALTH') {
    (async () => {
      try {
        const { json } = await layaFetch('/health', { method: 'GET' });
        sendResponse({ success: true, loaded: json?.loaded || [], device: json?.device || '' });
      } catch (err) {
        sendResponse({ success: false, message: err.message });
      }
    })();
    return true;
  }

  if (req.action === 'LAYA_SOLVE_SLIDE') {
    (async () => {
      try {
        const { question, choices } = req;
        if (!question || !Array.isArray(choices) || choices.length === 0) {
          sendResponse({ success: false, message: 'Dữ liệu câu hỏi hoặc lựa chọn không hợp lệ' });
          return;
        }

        const labels = layaLabels(choices);

        const started = Date.now();
        const { json, res } = await layaFetch('/v1/systemone', {
          method: 'POST',
          body: JSON.stringify({
            state: question,
            // EDUX toàn tiếng Việt -> chỉ định checkpoint đa ngôn ngữ, tránh router
            // gửi câu hỏi có nhiều thuật ngữ tiếng Anh sang checkpoint English-only.
            model: 'multilingual',
            questions: {
              answer: {
                type: 'choice',
                instructions: 'Which option correctly answers the question?',
                criteria: labels
              }
            }
          })
        });

        const probs = json?.answers?.answer?.probabilities || {};
        const ranking = labels
          .map((label, i) => ({ index: i, p: Number(probs[label]) || 0 }))
          .sort((a, b) => b.p - a.p);

        if (ranking.length === 0 || !(labels[ranking[0].index] in probs)) {
          sendResponse({ success: false, message: 'Laya không trả về xác suất cho các đáp án' });
          return;
        }

        sendResponse({
          success: true,
          index: ranking[0].index,
          ranking: ranking.map((r) => r.index),
          probabilities: ranking.map((r) => r.p),
          elapsedMs: Number(res.headers.get('X-Inference-Time-Ms')) || Date.now() - started,
          model: json?.routing?.model || ''
        });
      } catch (err) {
        sendResponse({ success: false, message: err.message });
      }
    })();
    return true;
  }

  if (req.action === 'AI_SOLVE_SLIDE') {
    (async () => {
      try {
        const { question, choices } = req;
        if (!question || !Array.isArray(choices) || choices.length === 0) {
          sendResponse({ success: false, message: 'Dữ liệu câu hỏi hoặc lựa chọn không hợp lệ' });
          return;
        }

        const systemPrompt =
          'Bạn là chuyên gia giáo dục và bài giảng học tập với độ chính xác tuyệt đối.\n' +
          'Nhiệm vụ: Phân tích kỹ lưỡng câu hỏi và chọn duy nhất 1 đáp án chính xác nhất trong các lựa chọn được cung cấp.\n' +
          'Định dạng đầu ra BẮT BUỘC là JSON duy nhất: {"index": X} trong đó X là số thứ tự (từ 0 đến ' +
          (choices.length - 1) +
          ') của lựa chọn đúng nhất.\n' +
          'Tuyệt đối không giải thích, không viết thêm bất kỳ chữ nào ngoài chuỗi JSON.';

        const prompt =
          `Câu hỏi bài giảng:\n${question}\n\n` +
          `Các lựa chọn:\n` +
          choices.map((c, i) => `${i}. ${c}`).join('\n') +
          `\n\nHãy chọn đáp án đúng nhất (trả về JSON dạng {"index": X}):`;

        const rawResult = await callAiService({ prompt, systemPrompt, temperature: 0, purpose: 'slide' });

        let parsedIndex = -1;
        try {
          const matchJson = rawResult.match(/\{[\s\S]*?\}/);
          if (matchJson) {
            const parsed = JSON.parse(matchJson[0]);
            if (typeof parsed.index === 'number') {
              parsedIndex = parsed.index;
            } else if (typeof parsed.index === 'string' && /^\d+$/.test(parsed.index)) {
              parsedIndex = parseInt(parsed.index, 10);
            }
          }
        } catch (e) {}

        if (parsedIndex < 0 || parsedIndex >= choices.length) {
          const digitMatch = rawResult.match(/\b([0-9])\b/);
          if (digitMatch) {
            const d = parseInt(digitMatch[1], 10);
            if (d >= 0 && d < choices.length) parsedIndex = d;
          }
        }

        if (parsedIndex < 0 || parsedIndex >= choices.length) {
          const lowerRes = rawResult.toLowerCase();
          for (let i = 0; i < choices.length; i++) {
            const lowerC = choices[i].toLowerCase();
            if (lowerC.length > 2 && (lowerRes.includes(lowerC) || lowerC.includes(lowerRes))) {
              parsedIndex = i;
              break;
            }
          }
        }

        if (parsedIndex >= 0 && parsedIndex < choices.length) {
          sendResponse({
            success: true,
            index: parsedIndex,
            answerText: choices[parsedIndex],
            rawResult
          });
        } else {
          sendResponse({
            success: false,
            message: 'Không phân tích được chỉ số đáp án từ kết quả AI: ' + rawResult.substring(0, 100)
          });
        }
      } catch (err) {
        sendResponse({ success: false, message: err.message });
      }
    })();
    return true; // Giữ kết nối async cho sendResponse
  }

  if (req.action === 'AI_SOLVE_EXAM') {
    (async () => {
      try {
        const { promptText } = req;
        const systemPrompt =
          'Bạn là chuyên gia khảo thí và học thuật cao cấp hàng đầu, có độ chính xác tuyệt đối 100% trong việc giải quyết các bài kiểm tra trắc nghiệm, đúng/sai, điền khuyết và tự luận.\n' +
          'Yêu cầu:\n' +
          '1. Phân tích cẩn thận từng câu hỏi và các lựa chọn loại trừ để chọn phương án đúng tuyệt đối.\n' +
          '2. Trả về JSONL một dòng duy nhất (hoặc JSON Array các object);\n' +
          '3. Mỗi phần tử có "so_cau" và "dap_an";\n' +
          '4. "dap_an" là A/B/C/D hoặc từ/cụm từ cần điền;\n' +
          '5. Với câu đúng/sai, "dap_an" là mảng giá trị Đúng/Sai theo thứ tự từng mệnh đề (ví dụ: [true, false, true, true]);\n' +
          '6. Tuyệt đối không thêm lời dẫn hay giải thích.';

        const answersText = await callAiService({
          prompt: promptText,
          systemPrompt,
          temperature: 0,
          purpose: 'exam'
        });

        sendResponse({ success: true, answersText });
      } catch (err) {
        sendResponse({ success: false, message: err.message });
      }
    })();
    return true;
  }
});

