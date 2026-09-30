// Client AI duy nhất của extension: Slide (AI) và Bài tập (API) đều gọi qua đây.
import { resolveRequest, validReasoningEffort } from '../shared/providers.js';

// Cấu hình AI được gán cho chức năng (purpose: 'slide' | 'exam'). Popup chuyển cấu hình
// đơn cũ (apiKey/apiModel/...) sang aiProfiles khi mở lần đầu; trước đó vẫn đọc key cũ.
async function getAiProfile(purpose) {
  const s = await chrome.storage.local.get([
    'aiProfiles',
    'aiAssign',
    'apiKey',
    'apiModel',
    'apiEndpoint',
    'apiProvider',
  ]);
  if (!Array.isArray(s.aiProfiles)) {
    return {
      provider: s.apiProvider,
      endpoint: s.apiEndpoint,
      apiKey: s.apiKey,
      model: s.apiModel,
    };
  }
  const id = s.aiAssign?.[purpose];
  return s.aiProfiles.find((p) => p.id === id) || s.aiProfiles[0] || null;
}

async function callGemini(req, { prompt, systemPrompt, temperature }) {
  const payload = {
    contents: [{ parts: [{ text: prompt }] }],
    generationConfig: { temperature },
  };
  if (systemPrompt) {
    payload.systemInstruction = { parts: [{ text: systemPrompt }] };
  }

  const res = await fetch(req.url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errJson = await res.json().catch(() => ({}));
    throw new Error(`Gemini API (${res.status}): ${errJson.error?.message || res.statusText}`);
  }

  const resJson = await res.json();
  return resJson?.candidates?.[0]?.content?.parts?.[0]?.text || '';
}

// Ghép nội dung từ phản hồi dạng stream (SSE "data: {...}") khi server bỏ qua stream: false
function joinSseChunks(rawText) {
  let accumulated = '';
  for (const line of rawText.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) continue;
    const dataStr = trimmed.replace(/^data:\s*/, '').trim();
    if (dataStr === '[DONE]') break;
    try {
      const chunk = JSON.parse(dataStr);
      accumulated +=
        chunk.choices?.[0]?.delta?.content || chunk.choices?.[0]?.message?.content || '';
    } catch (e) {}
  }
  return accumulated;
}

async function callOpenAiCompatible(req, { prompt, systemPrompt, temperature, reasoningEffort }) {
  const headers = { 'Content-Type': 'application/json' };
  if (req.key) {
    headers['Authorization'] = `Bearer ${req.key}`;
  }
  if (req.url.includes('openrouter.ai')) {
    headers['HTTP-Referer'] = 'https://edux.cmcu.edu.vn';
    headers['X-Title'] = 'EDUX Slayers';
  }

  const messages = [];
  if (systemPrompt) {
    messages.push({ role: 'system', content: systemPrompt });
  }
  messages.push({ role: 'user', content: prompt });

  const body = {
    model: req.model,
    messages,
    // Inception (Mercury) chỉ nhận 0.5–1.0; ngoài khoảng sẽ bị đặt về mặc định 1.0
    temperature: req.url.includes('inceptionlabs.ai') ? Math.max(temperature, 0.5) : temperature,
    stream: false,
  };
  const effort = validReasoningEffort(req.provider, reasoningEffort);
  if (effort) body.reasoning_effort = effort;

  const post = () => fetch(req.url, { method: 'POST', headers, body: JSON.stringify(body) });
  let res = await post();
  let rawText = await res.text();

  // Server từ chối reasoning_effort -> thử lại 1 lần không kèm tham số này
  if (
    !res.ok &&
    body.reasoning_effort &&
    (res.status === 400 || res.status === 422) &&
    /reasoning/i.test(rawText)
  ) {
    delete body.reasoning_effort;
    res = await post();
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
    return joinSseChunks(rawText);
  }
  try {
    const choiceMessage = JSON.parse(rawText)?.choices?.[0]?.message;
    return choiceMessage?.content || choiceMessage?.reasoning_content || '';
  } catch (e) {
    return rawText;
  }
}

// Bỏ khối ```json ... ``` bao ngoài kết quả
function stripCodeFence(text) {
  const cleaned = text.trim();
  if (cleaned.startsWith('```')) {
    const match = cleaned.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
    if (match) return match[1].trim();
  }
  return cleaned;
}

export async function callAiService({ prompt, systemPrompt, temperature = 0, purpose = 'slide' }) {
  const profile = await getAiProfile(purpose);
  if (!profile) {
    throw new Error('Chưa có cấu hình AI. Vào tab Cài đặt → Cấu hình AI để thêm.');
  }

  const req = resolveRequest(profile);
  const options = { prompt, systemPrompt, temperature, reasoningEffort: profile.reasoningEffort };
  const responseText =
    req.protocol === 'gemini'
      ? await callGemini(req, options)
      : await callOpenAiCompatible(req, options);

  return stripCodeFence(responseText);
}
