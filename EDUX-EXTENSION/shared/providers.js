// Bảng provider AI dùng chung cho popup (form cấu hình) và background (gọi API).
// Thêm provider mới: thêm 1 mục vào PROVIDERS; nếu dùng API chuẩn OpenAI thì không cần sửa chỗ nào khác.

export const PROVIDERS = {
  gemini: {
    name: 'Gemini',
    endpoint: '',
    model: 'gemini-2.0-flash',
    placeholderEndpoint: 'Mặc định: https://generativelanguage.googleapis.com',
    keyPlaceholder: 'Nhập Google Gemini API Key (AIza...)',
    models: [
      { id: 'gemini-2.0-flash', label: 'gemini-2.0-flash (Khuyên dùng - Nhanh & Chuẩn)' },
      { id: 'gemini-2.0-pro-exp-02-05', label: 'gemini-2.0-pro-exp-02-05 (Suy luận sâu)' },
      { id: 'gemini-1.5-flash', label: 'gemini-1.5-flash' },
      { id: 'gemini-1.5-pro', label: 'gemini-1.5-pro' },
    ],
  },
  openai: {
    name: 'OpenAI',
    endpoint: 'https://api.openai.com/v1',
    model: 'gpt-4o-mini',
    placeholderEndpoint: 'https://api.openai.com/v1',
    keyPlaceholder: 'Nhập OpenAI API Key (sk-...)',
    models: [
      { id: 'gpt-4o-mini', label: 'gpt-4o-mini (Khuyên dùng - Nhanh & Rẻ)' },
      { id: 'gpt-4o', label: 'gpt-4o (Toàn diện nhất)' },
      { id: 'o3-mini', label: 'o3-mini (Lý luận cao cấp)' },
      { id: 'gpt-4-turbo', label: 'gpt-4-turbo' },
    ],
  },
  deepseek: {
    name: 'DeepSeek',
    endpoint: 'https://api.deepseek.com/v1',
    model: 'deepseek-chat',
    placeholderEndpoint: 'https://api.deepseek.com/v1',
    keyPlaceholder: 'Nhập DeepSeek API Key (sk-...)',
    models: [
      { id: 'deepseek-chat', label: 'deepseek-chat (DeepSeek-V3)' },
      { id: 'deepseek-reasoner', label: 'deepseek-reasoner (DeepSeek-R1)' },
    ],
  },
  openrouter: {
    name: 'OpenRouter',
    endpoint: 'https://openrouter.ai/api/v1',
    model: 'google/gemini-2.0-flash-001',
    // Model gửi đi khi cấu hình để trống model (khác model gợi ý sẵn trong form)
    fallbackModel: 'gpt-4o-mini',
    placeholderEndpoint: 'https://openrouter.ai/api/v1',
    keyPlaceholder: 'Nhập OpenRouter API Key (sk-or-v1-...)',
    models: [
      { id: 'google/gemini-2.0-flash-001', label: 'google/gemini-2.0-flash-001' },
      { id: 'deepseek/deepseek-r1', label: 'deepseek/deepseek-r1' },
      { id: 'meta-llama/llama-3.3-70b-instruct', label: 'meta-llama/llama-3.3-70b-instruct' },
      { id: 'anthropic/claude-3.5-sonnet', label: 'anthropic/claude-3.5-sonnet' },
    ],
  },
  inception: {
    name: 'Inception',
    endpoint: 'https://api.inceptionlabs.ai/v1',
    model: 'mercury-2.5',
    placeholderEndpoint: 'https://api.inceptionlabs.ai/v1',
    keyPlaceholder: 'Nhập Inception API Key',
    // Mức suy luận (tham số reasoning_effort) — chỉ provider có danh sách này mới hiện lựa chọn
    reasoningEfforts: ['instant', 'low', 'medium', 'high'],
    models: [
      { id: 'mercury-2.5', label: 'mercury-2.5 (Khuyên dùng)' },
      { id: 'mercury-2', label: 'mercury-2' },
    ],
  },
  ollama: {
    name: 'Ollama',
    endpoint: 'http://localhost:11434/v1',
    model: 'llama3.2',
    placeholderEndpoint: 'http://localhost:11434/v1',
    keyPlaceholder: 'Không cần API Key đối với Ollama (để trống)',
    models: [
      { id: 'llama3.2', label: 'llama3.2' },
      { id: 'qwen2.5:7b', label: 'qwen2.5:7b' },
      { id: 'deepseek-r1:7b', label: 'deepseek-r1:7b' },
      { id: 'mistral', label: 'mistral' },
    ],
  },
  custom: {
    name: 'Custom',
    endpoint: 'http://localhost:20128/v1',
    model: '',
    fallbackModel: 'gpt-4o-mini',
    placeholderEndpoint: 'http://localhost:20128/v1',
    keyPlaceholder: 'Nhập API Key nếu có (hoặc để trống)...',
    models: [],
  },
};

export const REASONING_LABELS = { instant: 'instant (nhanh nhất)', high: 'high (kỹ nhất)' };

const GEMINI_DEFAULT_MODEL = 'gemini-2.0-flash';
const OPENAI_COMPAT_DEFAULT_MODEL = 'gpt-4o-mini';

export function reasoningEffortsFor(provider) {
  return PROVIDERS[provider]?.reasoningEfforts || [];
}

export function validReasoningEffort(provider, value) {
  return reasoningEffortsFor(provider).includes(value) ? value : '';
}

// Model thực sự được dùng khi cấu hình để trống model
export function displayModel(model, provider) {
  if (model) return model;
  const p = PROVIDERS[provider];
  if (!p || provider === 'gemini') return GEMINI_DEFAULT_MODEL;
  return p.fallbackModel || p.model || OPENAI_COMPAT_DEFAULT_MODEL;
}

const isLocalUrl = (url) => /localhost|127\.0\.0\.1/.test(url || '');

// Cấu hình dùng được ngay: có key, hoặc chạy local (không cần key)
export function isProfileReady(p) {
  return !!p && (!!p.apiKey || p.provider === 'ollama' || isLocalUrl(p.endpoint));
}

/**
 * Tính request cho 1 cấu hình AI: giao thức (Gemini generateContent hay OpenAI chat/completions),
 * URL đầy đủ, model và key. Ném lỗi nếu thiếu key với provider không chạy local.
 */
export function resolveRequest(profile) {
  const key = (profile.apiKey || '').trim();
  const rawModel = (profile.model || '').trim();
  const provider = (profile.provider || 'gemini').toLowerCase().trim();
  let customEndpoint = (profile.endpoint || '').trim();
  if (provider === 'custom' && !customEndpoint) customEndpoint = PROVIDERS.custom.endpoint;

  if (!key && provider !== 'ollama' && !isLocalUrl(customEndpoint)) {
    throw new Error('Chưa cấu hình API Key. Vui lòng vào tab Cài đặt để nhập key.');
  }

  let isGemini;
  if (provider === 'gemini') isGemini = true;
  else if (PROVIDERS[provider] && provider !== 'custom') isGemini = false;
  else if (customEndpoint)
    isGemini =
      customEndpoint.includes('googleapis.com') || customEndpoint.includes(':generateContent');
  else isGemini = key.startsWith('AIza') || rawModel.toLowerCase().includes('gemini') || !rawModel;

  if (isGemini) {
    const model = rawModel ? rawModel.replace(/^gemini\//i, '') : GEMINI_DEFAULT_MODEL;
    const keyParam = `key=${encodeURIComponent(key)}`;
    let url;
    if (!customEndpoint) {
      url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?${keyParam}`;
    } else if (customEndpoint.includes(':generateContent')) {
      url = customEndpoint;
      if (key && !url.includes('key=')) url += (url.includes('?') ? '&' : '?') + keyParam;
    } else {
      const base = customEndpoint.replace(/\/+$/, '');
      const version = base.endsWith('/v1beta') || base.endsWith('/v1') ? '' : '/v1beta';
      url = `${base}${version}/models/${model}:generateContent?${keyParam}`;
    }
    return { protocol: 'gemini', url, model, key, provider };
  }

  const model =
    rawModel || (PROVIDERS[provider] ? displayModel('', provider) : OPENAI_COMPAT_DEFAULT_MODEL);
  let url;
  if (customEndpoint) {
    const base = customEndpoint.replace(/\/+$/, '');
    if (base.endsWith('/chat/completions')) url = base;
    else if (base.endsWith('/v1')) url = `${base}/chat/completions`;
    else url = `${base}/v1/chat/completions`;
  } else {
    url = `${(PROVIDERS[provider] || PROVIDERS.openai).endpoint}/chat/completions`;
  }
  return { protocol: 'openai', url, model, key, provider };
}
