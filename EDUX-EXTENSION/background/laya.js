// Laya (https://github.com/NandhaKishorM/laya) — encoder chấm điểm đáp án 1 lượt,
// chạy local qua `laya-serve` (POST /v1/systemone). Không sinh văn bản nên nhanh
// hơn LLM, và trả về xác suất cho TỪNG đáp án để thử sai theo thứ tự tốt nhất.

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

export async function layaHealth() {
  const { json } = await layaFetch('/health', { method: 'GET' });
  return { success: true, loaded: json?.loaded || [], device: json?.device || '' };
}

export async function layaSolveSlide(question, choices) {
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
    return { success: false, message: 'Laya không trả về xác suất cho các đáp án' };
  }

  return {
    success: true,
    index: ranking[0].index,
    ranking: ranking.map((r) => r.index),
    probabilities: ranking.map((r) => r.p),
    elapsedMs: Number(res.headers.get('X-Inference-Time-Ms')) || Date.now() - started,
    model: json?.routing?.model || ''
  };
}
