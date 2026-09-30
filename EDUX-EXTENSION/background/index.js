// EDUX Slayers Background Service Worker (Manifest V3, ES module)
// Lắng nghe sự kiện trình duyệt và định tuyến tin nhắn từ popup / content scripts tới từng module.
import { DEFAULT_SETTINGS } from '../shared/storage.js';
import { callAiService } from './ai-client.js';
import {
  slideSystemPrompt,
  slideUserPrompt,
  EXAM_SYSTEM_PROMPT,
  parseSlideIndex,
} from './prompts.js';
import { layaHealth, layaSolveSlide } from './laya.js';
import { checkForUpdate } from './updater.js';
import { INJECTED_SCRIPT } from '../shared/content-scripts.js';

chrome.runtime.onInstalled.addListener(() => {
  console.log('⚔️ EDUX Slayers Extension installed successfully.');

  // Ghi giá trị mặc định cho các cài đặt chưa có
  chrome.storage.local.get(Object.keys(DEFAULT_SETTINGS), (existing) => {
    const defaults = {};
    for (const [key, value] of Object.entries(DEFAULT_SETTINGS)) {
      if (existing[key] === undefined) defaults[key] = value;
    }
    if (Object.keys(defaults).length > 0) {
      chrome.storage.local.set(defaults);
    }
  });

  checkForUpdate(true).catch(() => {});
});

chrome.runtime.onStartup.addListener(() => {
  checkForUpdate().catch(() => {});
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
        files: [INJECTED_SCRIPT],
        world: 'MAIN',
      })
      .catch(() => {});
  }
});

const isValidSlide = (req) =>
  !!req.question && Array.isArray(req.choices) && req.choices.length > 0;
const INVALID_SLIDE = { success: false, message: 'Dữ liệu câu hỏi hoặc lựa chọn không hợp lệ' };

// action -> handler(req) trả về object phản hồi. Lỗi ném ra được đổi thành { success: false, message }.
const handlers = {
  async CHECK_UPDATE(req) {
    return { success: true, ...(await checkForUpdate(!!req.force)) };
  },

  LAYA_HEALTH: () => layaHealth(),

  async LAYA_SOLVE_SLIDE(req) {
    if (!isValidSlide(req)) return INVALID_SLIDE;
    return layaSolveSlide(req.question, req.choices);
  },

  async AI_SOLVE_SLIDE(req) {
    if (!isValidSlide(req)) return INVALID_SLIDE;
    const { question, choices } = req;

    const rawResult = await callAiService({
      prompt: slideUserPrompt(question, choices),
      systemPrompt: slideSystemPrompt(choices.length),
      temperature: 0,
      purpose: 'slide',
    });

    const index = parseSlideIndex(rawResult, choices);
    if (index < 0) {
      return {
        success: false,
        message: 'Không phân tích được chỉ số đáp án từ kết quả AI: ' + rawResult.substring(0, 100),
      };
    }
    return { success: true, index, answerText: choices[index], rawResult };
  },

  async AI_SOLVE_EXAM(req) {
    const answersText = await callAiService({
      prompt: req.promptText,
      systemPrompt: EXAM_SYSTEM_PROMPT,
      temperature: 0,
      purpose: 'exam',
    });
    return { success: true, answersText };
  },
};

chrome.runtime.onMessage.addListener((req, sender, sendResponse) => {
  if (!req || !Object.hasOwn(handlers, req.action)) return false;
  const handler = handlers[req.action];

  Promise.resolve()
    .then(() => handler(req))
    .then(sendResponse, (err) => sendResponse({ success: false, message: err.message }));
  return true; // Giữ kết nối async cho sendResponse
});
