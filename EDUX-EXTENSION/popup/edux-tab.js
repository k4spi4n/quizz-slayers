// Giao tiếp với tab EDUX: tìm tab, gửi lệnh tới content script (tự nạp lại script nếu chưa có)
import { CONTENT_SCRIPTS, INJECTED_SCRIPT } from '../shared/content-scripts.js';

const CONTENT_CSS = 'content/content.css';

export const isEduxUrl = (url) => !!url && (url.includes('cmcu.edu.vn') || url.includes('edux'));

export async function getActiveTab() {
  const [tab] = await chrome.tabs.query({
    active: true,
    currentWindow: true,
  });
  if (tab && isEduxUrl(tab.url)) return tab;
  const allTabs = await chrome.tabs.query({});
  return allTabs.find((t) => isEduxUrl(t.url)) || tab;
}

export async function sendTabMessage(tabId, message) {
  try {
    return await chrome.tabs.sendMessage(tabId, message);
  } catch (err) {
    // Content script not loaded or tab disconnected: inject required scripts
    try {
      await chrome.scripting
        .executeScript({
          target: { tabId },
          files: [INJECTED_SCRIPT],
          world: 'MAIN',
        })
        .catch(() => {});

      await chrome.scripting.executeScript({
        target: { tabId },
        files: CONTENT_SCRIPTS,
      });
      await chrome.scripting.insertCSS({
        target: { tabId },
        files: [CONTENT_CSS],
      });
      await new Promise((r) => setTimeout(r, 200));
      return await chrome.tabs.sendMessage(tabId, message);
    } catch (injectErr) {
      throw err;
    }
  }
}

// Đảm bảo network interceptor luôn hoạt động trong MAIN world
export function ensureInterceptor(tabId) {
  chrome.scripting
    .executeScript({ target: { tabId }, files: [INJECTED_SCRIPT], world: 'MAIN' })
    .catch(() => {});
}
