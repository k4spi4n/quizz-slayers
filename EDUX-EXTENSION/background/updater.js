// Kiểm tra bản mới trên GitHub Releases.
// Dùng redirect của /releases/latest -> /releases/tag/vX.Y.Z thay vì GitHub API
// (API giới hạn 60 lượt/giờ/IP, dễ hết khi cả lớp dùng chung mạng trường).
import { compareVersions } from '../shared/version.js';

const UPDATE_REPO = 'k4spi4n/quizz-slayers';
const UPDATE_CHECK_INTERVAL_MS = 6 * 60 * 60 * 1000;

function withUpdateStatus(info) {
  const current = chrome.runtime.getManifest().version;
  return {
    ...info,
    current,
    hasUpdate: !!info.latest && compareVersions(info.latest, current) > 0,
  };
}

export async function checkForUpdate(force = false) {
  const { updateInfo } = await chrome.storage.local.get('updateInfo');
  if (
    !force &&
    updateInfo?.latest &&
    Date.now() - updateInfo.checkedAt < UPDATE_CHECK_INTERVAL_MS
  ) {
    return withUpdateStatus(updateInfo);
  }

  const res = await fetch(`https://github.com/${UPDATE_REPO}/releases/latest`, {
    method: 'HEAD',
    cache: 'no-store',
  });
  const match = res.url.match(/\/releases\/tag\/v?(\d+(?:\.\d+)*)/);
  if (!match) throw new Error('Không đọc được phiên bản mới nhất từ GitHub');

  const info = {
    latest: match[1],
    releaseUrl: res.url,
    downloadUrl: `https://github.com/${UPDATE_REPO}/releases/latest/download/edux-extension.zip`,
    checkedAt: Date.now(),
  };
  await chrome.storage.local.set({ updateInfo: info });

  const status = withUpdateStatus(info);
  chrome.action.setBadgeText({ text: status.hasUpdate ? 'NEW' : '' });
  if (status.hasUpdate) chrome.action.setBadgeBackgroundColor({ color: '#10b981' });
  return status;
}
