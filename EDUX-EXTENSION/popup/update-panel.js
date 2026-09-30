// Cập nhật & Sao lưu: hiện phiên bản, báo bản mới / nút Áp dụng, xuất & khôi phục cấu hình
import { UI, showTab } from './ui.js';
import { compareVersions } from '../shared/version.js';
import { BACKUP_KEYS } from '../shared/storage.js';

// Cấu hình nằm trong chrome.storage.local: giữ nguyên khi chép đè file + Reload,
// nhưng bị xóa khi Remove extension hoặc cài từ thư mục khác (đổi extension ID).
const runningVersion = chrome.runtime.getManifest().version;
const RELEASES_URL = 'https://github.com/k4spi4n/quizz-slayers/releases/latest';

export function initUpdatePanel() {
  if (UI.appVersion) UI.appVersion.textContent = `v${runningVersion} • Manifest V3`;
  if (UI.currentVersion) UI.currentVersion.textContent = `v${runningVersion}`;

  // Extension dạng unpacked đọc file trực tiếp từ ổ đĩa, nên manifest.json trên đĩa
  // cho biết update.bat đã chép bản mới vào chưa (getManifest() vẫn là bản đang chạy).
  async function getVersionOnDisk() {
    try {
      const res = await fetch(chrome.runtime.getURL('manifest.json'), { cache: 'no-store' });
      return (await res.json()).version || null;
    } catch (e) {
      return null;
    }
  }

  let updateBannerAction = null;
  function showUpdateBanner(title, desc, actionLabel, action) {
    if (!UI.updateBanner) return;
    UI.updateBannerTitle.textContent = title;
    UI.updateBannerDesc.textContent = desc;
    UI.btnUpdateAction.textContent = actionLabel;
    updateBannerAction = action;
    UI.updateBanner.classList.remove('hidden');
  }

  if (UI.btnUpdateAction) {
    UI.btnUpdateAction.addEventListener('click', () => updateBannerAction?.());
  }

  function openUpdateGuide() {
    showTab('tab-settings');
    UI.updateCard?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function setUpdateStatusText(text, color = '') {
    if (!UI.updateStatusText) return;
    UI.updateStatusText.textContent = text;
    UI.updateStatusText.style.color = color;
  }

  async function refreshUpdateStatus(force = false) {
    const diskVersion = await getVersionOnDisk();
    if (diskVersion && compareVersions(diskVersion, runningVersion) > 0) {
      showUpdateBanner(
        `✅ Đã tải bản v${diskVersion}`,
        'Bấm Áp dụng để chạy bản mới — cấu hình AI được giữ nguyên.',
        '🔄 Áp dụng',
        () => chrome.runtime.reload(),
      );
      setUpdateStatusText(`Đã có file v${diskVersion}, chưa áp dụng`, '#34d399');
      return;
    }

    let info;
    try {
      info = await chrome.runtime.sendMessage({ action: 'CHECK_UPDATE', force });
    } catch (err) {
      info = { success: false, message: err.message };
    }
    if (!info?.success) {
      setUpdateStatusText(
        `❌ Không kiểm tra được: ${info?.message || 'không có phản hồi'}`,
        '#f87171',
      );
      return;
    }

    if (info.hasUpdate) {
      showUpdateBanner(
        `🎉 Có bản mới v${info.latest}`,
        'Chạy update.bat trong thư mục extension — giữ nguyên cấu hình.',
        'Cách cập nhật',
        openUpdateGuide,
      );
      setUpdateStatusText(`🎉 Có bản mới v${info.latest}`, '#34d399');
    } else {
      UI.updateBanner?.classList.add('hidden');
      const checkedAt = new Date(info.checkedAt).toLocaleTimeString('vi-VN', {
        hour: '2-digit',
        minute: '2-digit',
      });
      setUpdateStatusText(`✓ Đang dùng bản mới nhất (kiểm tra lúc ${checkedAt})`);
    }
  }

  refreshUpdateStatus();

  if (UI.btnCheckUpdate) {
    UI.btnCheckUpdate.addEventListener('click', async () => {
      UI.btnCheckUpdate.disabled = true;
      setUpdateStatusText('⏳ Đang kiểm tra...');
      await refreshUpdateStatus(true);
      UI.btnCheckUpdate.disabled = false;
    });
  }
  if (UI.btnDownloadUpdate) {
    UI.btnDownloadUpdate.addEventListener('click', () =>
      chrome.tabs.create({ url: `${RELEASES_URL}/download/edux-extension.zip` }),
    );
  }
  if (UI.btnReleaseNotes) {
    UI.btnReleaseNotes.addEventListener('click', () => chrome.tabs.create({ url: RELEASES_URL }));
  }

  function showBackupStatus(text, isError = false) {
    if (!UI.backupStatus) return;
    UI.backupStatus.style.display = 'block';
    UI.backupStatus.style.color = isError ? '#f87171' : '#34d399';
    UI.backupStatus.textContent = text;
  }

  if (UI.btnExportSettings) {
    UI.btnExportSettings.addEventListener('click', async () => {
      const settings = await chrome.storage.local.get(BACKUP_KEYS);
      const backup = {
        app: 'edux-slayers',
        backupVersion: 1,
        extensionVersion: runningVersion,
        exportedAt: new Date().toISOString(),
        settings,
      };
      const blob = new Blob([JSON.stringify(backup, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `edux-slayers-backup-${new Date().toISOString().slice(0, 10)}.json`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      showBackupStatus(`✓ Đã xuất ${(settings.aiProfiles || []).length} cấu hình AI cùng cài đặt.`);
    });
  }

  if (UI.btnImportSettings && UI.importSettingsFile) {
    UI.btnImportSettings.addEventListener('click', () => UI.importSettingsFile.click());
    UI.importSettingsFile.addEventListener('change', async () => {
      const file = UI.importSettingsFile.files?.[0];
      UI.importSettingsFile.value = '';
      if (!file) return;
      try {
        let data;
        try {
          data = JSON.parse(await file.text());
        } catch (e) {
          throw new Error('File không phải JSON hợp lệ.');
        }
        if (data?.app !== 'edux-slayers' || !data.settings || typeof data.settings !== 'object') {
          throw new Error('Không phải file sao lưu của EDUX Slayers.');
        }

        const restored = {};
        BACKUP_KEYS.forEach((k) => {
          if (data.settings[k] !== undefined) restored[k] = data.settings[k];
        });
        if (restored.aiProfiles !== undefined) {
          if (!Array.isArray(restored.aiProfiles))
            throw new Error('Danh sách cấu hình AI trong file bị hỏng.');
          restored.aiProfiles = restored.aiProfiles.filter(
            (p) => p && typeof p === 'object' && typeof p.id === 'string',
          );
        }
        if (Object.keys(restored).length === 0) throw new Error('File sao lưu trống.');

        await chrome.storage.local.set(restored);
        showBackupStatus(
          `✓ Đã khôi phục ${(restored.aiProfiles || []).length} cấu hình AI — đang tải lại...`,
        );
        setTimeout(() => location.reload(), 800);
      } catch (err) {
        showBackupStatus(`❌ ${err.message}`, true);
      }
    });
  }
}
