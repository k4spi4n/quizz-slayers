// Tab Điểm số: tiến độ các môn (trang /student) hoặc điểm từng bài tập (trang môn học)
import { UI } from './ui.js';
import { getActiveTab, sendTabMessage } from './edux-tab.js';

export async function loadExerciseScores() {
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
      const res = await sendTabMessage(tab.id, {
        action: 'GET_ALL_SUBJECTS_PROGRESS',
      });
      if (!res || !res.success || !Array.isArray(res.subjects)) {
        if (UI.scoresSubjectTitle)
          UI.scoresSubjectTitle.textContent = 'Không thể lấy dữ liệu học phần.';
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
    const res = await sendTabMessage(tab.id, {
      action: 'GET_EXERCISE_SCORES',
    });
    if (!res || !res.success || !Array.isArray(res.models)) {
      if (UI.scoresSubjectTitle)
        UI.scoresSubjectTitle.textContent = res?.message || 'Không tìm thấy dữ liệu bài tập.';
      return;
    }

    const models = res.models;
    const examModels = models.filter((m) => m.exist_exam);
    const totalExams = examModels.length;
    const completedExams = examModels.filter(
      (m) => m.highest_score !== null && m.highest_score !== undefined,
    );
    const pendingExams = examModels.filter(
      (m) => m.highest_score === null || m.highest_score === undefined,
    );

    const scores = completedExams.map((m) => parseFloat(m.highest_score)).filter((s) => !isNaN(s));
    const maxScore = scores.length
      ? Math.max(...scores)
          .toFixed(2)
          .replace(/\.00$/, '')
      : '--';

    if (UI.scoresSubjectTitle) {
      UI.scoresSubjectTitle.textContent = `Môn học: ${res.subjectId ? res.subjectId.slice(0, 8) + '...' : 'Hiện tại'}`;
    }
    if (UI.scoresCompleted)
      UI.scoresCompleted.textContent = `${completedExams.length}/${totalExams}`;
    if (UI.scoresHighest)
      UI.scoresHighest.textContent = maxScore !== '--' ? `${maxScore}/10` : '--';

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
          scoreVal !== null && !isNaN(scoreVal)
            ? scoreVal.toFixed(2).replace(/\.00$/, '')
            : m.highest_score;
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
        UI.scoresList.innerHTML =
          '<div class="log-entry info">Môn học này không có bài tập AI.</div>';
      }
    }
  } catch (err) {
    if (UI.scoresSubjectTitle) UI.scoresSubjectTitle.textContent = 'Lỗi kết nối trang EDUX';
    if (UI.scoresList)
      UI.scoresList.innerHTML = `<div class="log-entry error">Không thể lấy điểm số: ${err.message}</div>`;
  }
}

export function initScoresPanel() {
  if (UI.btnRefreshScores) {
    UI.btnRefreshScores.addEventListener('click', loadExerciseScores);
  }
}
