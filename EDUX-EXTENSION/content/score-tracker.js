/**
 * EDUX Slayers - Exercise Score & Badge Tracker
 * Tracks assignment completion status, injects highest score badges & warning banners,
 * and renders visual progress (slides & exams) on the student dashboard (/student).
 */

(function () {
  'use strict';

  // Subject page state
  let cachedSubjectModels = null;
  let cachedSubjectId = null;

  // Student dashboard state
  let cachedJoinedSubjects = null;
  let cachedSubjectsSummary = null;
  let isFetchingDashboard = false;

  let scoreObserver = null;
  let scoreRenderDebounceTimer = null;

  function getSubjectIdFromUrl() {
    try {
      const url = new URL(window.location.href);
      return url.searchParams.get('id');
    } catch (e) {
      return null;
    }
  }

  function isStudentDashboardPage() {
    return window.location.pathname.includes('/student') && !getSubjectIdFromUrl();
  }

  function normalizeTitleKey(str) {
    if (!str) return '';
    return str
      .toLowerCase()
      .replace(/[\s\-_.:,()]+/g, '')
      .trim();
  }

  // =========================================================================
  // 1. Single Subject Operations (/subject?id=...)
  // =========================================================================

  async function fetchSubjectModels(subjectId, force = false) {
    if (!subjectId) return null;
    if (cachedSubjectModels && cachedSubjectId === subjectId && !force) {
      return cachedSubjectModels;
    }

    try {
      const res = await fetch(`/api/subjects/${subjectId}/models`, {
        credentials: 'include'
      });
      if (!res.ok) return null;
      const json = await res.json();
      if (json && Array.isArray(json.data)) {
        cachedSubjectModels = json.data;
        cachedSubjectId = subjectId;
        renderExerciseScoreBadges(cachedSubjectModels);
        return cachedSubjectModels;
      }
    } catch (e) {
      console.warn('[EDUX Slayers] Error fetching subject models:', e);
    }
    return null;
  }

  function renderSubjectOverviewBanner(models, container) {
    if (!models || !models.length || !container) return;

    let existingBanner = document.getElementById('edux-subject-exercise-banner');
    if (!existingBanner) {
      existingBanner = document.createElement('div');
      existingBanner.id = 'edux-subject-exercise-banner';
      existingBanner.className = 'edux-subject-overview-banner';
      container.prepend(existingBanner);
    }

    const examModels = models.filter((m) => m.exist_exam);
    const totalExams = examModels.length;
    const completedExams = examModels.filter((m) => m.highest_score !== null && m.highest_score !== undefined);
    const pendingExams = examModels.filter((m) => m.highest_score === null || m.highest_score === undefined);

    const scores = completedExams
      .map((m) => parseFloat(m.highest_score))
      .filter((s) => !isNaN(s));

    const maxScore = scores.length ? Math.max(...scores).toFixed(2).replace(/\.00$/, '') : '0';
    const avgScore = scores.length ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(2).replace(/\.00$/, '') : '0';
    const completionPercent = totalExams > 0 ? Math.round((completedExams.length / totalExams) * 100) : 0;

    let alertHtml = '';
    if (pendingExams.length > 0) {
      alertHtml = `
        <div class="edux-banner-alert">
          <div style="display: flex; align-items: center; gap: 6px;">
            <span>⚠️</span>
            <span>Cảnh báo: Bạn còn <strong>${pendingExams.length}</strong> bài tập chưa làm (chưa có kết quả)!</span>
          </div>
          <button type="button" id="edux-btn-jump-pending" style="background: #e11d48; color: white; border: none; border-radius: 6px; padding: 4px 10px; font-size: 11px; font-weight: 700; cursor: pointer;">
            🎯 Cuộn đến bài chưa làm
          </button>
        </div>
      `;
    } else if (totalExams > 0) {
      alertHtml = `
        <div class="edux-banner-alert all-done">
          <span>🎉 Tuyệt vời! Bạn đã hoàn thành 100% tất cả các bài tập môn học này!</span>
        </div>
      `;
    }

    existingBanner.innerHTML = `
      <div class="edux-banner-header">
        <div class="edux-banner-title">
          <span>⚔️ EDUX Slayers • Trạng Thái Hoàn Thành Bài Tập</span>
        </div>
        <div class="edux-banner-stats">
          <div class="edux-stat-item" title="Số lượng bài tập AI đã có điểm">
            <span>📝 Đã nộp:</span>
            <strong>${completedExams.length}/${totalExams} (${completionPercent}%)</strong>
          </div>
          <div class="edux-stat-item" title="Điểm số cao nhất đạt được trong các bài tập">
            <span>🏆 Cao nhất:</span>
            <strong style="color: #059669;">${maxScore}/10</strong>
          </div>
          <div class="edux-stat-item" title="Điểm trung bình các bài đã nộp">
            <span>⭐ Trung bình:</span>
            <strong style="color: #2563eb;">${avgScore}/10</strong>
          </div>
        </div>
      </div>
      ${alertHtml}
    `;

    const jumpBtn = existingBanner.querySelector('#edux-btn-jump-pending');
    if (jumpBtn) {
      jumpBtn.onclick = () => {
        const firstWarning = document.querySelector('.edux-exercise-badge-warning');
        if (firstWarning) {
          firstWarning.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      };
    }
  }

  function renderExerciseScoreBadges(models) {
    if (!models || !models.length) return;

    // 1. Locate curriculum container for overview banner
    const firstAccordion = document.querySelector('button.w-full.flex.items-center.justify-between');
    const curriculumContainer = firstAccordion
      ? firstAccordion.closest('div.space-y-4, div.flex-1, main, section') || firstAccordion.parentElement
      : null;
    if (curriculumContainer) {
      renderSubjectOverviewBanner(models, curriculumContainer);
    }

    // 2. Build map of models by normalized title for fast lookup
    const modelByTitle = new Map();
    models.forEach((m) => {
      const key = normalizeTitleKey(m.title);
      if (key) modelByTitle.set(key, m);
    });

    // 3. Find all lesson cards/rows
    const actionButtons = Array.from(document.querySelectorAll('button, a')).filter((el) => {
      const text = (el.textContent || '').trim();
      return text.includes('Bài tập AI') || text.includes('Bài giảng AI') || text === 'Bài giảng';
    });

    const cardContainers = new Set();
    actionButtons.forEach((btn) => {
      const btnRow = btn.parentElement;
      if (btnRow) cardContainers.add(btnRow);
    });

    cardContainers.forEach((btnRow) => {
      // Walk up to find the lesson card that contains the h3 title
      let card = btnRow.parentElement;
      while (card && card !== document.body && !card.querySelector('h3, h4, h2')) {
        card = card.parentElement;
      }
      if (!card || card === document.body) return;

      // Extract card title (EDUX puts lesson titles in h3 / h4 tags)
      let titleEl = card.querySelector('h3, h4, h2');
      if (!titleEl) {
        const candidates = Array.from(card.querySelectorAll('p, div, span, button[title]')).filter((el) => {
          const t = (el.getAttribute('title') || el.textContent || '').trim();
          return t && !/^\d+(\.\d+)?%$/.test(t) && t.length > 3 && !t.includes('Bài giảng') && !t.includes('Bài tập');
        });
        titleEl = candidates[0] || null;
      }
      const cardTitle = titleEl ? (titleEl.getAttribute('title') || titleEl.textContent || '').trim() : (card.textContent || '').trim();
      const normCardTitle = normalizeTitleKey(cardTitle);

      // Match model
      let matchedModel = modelByTitle.get(normCardTitle);
      if (!matchedModel) {
        for (const [key, m] of modelByTitle.entries()) {
          if (key && (normCardTitle.includes(key) || key.includes(normCardTitle))) {
            matchedModel = m;
            break;
          }
        }
      }

      if (!matchedModel) return;

      const btBtn = Array.from(btnRow.querySelectorAll('button, a')).find((b) => (b.textContent || '').includes('Bài tập AI'));
      if (!matchedModel.exist_exam && !btBtn) return;

      // Check / create badge element
      let badge = btnRow.querySelector('.edux-exercise-badge');
      if (!badge) {
        badge = document.createElement('div');
        badge.className = 'edux-exercise-badge';
        if (btBtn && btBtn.nextSibling) {
          btnRow.insertBefore(badge, btBtn.nextSibling);
        } else if (btBtn) {
          btnRow.appendChild(badge);
        } else {
          btnRow.appendChild(badge);
        }
      }

      const hasScore = matchedModel.highest_score !== null && matchedModel.highest_score !== undefined;

      if (hasScore) {
        const scoreVal = parseFloat(matchedModel.highest_score);
        const displayScore = isNaN(scoreVal) ? matchedModel.highest_score : scoreVal.toFixed(2).replace(/\.00$/, '');
        badge.className = 'edux-exercise-badge edux-exercise-badge-success';
        badge.innerHTML = `<span>🏆</span><span>Điểm: ${displayScore}/10</span>`;
        badge.title = `Điểm số cao nhất: ${displayScore} / 10`;
      } else {
        badge.className = 'edux-exercise-badge edux-exercise-badge-warning';
        badge.innerHTML = `<span>⚠️</span><span>Chưa làm</span>`;
        badge.title = `Cảnh báo: Bạn chưa hoàn thành bài tập này!`;
      }
    });
  }

  // =========================================================================
  // 2. Student Dashboard Operations (/student)
  // =========================================================================

  function calculateSubjectStats(subject, models) {
    let totalSlides = 0;
    let doneSlides = 0;
    let totalPresentations = 0;
    let donePresentations = 0;
    let totalExams = 0;
    let doneExams = 0;
    const scores = [];

    (models || []).forEach((m) => {
      // Slides calculation from process_document ("38/38") or exist_presentation
      const pDoc = m.process_document;
      const hasPres = m.exist_presentation;
      if (hasPres || pDoc) {
        totalPresentations++;
        if (pDoc && typeof pDoc === 'string' && pDoc.includes('/')) {
          const parts = pDoc.split('/');
          const cur = parseInt(parts[0], 10) || 0;
          const tot = parseInt(parts[1], 10) || 0;
          doneSlides += cur;
          totalSlides += tot;
          if (cur >= tot && tot > 0) {
            donePresentations++;
          }
        }
      }

      // Exams calculation from exist_exam & highest_score
      if (m.exist_exam) {
        totalExams++;
        if (m.highest_score !== null && m.highest_score !== undefined) {
          doneExams++;
          const val = parseFloat(m.highest_score);
          if (!isNaN(val)) scores.push(val);
        }
      }
    });

    const slidePercent = totalSlides > 0
      ? Math.round((doneSlides / totalSlides) * 100)
      : (totalPresentations > 0 && donePresentations === totalPresentations ? 100 : 0);

    const examPercent = totalExams > 0 ? Math.round((doneExams / totalExams) * 100) : 100;
    const pendingExams = totalExams - doneExams;
    const pendingSlides = totalSlides - doneSlides;
    const isAllDone = (pendingExams <= 0) && (pendingSlides <= 0);

    const avgScore = scores.length ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(2).replace(/\.00$/, '') : null;
    const maxScore = scores.length ? Math.max(...scores).toFixed(2).replace(/\.00$/, '') : null;

    return {
      id: subject.id,
      code: (subject.subject_code || '').trim(),
      name: (subject.name || '').trim(),
      semesterId: subject.semester_id,
      totalModels: models ? models.length : 0,
      totalSlides,
      doneSlides,
      totalPresentations,
      donePresentations,
      totalExams,
      doneExams,
      pendingExams,
      pendingSlides,
      slidePercent,
      examPercent,
      isAllDone,
      avgScore,
      maxScore
    };
  }

  async function fetchJoinedSubjectsProgress(force = false) {
    if (isFetchingDashboard) return cachedSubjectsSummary;
    isFetchingDashboard = true;

    try {
      // 1. Check local storage cache for instant rendering
      if (!force && !cachedSubjectsSummary && typeof chrome !== 'undefined' && chrome.storage && chrome.storage.local) {
        const stored = await new Promise((resolve) => {
          chrome.storage.local.get(['edux_subjects_progress_cache'], (res) => {
            resolve(res ? res.edux_subjects_progress_cache : null);
          });
        });

        if (stored && stored.timestamp && Date.now() - stored.timestamp < 10 * 60 * 1000 && Array.isArray(stored.data)) {
          cachedSubjectsSummary = stored.data;
          renderStudentDashboardProgress(cachedSubjectsSummary);
        }
      }

      // 2. Fetch list of joined subjects
      let subjects = cachedJoinedSubjects;
      if (!subjects || force) {
        const res = await fetch('/api/subjects/joined', { credentials: 'include' });
        if (!res.ok) {
          isFetchingDashboard = false;
          return cachedSubjectsSummary;
        }
        const json = await res.json();
        if (json && Array.isArray(json.data)) {
          subjects = json.data;
          cachedJoinedSubjects = subjects;
        }
      }

      if (!subjects || !subjects.length) {
        isFetchingDashboard = false;
        return null;
      }

      // 3. Fetch models for each subject concurrently
      const summaryList = await Promise.all(
        subjects.map(async (subj) => {
          try {
            const mRes = await fetch(`/api/subjects/${subj.id}/models`, { credentials: 'include' });
            if (mRes.ok) {
              const mJson = await mRes.json();
              if (mJson && Array.isArray(mJson.data)) {
                return calculateSubjectStats(subj, mJson.data);
              }
            }
          } catch (e) {}
          return calculateSubjectStats(subj, []);
        })
      );

      cachedSubjectsSummary = summaryList;
      renderStudentDashboardProgress(cachedSubjectsSummary);

      // 4. Save to chrome.storage.local for fast cache
      if (typeof chrome !== 'undefined' && chrome.storage && chrome.storage.local) {
        chrome.storage.local.set({
          edux_subjects_progress_cache: {
            timestamp: Date.now(),
            data: cachedSubjectsSummary
          }
        });
      }

      isFetchingDashboard = false;
      return cachedSubjectsSummary;
    } catch (e) {
      console.warn('[EDUX Slayers] Error fetching dashboard progress:', e);
      isFetchingDashboard = false;
      return null;
    }
  }

  function renderStudentDashboardProgress(summaryList) {
    if (!summaryList || !summaryList.length) return;
    if (!isStudentDashboardPage()) return;

    // Fast lookup maps
    const byCode = new Map();
    const byName = new Map();

    summaryList.forEach((s) => {
      if (s.code) byCode.set(s.code.toUpperCase(), s);
      const nameKey = normalizeTitleKey(s.name);
      if (nameKey) byName.set(nameKey, s);
    });

    // Find course cards on /student
    const cards = Array.from(document.querySelectorAll('div')).filter((d) => {
      const cls = d.className || '';
      return typeof cls === 'string' && cls.includes('cursor-pointer') && (d.textContent || '').includes('Học kỳ');
    });

    cards.forEach((card) => {
      const cardText = card.textContent || '';
      let matched = null;

      // 1. Match by subject code (INFO3008, PROG3002...)
      for (const [code, item] of byCode.entries()) {
        if (cardText.includes(code)) {
          matched = item;
          break;
        }
      }

      // 2. Fallback: match by title
      if (!matched) {
        const titleEl = card.querySelector('h3, h4, h2');
        const cardTitle = titleEl ? (titleEl.textContent || '').trim() : '';
        const normTitle = normalizeTitleKey(cardTitle);
        matched = byName.get(normTitle);
        if (!matched && normTitle) {
          for (const [nameKey, item] of byName.entries()) {
            if (normTitle.includes(nameKey) || nameKey.includes(normTitle)) {
              matched = item;
              break;
            }
          }
        }
      }

      if (!matched) return;

      // Find the row container inside the card
      const rowFlex = card.querySelector('.flex.flex-row.w-full.justify-between') || card.querySelector('.flex.flex-row') || card;
      if (!rowFlex) return;

      // Ensure rowFlex has flex layout to place progress on the right
      rowFlex.style.display = 'flex';
      rowFlex.style.alignItems = 'center';

      let progEl = card.querySelector('.edux-card-progress');
      if (!progEl) {
        progEl = document.createElement('div');
        progEl.className = 'edux-card-progress';
        rowFlex.appendChild(progEl);
      }

      const slidePct = matched.slidePercent;
      const examPct = matched.examPercent;
      const isAllDone = matched.isAllDone;
      const pendingExams = matched.pendingExams;
      const pendingSlides = matched.pendingSlides;

      let badgeHtml = '';
      if (isAllDone) {
        badgeHtml = '<span class="edux-badge-complete">✓ Đã xong 100%</span>';
      } else if (pendingExams > 0) {
        badgeHtml = `<span class="edux-badge-warning">⚠️ Còn ${pendingExams} bài tập</span>`;
      } else if (pendingSlides > 0) {
        badgeHtml = `<span class="edux-badge-slide-warning">📖 Còn ${pendingSlides} slide</span>`;
      } else {
        badgeHtml = '<span class="edux-badge-complete">✓ Đã xong</span>';
      }

      const scoreTooltip = matched.avgScore ? `\n- ⭐ Điểm trung bình: ${matched.avgScore}/10 (Cao nhất: ${matched.maxScore}/10)` : '';
      progEl.title = `Chi tiết môn học:\n- 🖥️ Slide: ${matched.doneSlides}/${matched.totalSlides} trang (${matched.donePresentations}/${matched.totalPresentations} bài giảng)\n- 📝 Bài tập AI: ${matched.doneExams}/${matched.totalExams} bài đã nộp điểm${scoreTooltip}`;

      progEl.innerHTML = `
        <!-- Slide Stat -->
        <div class="edux-card-progress-stat" title="Tiến trình đọc Slide bài giảng">
          <div class="edux-stat-header">
            <span>🖥️ Slide</span>
            <span class="edux-stat-val">${matched.doneSlides}/${matched.totalSlides} (${slidePct}%)</span>
          </div>
          <div class="edux-progress-bar-bg">
            <div class="edux-progress-bar-fill ${slidePct === 100 ? 'edux-fill-done' : 'edux-fill-slide'}" style="width: ${slidePct}%"></div>
          </div>
        </div>

        <!-- Exam Stat -->
        <div class="edux-card-progress-stat" title="Tiến trình làm Bài tập AI">
          <div class="edux-stat-header">
            <span>📝 Bài tập</span>
            <span class="edux-stat-val">${matched.totalExams > 0 ? `${matched.doneExams}/${matched.totalExams} (${examPct}%)` : 'Không có'}</span>
          </div>
          <div class="edux-progress-bar-bg">
            <div class="edux-progress-bar-fill ${examPct === 100 ? 'edux-fill-done' : 'edux-fill-exam'}" style="width: ${examPct}%"></div>
          </div>
        </div>

        <!-- Badge -->
        <div>
          ${badgeHtml}
        </div>
      `;
    });
  }

  // =========================================================================
  // 3. Coordinator & Lifecycle
  // =========================================================================

  function initExerciseScoreTracker() {
    const subjectId = getSubjectIdFromUrl();
    if (subjectId) {
      fetchSubjectModels(subjectId);
    } else if (isStudentDashboardPage()) {
      fetchJoinedSubjectsProgress();
    }

    if (!scoreObserver) {
      scoreObserver = new MutationObserver(() => {
        clearTimeout(scoreRenderDebounceTimer);
        scoreRenderDebounceTimer = setTimeout(() => {
          const currentSubId = getSubjectIdFromUrl();
          if (currentSubId) {
            if (currentSubId !== cachedSubjectId) {
              fetchSubjectModels(currentSubId);
            } else if (cachedSubjectModels) {
              renderExerciseScoreBadges(cachedSubjectModels);
            }
          } else if (isStudentDashboardPage()) {
            if (cachedSubjectsSummary) {
              renderStudentDashboardProgress(cachedSubjectsSummary);
            } else {
              fetchJoinedSubjectsProgress();
            }
          }
        }, 200);
      });

      scoreObserver.observe(document.body, {
        childList: true,
        subtree: true
      });
    }
  }

  window.__EDUX_RENDER_SCORES__ = () => {
    if (cachedSubjectModels) {
      renderExerciseScoreBadges(cachedSubjectModels);
    }
    if (cachedSubjectsSummary) {
      renderStudentDashboardProgress(cachedSubjectsSummary);
    }
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initExerciseScoreTracker);
  } else {
    initExerciseScoreTracker();
  }

  window.EduxScoreTracker = {
    init: initExerciseScoreTracker,
    refresh: () => {
      const subjectId = getSubjectIdFromUrl();
      if (subjectId) {
        fetchSubjectModels(subjectId, true);
      } else if (isStudentDashboardPage()) {
        fetchJoinedSubjectsProgress(true);
      }
    },
    setCachedModels: (models) => {
      cachedSubjectModels = models;
      cachedSubjectId = getSubjectIdFromUrl();
      renderExerciseScoreBadges(cachedSubjectModels);
    },
    setJoinedSubjects: (payload) => {
      const list = Array.isArray(payload) ? payload : (payload && Array.isArray(payload.data) ? payload.data : null);
      if (list) {
        cachedJoinedSubjects = list;
        fetchJoinedSubjectsProgress(true);
      }
    },
    getScores: async () => {
      const subjectId = getSubjectIdFromUrl();
      if (cachedSubjectModels) {
        return { success: true, subjectId, models: cachedSubjectModels };
      }
      if (subjectId) {
        const models = await fetchSubjectModels(subjectId);
        return { success: true, subjectId, models };
      }
      return { success: false, message: 'Không phải trang môn học EDUX' };
    },
    getAllSubjectsProgress: async () => {
      if (cachedSubjectsSummary) {
        return { success: true, isDashboard: true, subjects: cachedSubjectsSummary };
      }
      const data = await fetchJoinedSubjectsProgress();
      return { success: true, isDashboard: true, subjects: data || [] };
    }
  };
})();
