/**
 * Lexis Wiki Dashboard - Hero Board & Level-2 Expert Re-Audit Module
 * LLM Leaderboard, Re-Audit batch triggering, and psychometric audit cards.
 */

import { esc, showInfoModal } from './utils.js';

// ------------------ LEVEL-2 EXPERT RE-AUDIT (UNIT) ------------------
let lastReAuditResult = null;
let currentReAuditUnit = null;
let activeReAuditTab = 'quiz';

function switchReAuditTab(tab) {
    activeReAuditTab = (tab === 'extraction') ? 'extraction' : 'quiz';
    const qBtn = document.getElementById('re-audit-tab-quiz');
    const eBtn = document.getElementById('re-audit-tab-extraction');
    if (qBtn) {
        qBtn.classList.toggle('btn-success', activeReAuditTab === 'quiz');
        qBtn.classList.toggle('btn-outline-secondary', activeReAuditTab !== 'quiz');
    }
    if (eBtn) {
        eBtn.classList.toggle('btn-success', activeReAuditTab === 'extraction');
        eBtn.classList.toggle('btn-outline-secondary', activeReAuditTab !== 'extraction');
    }
    if (lastReAuditResult) {
        renderReAuditRows(lastReAuditResult);
    }
}

function triggerUnitReAudit(unitName) {
    if (!unitName) return;
    currentReAuditUnit = unitName;
    const modal = document.getElementById('re-audit-modal');
    const titleEl = document.getElementById('reAuditModalLabel');
    if (titleEl) titleEl.textContent = `Re-Audit: ${unitName}`;
    const inst = bootstrap.Modal.getOrCreateInstance(modal);
    inst.show();

    setReAuditRunning(true);
    showReAuditSummary(null, true);
    updateReAuditProgress({ status: 'running', progress: 0, logs: [`Starting Level-2 re-audit for ${unitName} (Extractions + Quizzes)…`] });
    fetch('/api/re-audit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            target: 'all',
            template: null,
            units: [unitName],
            only_flagged: false,
        })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            if (typeof window !== 'undefined' && window.pollJobStatus) window.pollJobStatus(data.job_id, 're-audit');
            if (typeof window !== 'undefined' && window.fetchRecentJobs) window.fetchRecentJobs();
        } else {
            setReAuditRunning(false);
            showInfoModal('Re-audit Failed', data.error || 'Failed to start the re-audit job.');
        }
    })
    .catch(err => {
        setReAuditRunning(false);
        showInfoModal('Re-audit Error', err.message || String(err));
    });
}

function rerunCurrentUnitAudit() {
    if (currentReAuditUnit) triggerUnitReAudit(currentReAuditUnit);
}

function openReAuditModal() {
    const modal = document.getElementById('re-audit-modal');
    const inst = bootstrap.Modal.getOrCreateInstance(modal);
    const unitSel = document.getElementById('re-audit-unit');
    const fillUnits = (filesList) => {
        if (!unitSel || !Array.isArray(filesList)) return;
        const prevVal = unitSel.value || '';
        const unitSet = new Set();
        filesList.forEach(f => {
            const u = f.stem || (f.name ? String(f.name).replace(/\.md$/i, '') : '');
            if (u) unitSet.add(u);
        });
        const sortedUnits = Array.from(unitSet).sort((a, b) => a.localeCompare(b));
        const opts = ['<option value="">All Units</option>'];
        sortedUnits.forEach(u => {
            const escU = esc(u);
            opts.push(`<option value="${escU}">${escU}</option>`);
        });
        unitSel.innerHTML = opts.join('');
        unitSel.value = sortedUnits.includes(prevVal) ? prevVal : '';
    };
    fetch('/api/raw-files')
        .then(r => r.json())
        .then(d => {
            if (Array.isArray(d.files)) fillUnits(d.files);
        })
        .catch(() => {});
    onReAuditTargetChange();
    if (lastReAuditResult) showReAuditSummary(lastReAuditResult);
    inst.show();
}

function onReAuditTargetChange() {
    const targetEl = document.getElementById('re-audit-target');
    const tplEl = document.getElementById('re-audit-template');
    const flaggedEl = document.getElementById('re-audit-only-flagged');
    const targetVal = targetEl ? targetEl.value : 'quiz';

    if (tplEl) {
        const prevTpl = tplEl.value || '';
        const quizOpts = [
            ['', 'All Types'],
            ['vocabulary', 'Vocabulary'],
            ['reading', 'Reading'],
            ['translation', 'Translation'],
            ['listening', 'Listening'],
            ['video', 'Video'],
            ['cloze', 'Cloze'],
        ];
        const extOpts = [
            ['', 'All Types'],
            ['vocabulary', 'Vocabulary'],
            ['grammar', 'Grammar'],
            ['summary', 'Summary'],
            ['mindmap', 'Mindmap'],
        ];
        if (targetVal === 'all') {
            tplEl.innerHTML = '<option value="">All Types</option>';
            tplEl.value = '';
            tplEl.disabled = true;
        } else {
            const list = targetVal === 'extraction' ? extOpts : quizOpts;
            tplEl.innerHTML = list.map(([v, lbl]) => `<option value="${v}">${lbl}</option>`).join('');
            tplEl.disabled = false;
            tplEl.value = list.some(([v]) => v === prevTpl) ? prevTpl : '';
        }
    }
    if (flaggedEl) {
        const isExtOnly = targetVal === 'extraction';
        flaggedEl.disabled = isExtOnly;
        if (isExtOnly) flaggedEl.checked = false;
    }
}

function updateReAuditProgress(job) {
    const panel = document.getElementById('re-audit-live-panel');
    const txt = document.getElementById('re-audit-progress-text');
    const pct = document.getElementById('re-audit-progress-pct');
    const bar = document.getElementById('re-audit-progress-bar');
    const logsEl = document.getElementById('re-audit-live-logs');
    if (!panel || !job) return;

    panel.classList.remove('d-none');
    const p = Math.max(0, Math.min(100, Number(job.progress || 0)));
    if (pct) pct.textContent = `${p}%`;
    if (bar) {
        bar.style.width = `${p}%`;
        bar.classList.toggle('progress-bar-animated', job.status === 'running');
        bar.classList.toggle('bg-danger', job.status === 'failed');
        bar.classList.toggle('bg-success', job.status !== 'failed');
    }
    const logs = Array.isArray(job.logs) ? job.logs : [];
    const latest = logs.length > 0 ? logs[logs.length - 1] : (job.status === 'running' ? 'Running…' : job.status);
    if (txt) txt.textContent = latest;
    if (logsEl) {
        logsEl.textContent = logs.join('\n');
        logsEl.scrollTop = logsEl.scrollHeight;
    }
}

function setReAuditRunning(running) {
    const btn = document.getElementById('re-audit-run-btn');
    const spinner = document.getElementById('re-audit-run-spinner');
    const icon = document.getElementById('re-audit-run-icon');
    const label = document.getElementById('re-audit-run-label');
    if (btn) btn.disabled = running;
    if (spinner) spinner.classList.toggle('d-none', !running);
    if (icon) icon.classList.toggle('d-none', running);
    if (label) label.textContent = running ? 'RE-AUDITING…' : 'RE-RUN';
}

function startReAuditAll() {
    const target = (document.getElementById('re-audit-target')?.value || 'quiz').trim();
    const template = (document.getElementById('re-audit-template')?.value || '').trim();
    const unit = (document.getElementById('re-audit-unit')?.value || '').trim();
    const onlyFlagged = !!document.getElementById('re-audit-only-flagged')?.checked;

    const scopeDesc = [
        `Target: ${target}`,
        template ? `Type: ${template}` : 'Type: All',
        unit ? `Unit: ${unit}` : 'Unit: ALL',
        onlyFlagged ? 'Filter: Only Failed/Unaudited' : null,
    ].filter(Boolean).join(' | ');

    if (!confirm(`Run Level-2 expert re-audit?\n\n${scopeDesc}\n\nThis is read-only and only calls the judge model (the generator is not re-run).`)) return;
    setReAuditRunning(true);
    showReAuditSummary(null, true);
    updateReAuditProgress({ status: 'running', progress: 0, logs: [`Starting re-audit (${scopeDesc})…`] });
    fetch('/api/re-audit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            target,
            template: template || null,
            units: unit ? [unit] : null,
            only_flagged: onlyFlagged,
        })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            if (typeof window !== 'undefined' && window.pollJobStatus) window.pollJobStatus(data.job_id, 're-audit');
            if (typeof window !== 'undefined' && window.fetchRecentJobs) window.fetchRecentJobs();
        } else {
            setReAuditRunning(false);
            showInfoModal('Re-audit Failed', data.error || 'Failed to start the re-audit job.');
        }
    })
    .catch(err => {
        setReAuditRunning(false);
        showInfoModal('Re-audit Error', err.message || String(err));
    });
}

function regenerateFromReAudit(unit, template) {
    if (!unit || !template) return;
    const count = (template === 'reading' || template === 'listening') ? 5 : 10;
    fetch('/api/generate-quiz', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: unit, template, count })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            if (typeof window !== 'undefined' && window.showToast) window.showToast(`Regenerating ${unit} [${template}]…`, 'info');
            if (typeof window !== 'undefined' && window.pollJobStatus) window.pollJobStatus(data.job_id, 'quiz');
            if (typeof window !== 'undefined' && window.fetchRecentJobs) window.fetchRecentJobs();
        } else {
            showInfoModal('Regenerate Failed', data.error || 'Failed to start quiz generation.');
        }
    })
    .catch(err => showInfoModal('Regenerate Error', err.message || String(err)));
}

function recompileFromReAudit(unit, category) {
    if (!unit) return;
    const payload = category ? { filename: unit, categories: [category] } : { filename: unit };
    fetch('/api/compile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            const catTag = category ? ` [${category}]` : ' extractions';
            if (typeof window !== 'undefined' && window.showToast) window.showToast(`Re-compiling ${unit}${catTag}…`, 'info');
            if (typeof window !== 'undefined' && window.pollJobStatus) window.pollJobStatus(data.job_id, 'compile');
            if (typeof window !== 'undefined' && window.fetchRecentJobs) window.fetchRecentJobs();
        } else {
            showInfoModal('Re-compile Failed', data.error || 'Failed to start compilation.');
        }
    })
    .catch(err => showInfoModal('Re-compile Error', err.message || String(err)));
}

function showReAuditSummary(summary, running) {
    const stats = document.getElementById('re-audit-stats');
    const body = document.getElementById('re-audit-body');
    const qBadge = document.getElementById('re-audit-tab-quiz-badge');
    const eBadge = document.getElementById('re-audit-tab-extraction-badge');
    if (!summary) {
        if (stats) stats.innerHTML = '';
        if (qBadge) { qBadge.textContent = '0'; qBadge.className = 'badge rounded-pill bg-dark bg-opacity-25 ms-1'; }
        if (eBadge) { eBadge.textContent = '0'; eBadge.className = 'badge rounded-pill bg-secondary-subtle text-secondary ms-1'; }
        if (body) {
            body.innerHTML = running
                ? '<tr><td colspan="8" class="text-center text-muted py-4"><span class="spinner-border spinner-border-sm me-2"></span>Re-auditing in the background… (watch the Recent Jobs panel).</td></tr>'
                : '<tr><td colspan="8" class="text-center text-muted py-4">No re-audit run yet. Click <strong>RE-RUN</strong> to run the Level-2 expert judge.</td></tr>';
        }
        return;
    }
    lastReAuditResult = summary;

    const allRows = (summary.results || []).filter(r => r.template !== 'cloze');
    const quizRows = allRows.filter(r => !(r.artifact_group === 'extraction' || String(r.template || '').startsWith('extraction')));
    const extRows = allRows.filter(r => r.artifact_group === 'extraction' || String(r.template || '').startsWith('extraction'));

    if (qBadge) {
        const qFail = quizRows.filter(r => !r.passed || r.error).length;
        qBadge.textContent = qFail > 0 ? `${quizRows.length} · ${qFail} FAIL` : String(quizRows.length);
        qBadge.className = qFail > 0 ? 'badge rounded-pill bg-danger text-white ms-1' : 'badge rounded-pill bg-dark bg-opacity-25 ms-1';
    }
    if (eBadge) {
        const eFail = extRows.filter(r => !r.passed || r.error).length;
        eBadge.textContent = eFail > 0 ? `${extRows.length} · ${eFail} FAIL` : String(extRows.length);
        eBadge.className = eFail > 0 ? 'badge rounded-pill bg-danger text-white ms-1' : 'badge rounded-pill bg-secondary-subtle text-secondary ms-1';
    }

    const acc = (summary.avg_blind_solve_accuracy != null) ? Math.round(summary.avg_blind_solve_accuracy * 100) + '%' : '—';
    const statCell = (val, color, label) =>
        `<div class="col d-flex flex-column align-items-center border rounded-3 py-2" style="border-color: var(--card-border) !important;">
            <span class="fs-5 fw-bold">${val}</span><span class="small ${color}">${label}</span>
         </div>`;
    if (stats) {
        stats.innerHTML =
            statCell(summary.passed, 'text-success', 'PASSED') +
            statCell(summary.failed, 'text-danger', 'FAILED') +
            statCell(summary.errors, 'text-warning', 'ERRORS') +
            statCell(acc, 'text-info', 'AVG BLIND-SOLVE');
    }

    if (activeReAuditTab === 'quiz' && quizRows.length === 0 && extRows.length > 0) {
        switchReAuditTab('extraction');
        return;
    }
    if (activeReAuditTab === 'extraction' && extRows.length === 0 && quizRows.length > 0) {
        switchReAuditTab('quiz');
        return;
    }
    switchReAuditTab(activeReAuditTab);
}

function renderReAuditRows(summary) {
    const body = document.getElementById('re-audit-body');
    if (!body || !summary) return;
    const escHtml = (s) => String(s == null ? '' : s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

    const allRows = (summary.results || []).filter(r => r.template !== 'cloze');
    const targetRows = allRows.filter(r => {
        const isExt = r.artifact_group === 'extraction' || String(r.template || '').startsWith('extraction');
        return activeReAuditTab === 'extraction' ? isExt : !isExt;
    });

    const rows = targetRows.map(r => {
        const accR = (r.blind_solve_accuracy != null) ? Math.round(r.blind_solve_accuracy * 100) + '%' : '—';
        const prev = (r.prev_passed == null) ? '<span class="text-muted">n/a</span>'
            : (r.prev_passed ? '<span class="badge bg-success-subtle text-success">PASS</span>' : '<span class="badge bg-danger-subtle text-danger">FAIL</span>');
        const now = r.error ? '<span class="badge bg-warning-subtle text-warning">ERROR</span>'
            : (r.passed ? '<span class="badge bg-success-subtle text-success">PASS</span>' : '<span class="badge bg-danger-subtle text-danger">FAIL</span>');
        let change;
        if (r.notes) change = `<span class="${r.passed ? 'text-muted' : 'text-danger'}">${escHtml(r.notes)}</span>`;
        else if (r.prev_passed == null) change = '<span class="text-muted">first audit</span>';
        else if (r.prev_passed && !r.passed) change = 'PASS → FAIL';
        else if (!r.prev_passed && r.passed) change = 'FAIL → PASS';
        else change = '<span class="text-muted">unchanged</span>';
        if (!r.passed && summary.report_file) {
            change += ` <a href="/logs/${encodeURIComponent(summary.report_file)}" target="_blank" class="text-info text-decoration-none font-monospace" title="Open full JSON report">[JSON]</a>`;
        }

        const safeU = escHtml(r.unit).replace(/'/g, "\\'");
        const safeT = escHtml(r.template).replace(/'/g, "\\'");
        const isExtRow = r.artifact_group === 'extraction' || String(r.template || '').startsWith('extraction');
        const extCat = (String(r.template || '').startsWith('extraction:') ? String(r.template).split(':')[1] : '');
        const safeExtCat = escHtml(extCat).replace(/'/g, "\\'");
        const badgeLbl = extCat || r.template;
        const actionBtns = isExtRow
            ? `<button class="btn btn-sm btn-outline-warning py-0 px-1.5" onclick="recompileFromReAudit('${safeU}', '${safeExtCat}')" title="Re-compile ${escHtml(badgeLbl)}"><i class="bi bi-arrow-clockwise"></i></button>`
            : `<a class="btn btn-sm btn-outline-secondary py-0 px-1.5" href="/wiki/${encodeURIComponent(r.unit)}/handouts/${encodeURIComponent(r.unit)}_${encodeURIComponent(r.template)}_quiz.html" target="_blank" title="Open handout"><i class="bi bi-box-arrow-up-right"></i></a>` +
              `<button class="btn btn-sm btn-outline-warning py-0 px-1.5 ms-1" onclick="regenerateFromReAudit('${safeU}', '${safeT}')" title="Regenerate this quiz"><i class="bi bi-arrow-clockwise"></i></button>`;

        return `<tr>
            <td class="font-monospace small">${escHtml(r.unit)} <span class="badge bg-secondary-subtle text-secondary">${escHtml(badgeLbl)}</span></td>
            <td class="text-center">${r.item_count == null ? '—' : r.item_count}</td>
            <td class="text-center fw-semibold">${accR}</td>
            <td class="text-center">${r.overall_quality_score == null ? '—' : r.overall_quality_score}</td>
            <td class="text-center">${prev}</td>
            <td class="text-center">${now}</td>
            <td class="small">${change}${r.error ? `<div class="text-warning mt-1">${escHtml(r.error)}</div>` : ''}</td>
            <td class="text-end text-nowrap">${actionBtns}</td>
        </tr>`;
    }).join('');
    body.innerHTML = rows || `<tr><td colspan="8" class="text-center text-muted py-3">No ${activeReAuditTab} items in this audit.</td></tr>`;
}

// ------------------ LLM HERO BOARD & QA AUDIT MODAL ------------------
function openHeroBoardModal() {
    const modal = document.getElementById('hero-board-modal');
    const modalInstance = bootstrap.Modal.getOrCreateInstance(modal);
    modalInstance.show();
    loadHeroBoardData();
}

function setHeroBoardLoading(loading) {
    const btn = document.getElementById('hero-board-reload-btn');
    const spinner = document.getElementById('hero-board-reload-spinner');
    const icon = document.getElementById('hero-board-reload-icon');
    const label = document.getElementById('hero-board-reload-label');
    const topStat = document.getElementById('hero-board-top-stat');
    const tbody = document.getElementById('hero-board-leaderboard-body');
    const list = document.getElementById('hero-board-audits-list');

    if (btn) btn.disabled = loading;
    if (spinner) spinner.classList.toggle('d-none', !loading);
    if (icon) icon.classList.toggle('d-none', loading);
    if (label) label.textContent = loading ? 'EVALUATING…' : 'L1 LOG RE-EVALUATE';
    if (topStat) { topStat.classList.add('d-none'); topStat.innerHTML = ''; }

    if (loading) {
        const skelCell = (w) => `<td style="border-color: var(--card-border);"><div class="hb-skeleton" style="width:${w};"></div></td>`;
        if (tbody) {
            tbody.innerHTML = `<tr>${skelCell('34px')}${skelCell('55%')}${skelCell('80%')}${skelCell('70%')}${skelCell('70%')}${skelCell('70%')}${skelCell('45%')}</tr>`;
        }
        if (list) {
            list.innerHTML = `<div class="card border p-3" style="background: var(--surface-dim); border-color: var(--card-border) !important; border-radius: var(--radius-md);">
                <div class="hb-skeleton mb-2" style="width:40%;"></div>
                <div class="hb-skeleton mb-2" style="width:70%;"></div>
                <div class="hb-skeleton" style="width:90%;"></div>
            </div>`;
        }
    }
}

function loadHeroBoardData() {
    setHeroBoardLoading(true);
    fetch('/api/hero-board')
        .then(r => r.json())
        .then(data => {
            renderHeroBoard(data.hero_board || []);
            renderAuditsList(data.audits || []);
        })
        .catch(err => {
            console.error('Failed to load hero board data:', err);
            renderHeroBoard([]);
            renderAuditsList([]);
        })
        .finally(() => setHeroBoardLoading(false));
}

const HB_COLUMNS = 7; // single source of truth for the leaderboard's column count
// Keep the static "Loading…" placeholder's colspan in sync so adding/removing a column
// only ever requires bumping HB_COLUMNS above (no scattered 'colspan="7"' to hunt down).
(function () {
    if (typeof document !== 'undefined') {
        document.querySelectorAll('[data-hb-colspan]').forEach(function (td) { td.setAttribute('colspan', String(HB_COLUMNS)); });
    }
})();

function hbBarHTML(value, dim) {
    if (value === null || value === undefined) {
        return '<span class="hb-bar-value text-muted">—</span>';
    }
    const v = Math.max(0, Math.min(100, Number(value) || 0));
    const tone = v >= 90 ? 'good' : (v >= 75 ? 'warn' : 'bad');
    return `<div class="hb-bar-wrap"><div class="hb-bar ${dim ? 'dim' : ''}"><div class="hb-bar-fill ${tone}" style="width:${v}%"></div></div><span class="hb-bar-value">${v}%</span></div>`;
}

function hbRunsHTML(item) {
    const runs = Number(item.runs) || 0;
    const succ = Number(item.success_runs) || 0;
    const failed = Number(item.failed_runs) || 0;
    const rate = runs > 0 ? Math.round((succ / runs) * 100) : 0;
    const failedDot = failed > 0 ? ` <span class="hb-failed" title="${failed} failed run(s)">&#8226;</span>` : '';
    return `<span class="hb-runs">${runs} runs <span class="hb-succ">${rate}% ok</span>${failedDot}</span>`;
}

function hbDimHTML(label, val, max, title) {
    const esc = String(title || '').replace(/"/g, '&quot;');
    if (val === null || val === undefined) {
        return `<div class="hb-dim" title="${esc}"><div class="hb-dim-top"><span class="hb-dim-label">${label}</span><span class="hb-dim-val text-muted">N/A</span></div><div class="hb-bar dim"><div class="hb-bar-fill bad" style="width:0%"></div></div></div>`;
    }
    const pct = Math.max(0, Math.min(100, Math.round((Number(val) / max) * 100)));
    const tone = pct >= 80 ? 'good' : (pct >= 60 ? 'warn' : 'bad');
    return `<div class="hb-dim" title="${esc}"><div class="hb-dim-top"><span class="hb-dim-label">${label}</span><span class="hb-dim-val">${pct}%</span></div><div class="hb-bar dim"><div class="hb-bar-fill ${tone}" style="width:${pct}%"></div></div></div>`;
}

function hbFlagHTML(flag) {
    const text = String(flag).trim();
    let tone = 'hb-flag-info';
    let icon = 'bi-info-circle';
    if (text.startsWith('❌') || text.startsWith('🔴')) {
        tone = 'hb-flag-error';
        icon = 'bi-exclamation-octagon-fill';
    } else if (text.startsWith('⚠️') || text.startsWith('⚠')) {
        tone = 'hb-flag-warn';
        icon = 'bi-exclamation-triangle-fill';
    }
    const clean = text.replace(/^[❌🔴⚠\uFE0F]+\s*/, '').trim() || text;
    return `<div class="hb-flag ${tone}"><i class="bi ${icon}"></i><span>${clean}</span></div>`;
}

function renderTopStat(top) {
    const el = document.getElementById('hero-board-top-stat');
    if (!el || !top) return;
    el.innerHTML = `
        <div class="hb-top-medal"><i class="bi bi-trophy-fill"></i></div>
        <div class="flex-grow-1">
            <div class="hb-top-meta text-on-surface-variant">Top Performer</div>
            <div class="d-flex align-items-baseline gap-2 flex-wrap">
                <span class="hb-top-score">${top.composite_score}%</span>
                <span class="fw-bold text-on-surface" style="font-size:1.05rem;">${top.model}</span>
            </div>
            <div class="hb-top-meta text-on-surface-variant mt-1">${top.runs} runs · Schema ${top.schema_adherence_avg}% · Verbatim ${top.verbatim_faithfulness_avg}% · Pedagogy ${top.pedagogical_quality_avg}%</div>
        </div>`;
    el.classList.remove('d-none');
}

let _hbBoard = [];
let _hbSort = { key: 'composite_score', dir: 'desc' };

function sortHeroBoard(key) {
    if (_hbSort.key === key) {
        _hbSort.dir = _hbSort.dir === 'desc' ? 'asc' : 'desc';
    } else {
        _hbSort.key = key;
        _hbSort.dir = (key === 'model') ? 'asc' : 'desc';
    }
    renderHeroBoard(_hbBoard);
}

function renderHeroBoard(board) {
    const tbody = document.getElementById('hero-board-leaderboard-body');
    if (!tbody) return;
    const topStat = document.getElementById('hero-board-top-stat');
    if (!board || board.length === 0) {
        _hbBoard = [];
        if (topStat) { topStat.classList.add('d-none'); topStat.innerHTML = ''; }
        tbody.innerHTML = `<tr><td colspan="${HB_COLUMNS}"><div class="hb-empty"><i class="bi bi-inbox"></i><div>No log files found to evaluate yet.</div><div class="small mt-1">Run <code class="font-monospace">lexis compile</code> or generate quizzes to produce evaluation logs.</div></div></td></tr>`;
        document.querySelectorAll('.hb-sort-ind').forEach(ind => { ind.classList.remove('active'); ind.style.transform = ''; });
        return;
    }
    _hbBoard = board.slice();

    const leader = _hbBoard.reduce((a, b) => (Number(b.composite_score) > Number(a.composite_score) ? b : a));
    const leaderModel = leader.model;
    const leaderScore = Number(leader.composite_score);

    const sorted = _hbBoard.slice().sort((a, b) => {
        let av = a[_hbSort.key], bv = b[_hbSort.key];
        if (typeof av === 'string' || typeof bv === 'string') {
            av = String(av == null ? '' : av).toLowerCase();
            bv = String(bv == null ? '' : bv).toLowerCase();
        } else {
            av = Number(av) || 0;
            bv = Number(bv) || 0;
        }
        if (av < bv) return _hbSort.dir === 'asc' ? -1 : 1;
        if (av > bv) return _hbSort.dir === 'asc' ? 1 : -1;
        return 0;
    });

    renderTopStat(leader);
    document.querySelectorAll('.hb-sort-ind').forEach(ind => {
        const active = ind.getAttribute('data-key') === _hbSort.key;
        ind.classList.toggle('active', active);
        ind.style.transform = (active && _hbSort.dir === 'asc') ? 'rotate(180deg)' : '';
    });

    tbody.innerHTML = sorted.map((item, idx) => {
        const rankHTML = idx < 3
            ? `<span class="hb-rank hb-rank-${idx + 1}">${idx + 1}</span>`
            : `<span class="hb-rank hb-rank-n">${idx + 1}</span>`;
        const isLeader = item.model === leaderModel;
        const championCls = isLeader ? ' class="hb-champion"' : '';
        const gap = Math.round((leaderScore - Number(item.composite_score)) * 10) / 10;
        const gapHtml = isLeader
            ? '<div class="hb-leader-chip"><i class="bi bi-stars"></i> LEADER</div>'
            : (gap > 0 ? `<div class="hb-gap">−${gap} vs leader</div>` : '');
        return `<tr${championCls} style="border-color: var(--card-border);">
            <td class="pl-3">${rankHTML}</td>
            <td class="fw-semibold text-on-surface">${item.model}</td>
            <td>${hbBarHTML(item.composite_score)}${gapHtml}</td>
            <td>${hbBarHTML(item.schema_adherence_avg, true)}</td>
            <td>${hbBarHTML(item.verbatim_faithfulness_avg, true)}</td>
            <td>${hbBarHTML(item.pedagogical_quality_avg, true)}</td>
            <td>${hbRunsHTML(item)}</td>
        </tr>`;
    }).join('');
}

function renderAuditsList(audits) {
    const container = document.getElementById('hero-board-audits-list');
    if (!container) return;
    if (!audits || audits.length === 0) {
        container.innerHTML = '<div class="text-center text-muted py-3">No log audit records available.</div>';
        return;
    }
    
    container.innerHTML = audits.map(audit => {
        const score = audit.composite_score;
        const badgeClass = score >= 90 
            ? 'hb-score-good' 
            : (score >= 75 ? 'hb-score-warn' : 'hb-score-bad');
        const flagsHtml = audit.flags && audit.flags.length > 0
            ? audit.flags.map(hbFlagHTML).join('')
            : '<div class="hb-pass-chip"><i class="bi bi-check-circle-fill"></i> 100% Passed all quality checks</div>';
            
        // Dimension scores breakdown (value + mini bar)
        const sc = audit.scores || {};
        const dimBadgesHtml = `
            <div class="d-flex flex-wrap gap-3 mb-2" style="max-width: 720px;">
                ${hbDimHTML('Schema', sc.schema_adherence, 25, 'Schema & Structural Adherence (Weight: 25%)')}
                ${hbDimHTML('Verbatim', sc.verbatim_faithfulness, 30, 'Source Faithfulness & Verbatim Overlap (Weight: 30%)')}
                ${hbDimHTML('Pedagogy', sc.pedagogical_quality, 25, 'Pedagogy & Diagnostic Integrity (Weight: 25%)')}
                ${hbDimHTML('Uniqueness', sc.uniqueness, 20, 'Uniqueness & Deduplication (Weight: 20%)')}
            </div>
        `;
            
        let scoreBadgeHtml = `<span class="badge border ${badgeClass} font-monospace fs-6">${score}% Score</span>`;
        if (audit.status === 'FAILED') {
            scoreBadgeHtml = `<span class="badge border hb-score-bad font-monospace fs-6"><i class="bi bi-x-octagon-fill me-1"></i>FAILED (0%)</span>`;
        } else if (audit.post_cure_score != null && audit.pre_cure_score != null) {
            const curedClass = audit.post_cure_score >= 80 ? 'hb-score-good' : 'hb-score-warn';
            scoreBadgeHtml = `
                <div class="d-flex align-items-center gap-1">
                    <span class="badge border hb-score-bad font-monospace" title="Raw generation score before surgical audit and cure">Pre: ${audit.pre_cure_score}%</span>
                    <i class="bi bi-arrow-right text-muted small"></i>
                    <span class="badge border ${curedClass} font-monospace fs-6" title="Final post-cure score delivered to handout">Final: ${audit.post_cure_score}% (Cured)</span>
                </div>
            `;
        }

        return `
            <div class="card border p-3" style="background: var(--surface-dim); border-color: var(--card-border) !important; border-radius: var(--radius-md);">
                <div class="d-flex align-items-center justify-content-between mb-2">
                    <div class="d-flex align-items-center gap-2">
                        ${scoreBadgeHtml}
                        <span class="fw-bold text-on-surface text-truncate" style="max-width: 380px;">${audit.log_name}</span>
                    </div>
                    <span class="badge bg-body-secondary text-primary border font-monospace" style="border-color: var(--card-border) !important;">${audit.model}</span>
                </div>
                ${dimBadgesHtml}
                <div class="p-2.5 rounded border" style="background: var(--surface); border-color: var(--card-border) !important;">
                    ${flagsHtml}
                </div>
            </div>
        `;
    }).join('');
}

// Bindings for window global scope (backward compatibility with inline HTML onclick handlers)
if (typeof window !== 'undefined') {
    window.triggerUnitReAudit = triggerUnitReAudit;
    window.rerunCurrentUnitAudit = rerunCurrentUnitAudit;
    window.switchReAuditTab = switchReAuditTab;
    window.openReAuditModal = openReAuditModal;
    window.onReAuditTargetChange = onReAuditTargetChange;
    window.updateReAuditProgress = updateReAuditProgress;
    window.setReAuditRunning = setReAuditRunning;
    window.startReAuditAll = startReAuditAll;
    window.regenerateFromReAudit = regenerateFromReAudit;
    window.recompileFromReAudit = recompileFromReAudit;
    window.showReAuditSummary = showReAuditSummary;
    window.openHeroBoardModal = openHeroBoardModal;
    window.setHeroBoardLoading = setHeroBoardLoading;
    window.loadHeroBoardData = loadHeroBoardData;
    window.hbBarHTML = hbBarHTML;
    window.hbRunsHTML = hbRunsHTML;
    window.hbDimHTML = hbDimHTML;
    window.hbFlagHTML = hbFlagHTML;
    window.renderTopStat = renderTopStat;
    window.sortHeroBoard = sortHeroBoard;
    window.renderHeroBoard = renderHeroBoard;
    window.renderAuditsList = renderAuditsList;
}

export {
    triggerUnitReAudit,
    rerunCurrentUnitAudit,
    switchReAuditTab,
    openReAuditModal,
    onReAuditTargetChange,
    updateReAuditProgress,
    setReAuditRunning,
    startReAuditAll,
    regenerateFromReAudit,
    recompileFromReAudit,
    showReAuditSummary,
    openHeroBoardModal,
    setHeroBoardLoading,
    loadHeroBoardData,
    hbBarHTML,
    hbRunsHTML,
    hbDimHTML,
    hbFlagHTML,
    renderTopStat,
    sortHeroBoard,
    renderHeroBoard,
    renderAuditsList
};
