/**
 * Shared Core Quiz Utilities for Lexis Wiki Interactive Handouts
 */

function cleanMarkdownText(text) {
    if (!text || typeof text !== 'string') return '';
    // 1. Remove markdown bold/italic asterisks and backticks e.g. **Option B ("...")** -> Option B ("...")
    let res = text.replace(/[*`]/g, '');
    // 2. Clean escaped quote residues e.g. \"word\" -> "word"
    res = res.replace(/\\"/g, '"');
    // 3. Normalize multiple whitespace
    res = res.replace(/[ \t]+/g, ' ').trim();
    return res;
}

function escapeHtml(str) {
    if (!str || typeof str !== 'string') return '';
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function renderL2AuditBadgeHTML(expertAudit) {
    if (!expertAudit) return '';
    const passed = expertAudit.pass_audit !== false;
    const accuracy = Math.round((expertAudit.blind_solve_accuracy ?? 1.0) * 100);
    const score = expertAudit.overall_quality_score ?? 100;
    const isSuccess = passed && accuracy >= 80;

    const bg = isSuccess ? 'rgba(16, 185, 129, 0.12)' : 'rgba(245, 158, 11, 0.12)';
    const color = isSuccess ? '#059669' : '#d97706';
    const border = isSuccess ? 'rgba(16, 185, 129, 0.3)' : 'rgba(245, 158, 11, 0.3)';

    const cStats = expertAudit.cure_stats || null;
    let label = `L2 Expert Audit: ${score}% (${accuracy}% Blind Accuracy)`;
    if (cStats && (cStats.repair > 0 || cStats.rewrite > 0)) {
        label = `L2 Expert Audit: ${score}% (Surgically Cured: +${cStats.repair + cStats.rewrite})`;
    }
    
    let iconSvg = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
            <polyline points="9 12 11 14 15 10"></polyline>
        </svg>
    `;
    let titleTooltip = expertAudit.summary_verdict || '';

    if (!isSuccess) {
        iconSvg = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="8" x2="12" y2="12"></line>
                <line x1="12" y1="16" x2="12.01" y2="16"></line>
            </svg>
        `;
        label = `⚠️ L2 Review Needed: ${score}% (${accuracy}% Blind Accuracy)`;
    }

    return `
        <div class="l2-qa-badge" style="background: ${bg}; color: ${color}; border-color: ${border};">
            ${iconSvg}
            <span>${label}</span>
            <span class="l2-qa-tooltip">${escapeHtml(titleTooltip)}</span>
        </div>
    `;
}
