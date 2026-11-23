/**
 * Lexis Wiki Dashboard - Book Glossary Module
 * Cumulative Lexical Ledger (CLL) controller: browsing, filtering, unit sync & PDF ingest.
 */

import { esc, showToast } from './utils.js';

// =========================================================================
// SECTION 6: BOOK GLOSSARY WORKBENCH CONTROLLER
// =========================================================================
let activeGlossarySeries = null;
let activeGlossaryBook = null;
let activeGlossaryEntries = [];

async function openBookGlossaryModal(series, book) {
    activeGlossarySeries = series;
    activeGlossaryBook = String(book || '1');

    const titleEl = document.getElementById('bookGlossaryModalTitle');
    const searchInput = document.getElementById('glossary-search-input');
    const unitFilter = document.getElementById('glossary-unit-filter');
    const cefrFilter = document.getElementById('glossary-cefr-filter');
    const tierFilter = document.getElementById('glossary-tier-filter');

    const seriesName = typeof getSeriesDisplayName === 'function' ? getSeriesDisplayName(activeGlossarySeries) : (typeof window !== 'undefined' && window.getSeriesDisplayName ? window.getSeriesDisplayName(activeGlossarySeries) : activeGlossarySeries);
    if (titleEl) titleEl.textContent = `${seriesName} · Book ${activeGlossaryBook} Glossary`;
    if (searchInput) searchInput.value = '';
    if (unitFilter) unitFilter.value = 'all';
    if (cefrFilter) cefrFilter.value = 'all';
    if (tierFilter) tierFilter.value = 'all';

    const modalEl = document.getElementById('book-glossary-modal');
    if (modalEl) {
        bootstrap.Modal.getOrCreateInstance(modalEl).show();
    }

    await fetchAndRenderGlossary(activeGlossarySeries, activeGlossaryBook);
}

async function fetchAndRenderGlossary(series, book) {
    const tbody = document.getElementById('glossary-table-body');
    const totalBadge = document.getElementById('glossary-total-badge');
    const cefrBox = document.getElementById('glossary-cefr-breakdown');
    const tierBox = document.getElementById('glossary-tier-breakdown');
    const updatedEl = document.getElementById('glossary-last-updated');

    if (tbody) {
        tbody.innerHTML = `<tr><td colspan="6" class="text-center py-4 text-muted"><span class="spinner-border spinner-border-sm me-2"></span>Loading Book ${esc(String(book))} Glossary...</td></tr>`;
    }

    try {
        const resp = await fetch(`/api/book-glossary?series=${encodeURIComponent(series)}&book=${encodeURIComponent(book)}`);
        const data = await resp.json();
        if (!data.success) {
            if (tbody) tbody.innerHTML = `<tr><td colspan="6" class="text-center py-4 text-danger">${esc(data.error || 'Failed to load glossary')}</td></tr>`;
            return;
        }

        const summary = data.summary || {};
        const entriesObj = data.entries || {};
        activeGlossaryEntries = Object.values(entriesObj).sort((a, b) => a.lemma.localeCompare(b.lemma));

        if (totalBadge) totalBadge.textContent = `${activeGlossaryEntries.length} words`;
        if (updatedEl && summary.last_updated) {
            const dt = new Date(summary.last_updated);
            updatedEl.textContent = `Updated: ${dt.toLocaleDateString()} ${dt.toLocaleTimeString()}`;
        }

        // Render CEFR breakdown pills
        if (cefrBox) {
            const cefrs = summary.cefr_breakdown || {};
            const cefrOrder = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'Unassigned'];
            let pillsHtml = `<span class="text-muted small me-1">CEFR:</span>`;
            cefrOrder.forEach(lvl => {
                const cnt = cefrs[lvl];
                if (cnt) {
                    const bgClass = lvl === 'A1' ? 'bg-success-subtle text-success border-success-subtle' :
                                    lvl === 'A2' ? 'bg-info-subtle text-info border-info-subtle' :
                                    lvl === 'B1' ? 'bg-primary-subtle text-primary border-primary-subtle' :
                                    lvl === 'B2' ? 'bg-warning-subtle text-warning border-warning-subtle' :
                                    lvl === 'C1' ? 'bg-danger-subtle text-danger border-danger-subtle' :
                                    lvl === 'C2' ? 'bg-danger text-white' : 'bg-secondary-subtle text-secondary';
                    pillsHtml += `<span class="badge ${bgClass} border px-2 py-0.5 rounded-pill" style="font-size: 0.68rem; font-weight: 600;">${lvl}: ${cnt}</span>`;
                }
            });
            cefrBox.innerHTML = pillsHtml;
        }

        // Render SLA Progression breakdown pills
        if (tierBox) {
            const tiers = summary.acquisition_tier_breakdown || {};
            const totalE = activeGlossaryEntries.length || 1;
            const newCnt = tiers['target_new'] || 0;
            const shiftCnt = tiers['recycled_sense_shift'] || 0;
            const reviewCnt = tiers['recycled_spaced_review'] || 0;

            const newPct = Math.round((newCnt / totalE) * 100);
            const shiftPct = Math.round((shiftCnt / totalE) * 100);
            const reviewPct = Math.round((reviewCnt / totalE) * 100);

            let tierHtml = `<span class="text-muted small me-1">SLA Tiers:</span>`;
            tierHtml += `<span class="badge bg-success-subtle text-success border border-success-subtle px-2 py-0.5 rounded-pill" style="font-size: 0.68rem; font-weight: 600;" title="New Target Words (~80-85%)"><i class="bi bi-star-fill me-1" style="font-size: 0.6rem;"></i>New: ${newCnt} (${newPct}%)</span>`;
            if (shiftCnt > 0) {
                tierHtml += `<span class="badge bg-warning-subtle text-warning border border-warning-subtle px-2 py-0.5 rounded-pill" style="font-size: 0.68rem; font-weight: 600;" title="Sense Shifts / Recategorized (~10-15%)"><i class="bi bi-shuffle me-1" style="font-size: 0.6rem;"></i>Shift: ${shiftCnt} (${shiftPct}%)</span>`;
            }
            if (reviewCnt > 0) {
                tierHtml += `<span class="badge bg-info-subtle text-info border border-info-subtle px-2 py-0.5 rounded-pill" style="font-size: 0.68rem; font-weight: 600;" title="Spaced Retrieval Review (~0-5%)"><i class="bi bi-clock-history me-1" style="font-size: 0.6rem;"></i>Review: ${reviewCnt} (${reviewPct}%)</span>`;
            }
            tierBox.innerHTML = tierHtml;
        }

        // Populate unit filter options
        const unitFilter = document.getElementById('glossary-unit-filter');
        if (unitFilter) {
            const unitDist = summary.unit_distribution || {};
            let uOpts = `<option value="all">All Units (${activeGlossaryEntries.length})</option>`;
            Object.keys(unitDist).sort().forEach(u => {
                uOpts += `<option value="${esc(u)}">${esc(u.replace(/_/g, ' '))} (${unitDist[u]})</option>`;
            });
            unitFilter.innerHTML = uOpts;
        }

        filterGlossaryEntries();

    } catch (err) {
        if (tbody) {
            tbody.innerHTML = `<tr><td colspan="6" class="text-center py-4 text-danger">Error: ${esc(err.message)}</td></tr>`;
        }
    }
}

function filterGlossaryEntries() {
    const query = (document.getElementById('glossary-search-input')?.value || '').trim().toLowerCase();
    const selectedUnit = document.getElementById('glossary-unit-filter')?.value || 'all';
    const selectedCefr = document.getElementById('glossary-cefr-filter')?.value || 'all';
    const selectedTier = document.getElementById('glossary-tier-filter')?.value || 'all';
    const tbody = document.getElementById('glossary-table-body');
    const countEl = document.getElementById('glossary-visible-count');

    const filtered = activeGlossaryEntries.filter(e => {
        if (query) {
            const lemma = (e.lemma || '').toLowerCase();
            const disp = (e.display_word || '').toLowerCase();
            const def = (e.definition || '').toLowerCase();
            if (!lemma.includes(query) && !disp.includes(query) && !def.includes(query)) {
                return false;
            }
        }
        if (selectedUnit !== 'all') {
            const u = e.first_unit || e.unit_code || '';
            if (u !== selectedUnit) return false;
        }
        if (selectedCefr !== 'all') {
            const lvl = e.cefr || 'Unassigned';
            if (lvl !== selectedCefr) return false;
        }
        if (selectedTier !== 'all') {
            const tier = e.acquisition_tier || 'target_new';
            if (tier !== selectedTier) return false;
        }
        return true;
    });

    if (countEl) {
        countEl.textContent = `${filtered.length} of ${activeGlossaryEntries.length} entries`;
    }

    if (!tbody) return;

    if (filtered.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" class="text-center py-4 text-muted">No matching vocabulary entries found.</td></tr>`;
        return;
    }

    let rowsHtml = '';
    filtered.slice(0, 300).forEach((e, idx) => {
        const word = e.display_word || e.lemma;
        const pos = e.pos || 'noun';
        const cefr = e.cefr || '';
        const cefrBadge = cefr ? `<span class="badge bg-secondary-subtle text-secondary border px-1.5 py-0.5 rounded font-monospace" style="font-size: 0.65rem;">${esc(cefr)}</span>` : '<span class="text-muted" style="font-size: 0.7rem;">--</span>';
        const unitLabel = (e.first_unit || e.unit_code || 'General').replace(/_/g, ' ');
        const defText = e.definition ? `<div class="text-truncate" title="${esc(e.definition)}" style="max-width: 480px; color: var(--on-surface);">${esc(e.definition)}</div>` : '';
        const exText = (e.used_examples && e.used_examples[0]) ? `<div class="small text-muted fst-italic text-truncate mt-0.5" title="${esc(e.used_examples[0])}" style="max-width: 480px;"><i class="bi bi-quote me-1"></i>${esc(e.used_examples[0])}</div>` : '';

        // Progression Tier Badge
        let tierBadge = '';
        if (e.acquisition_tier === 'recycled_sense_shift') {
            const bNum = e.prior_exposure?.book ? ` · B${esc(e.prior_exposure.book)}` : '';
            tierBadge = `<span class="badge bg-warning-subtle text-warning border border-warning-subtle px-1.5 py-0.5 rounded font-monospace ms-1.5" style="font-size: 0.62rem;" title="${esc(e.tier_label || 'Sense Shift')}"><i class="bi bi-shuffle me-0.5"></i>Shift${bNum}</span>`;
        } else if (e.acquisition_tier === 'recycled_spaced_review') {
            const bNum = e.prior_exposure?.book ? ` · B${esc(e.prior_exposure.book)}` : '';
            tierBadge = `<span class="badge bg-info-subtle text-info border border-info-subtle px-1.5 py-0.5 rounded font-monospace ms-1.5" style="font-size: 0.62rem;" title="${esc(e.tier_label || 'Spaced Review')}"><i class="bi bi-clock-history me-0.5"></i>Review${bNum}</span>`;
        }

        rowsHtml += `
            <tr>
                <td class="ps-3 text-muted font-monospace text-center">${idx + 1}</td>
                <td>
                    <div class="d-flex align-items-center flex-wrap">
                        <span class="fw-semibold text-info font-monospace" style="font-size: 0.82rem;">${esc(word)}</span>
                        ${tierBadge}
                    </div>
                </td>
                <td class="text-center"><span class="badge bg-dark-subtle text-muted border px-1.5 py-0.5 rounded font-monospace" style="font-size: 0.65rem;">${esc(pos)}</span></td>
                <td class="text-center">${cefrBadge}</td>
                <td class="text-center">
                    <span class="badge bg-info-subtle text-info border border-info-subtle px-1.5 py-0.5 rounded text-truncate d-inline-block" style="max-width: 150px; font-size: 0.68rem;" title="${esc(unitLabel)}">
                        ${esc(unitLabel)}
                    </span>
                </td>
                <td>
                    ${defText || '<span class="text-muted" style="font-size: 0.72rem;">Pending extraction sync</span>'}
                    ${exText}
                </td>
            </tr>
        `;
    });

    if (filtered.length > 300) {
        rowsHtml += `<tr><td colspan="6" class="text-center py-2 text-muted small">Showing first 300 of ${filtered.length} entries. Use search to narrow down.</td></tr>`;
    }

    tbody.innerHTML = rowsHtml;
}

async function triggerGlossarySync() {
    if (!activeGlossarySeries || !activeGlossaryBook) {
        showToast('No active series or book selected.', 'error');
        return;
    }

    const syncBtn = document.getElementById('btn-glossary-sync');
    if (syncBtn) {
        syncBtn.disabled = true;
        syncBtn.innerHTML = `<span class="spinner-border spinner-border-sm me-1.5"></span>Syncing...`;
    }

    try {
        const resp = await fetch('/api/sync-book-glossary', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ series: activeGlossarySeries, book: activeGlossaryBook })
        });
        const data = await resp.json();
        if (data.success) {
            const res = data.result || {};
            showToast(`Synced ${res.harvested_count || 0} entries from ${res.matching_units_scanned || 0} units!`, 'success');
            await fetchAndRenderGlossary(activeGlossarySeries, activeGlossaryBook);
        } else {
            showToast(`Sync failed: ${data.error || 'Unknown error'}`, 'error');
        }
    } catch (err) {
        showToast(`Sync error: ${err.message}`, 'error');
    } finally {
        if (syncBtn) {
            syncBtn.disabled = false;
            syncBtn.innerHTML = `<i class="bi bi-arrow-repeat me-1.5"></i>Sync from Units`;
        }
    }
}

function triggerGlossaryPdfInput() {
    const input = document.getElementById('glossary-pdf-input');
    if (input) {
        input.value = "";
        input.click();
    }
}

function handleGlossaryPdfSelect(e) {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    uploadAndImportGlossaryPdf(file);
}

function uploadAndImportGlossaryPdf(file) {
    const filename = file.name;
    if (!filename.toLowerCase().endsWith('.pdf')) {
        showToast('Please select a valid PDF file.', 'error');
        return;
    }

    const importBtn = document.getElementById('btn-glossary-import');
    if (importBtn) {
        importBtn.disabled = true;
        importBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-1.5" role="status" aria-hidden="true"></span>Importing...';
    }

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/import-pdf-glossary', true);
    xhr.setRequestHeader('X-Filename', encodeURIComponent(filename));
    if (activeGlossarySeries) {
        xhr.setRequestHeader('X-Series', encodeURIComponent(activeGlossarySeries));
    }
    if (activeGlossaryBook) {
        xhr.setRequestHeader('X-Book', encodeURIComponent(activeGlossaryBook));
    }
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');

    xhr.onload = async function() {
        if (importBtn) {
            importBtn.disabled = false;
            importBtn.innerHTML = '<i class="bi bi-file-earmark-pdf me-1.5"></i>Import PDF';
        }
        if (xhr.status === 200) {
            try {
                const data = JSON.parse(xhr.responseText);
                if (data.success) {
                    const r = data.result || {};
                    showToast(`Successfully imported ${r.entries_added || 0} new words and updated ${r.entries_updated || 0}! Total: ${r.total_entries || 0}`, 'success');
                    await fetchAndRenderGlossary(activeGlossarySeries, activeGlossaryBook);
                } else {
                    showToast(`Import failed: ${data.error || 'Unknown error'}`, 'error');
                }
            } catch (err) {
                showToast(`Response parse error: ${err.message}`, 'error');
            }
        } else {
            showToast(`Upload failed with HTTP status ${xhr.status}`, 'error');
        }
    };

    xhr.onerror = function() {
        if (importBtn) {
            importBtn.disabled = false;
            importBtn.innerHTML = '<i class="bi bi-file-earmark-pdf me-1.5"></i>Import PDF';
        }
        showToast('Network error during PDF upload.', 'error');
    };

    xhr.send(file);
}

// Bindings for window global scope (backward compatibility with inline HTML onclick handlers)
if (typeof window !== 'undefined') {
    window.openBookGlossaryModal = openBookGlossaryModal;
    window.fetchAndRenderGlossary = fetchAndRenderGlossary;
    window.filterGlossaryEntries = filterGlossaryEntries;
    window.triggerGlossarySync = triggerGlossarySync;
    window.triggerGlossaryPdfInput = triggerGlossaryPdfInput;
    window.handleGlossaryPdfSelect = handleGlossaryPdfSelect;
    window.uploadAndImportGlossaryPdf = uploadAndImportGlossaryPdf;
    window.openImportPdfModal = triggerGlossaryPdfInput;
}

export {
    openBookGlossaryModal,
    fetchAndRenderGlossary,
    filterGlossaryEntries,
    triggerGlossarySync,
    triggerGlossaryPdfInput,
    handleGlossaryPdfSelect,
    uploadAndImportGlossaryPdf
};
