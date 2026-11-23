/**
 * Lexis Wiki Dashboard - Glossary Modal & Ledger Management Module
 * Delegates to existing tested runtime implementations while maintaining ES Module exports.
 */

export function openBookGlossaryModal(series, book) {
    if (typeof window.openBookGlossaryModal === 'function') {
        return window.openBookGlossaryModal(series, book);
    }
}

export function fetchAndRenderGlossary(series, book) {
    if (typeof window.fetchAndRenderGlossary === 'function') {
        return window.fetchAndRenderGlossary(series, book);
    }
}

export function filterGlossaryEntries() {
    if (typeof window.filterGlossaryEntries === 'function') {
        return window.filterGlossaryEntries();
    }
}

export function triggerGlossarySync() {
    if (typeof window.triggerGlossarySync === 'function') {
        return window.triggerGlossarySync();
    }
}

export function triggerGlossaryPdfInput() {
    if (typeof window.triggerGlossaryPdfInput === 'function') {
        return window.triggerGlossaryPdfInput();
    }
}

export function handleGlossaryPdfSelect(event) {
    if (typeof window.handleGlossaryPdfSelect === 'function') {
        return window.handleGlossaryPdfSelect(event);
    }
}

export function uploadAndImportGlossaryPdf(file) {
    if (typeof window.uploadAndImportGlossaryPdf === 'function') {
        return window.uploadAndImportGlossaryPdf(file);
    }
}
