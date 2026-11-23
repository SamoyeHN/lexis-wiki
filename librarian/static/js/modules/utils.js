/**
 * Lexis Wiki Dashboard - Utilities Module
 * String formatting, modal wrappers, toast notifications, and DOM helpers.
 */

export function esc(str) {
    if (!str) return '';
    return str.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/"/g, '\\"');
}

export function formatRelativeTime(isoString) {
    if (!isoString) return '';
    try {
        const date = new Date(isoString);
        const now = new Date();
        const diffSeconds = Math.floor((now - date) / 1000);
        if (diffSeconds < 60) return 'Just now';
        const diffMinutes = Math.floor(diffSeconds / 60);
        if (diffMinutes < 60) return `${diffMinutes}m ago`;
        const diffHours = Math.floor(diffMinutes / 60);
        if (diffHours < 24) return `${diffHours}h ago`;
        const diffDays = Math.floor(diffHours / 24);
        return `${diffDays}d ago`;
    } catch {
        return '';
    }
}

export function showToast(message, type = 'info') {
    if (typeof window.showToast === 'function') {
        window.showToast(message, type);
    }
}
