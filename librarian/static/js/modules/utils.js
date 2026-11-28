/**
 * Lexis Wiki Dashboard - Utilities Module
 * String formatting, modal wrappers, toast notifications, and DOM helpers.
 */

// Helper to safely escape string values for use in inline JS inside HTML attributes (e.g. onclick)
export function esc(str) {
    if (!str) return '';
    return str.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/"/g, '\\"');
}

export function formatRelativeTime(val) {
    if (!val) return '';
    try {
        const date = typeof val === 'number'
            ? new Date(val < 1e12 ? val * 1000 : val)
            : new Date(val);
        if (Number.isNaN(date.getTime())) return '';
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

// ------------------ REUSABLE BOOTSTRAP 5 SYSTEM MODAL ------------------
let appSystemModalResolver = null;

export function showSystemModal({ title = 'Notice', message = '', icon = 'bi-info-circle-fill', iconColor = 'text-info', iconBg = 'rgba(14, 165, 233, 0.12)', confirmText = 'OK', cancelText = null, isConfirm = false }) {
    return new Promise((resolve) => {
        appSystemModalResolver = resolve;
        
        const labelEl = document.getElementById('appSystemModalLabel');
        const bodyEl = document.getElementById('appSystemModalBody');
        const iconEl = document.getElementById('appSystemModalIcon');
        const iconBgEl = document.getElementById('appSystemModalIconBg');
        const cancelBtn = document.getElementById('appSystemModalCancelBtn');
        const confirmBtn = document.getElementById('appSystemModalConfirmBtn');
        const modalEl = document.getElementById('appSystemModal');

        if (!modalEl || !confirmBtn) {
            resolve(false);
            return;
        }

        if (labelEl) labelEl.innerText = title;
        if (bodyEl) bodyEl.innerHTML = message;
        
        if (iconEl) iconEl.className = `bi ${icon} ${iconColor} fs-5`;
        if (iconBgEl) iconBgEl.style.background = iconBg;
        
        if (cancelBtn) {
            if (isConfirm || cancelText) {
                cancelBtn.classList.remove('d-none');
                cancelBtn.innerText = cancelText || 'Cancel';
            } else {
                cancelBtn.classList.add('d-none');
            }
        }
        
        confirmBtn.innerText = confirmText;
        if (iconColor.includes('danger')) {
            confirmBtn.className = 'btn btn-sm btn-danger px-3.5 py-1.5 rounded-pill font-monospace fw-semibold';
        } else if (iconColor.includes('warning')) {
            confirmBtn.className = 'btn btn-sm btn-warning text-dark px-3.5 py-1.5 rounded-pill font-monospace fw-semibold';
        } else {
            confirmBtn.className = 'btn btn-sm btn-primary px-3.5 py-1.5 rounded-pill font-monospace fw-semibold';
        }
        
        const modalInstance = typeof bootstrap !== 'undefined' ? bootstrap.Modal.getOrCreateInstance(modalEl) : null;
        if (!modalInstance) {
            resolve(false);
            return;
        }
        
        let resolved = false;
        
        const onConfirmClick = () => {
            if (resolved) return;
            resolved = true;
            cleanup();
            modalInstance.hide();
            if (appSystemModalResolver) {
                appSystemModalResolver(true);
                appSystemModalResolver = null;
            }
        };
        
        const onHidden = () => {
            if (resolved) return;
            resolved = true;
            cleanup();
            if (appSystemModalResolver) {
                appSystemModalResolver(false);
                appSystemModalResolver = null;
            }
        };
        
        function cleanup() {
            confirmBtn.removeEventListener('click', onConfirmClick);
            modalEl.removeEventListener('hidden.bs.modal', onHidden);
        }
        
        confirmBtn.addEventListener('click', onConfirmClick);
        modalEl.addEventListener('hidden.bs.modal', onHidden);
        
        modalInstance.show();
    });
}

export function showConfirmModal(title, message, confirmText = 'Overwrite') {
    return showSystemModal({
        title: title || 'Confirmation Required',
        message: message,
        icon: 'bi-exclamation-triangle-fill',
        iconColor: 'text-warning',
        iconBg: 'rgba(245, 158, 11, 0.15)',
        confirmText: confirmText,
        cancelText: 'Cancel',
        isConfirm: true
    });
}

export function showInfoModal(title, message) {
    return showSystemModal({
        title: title || 'Information',
        message: message,
        icon: 'bi-info-circle-fill',
        iconColor: 'text-info',
        iconBg: 'rgba(14, 165, 233, 0.12)',
        confirmText: 'OK',
        isConfirm: false
    });
}

// ------------------ TOAST NOTIFICATION ------------------
export function showToast(message, type = 'success') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `bento-card p-3 shadow-lg d-flex align-items-center gap-2 border`;
    toast.style.minWidth = '280px';
    toast.style.background = 'var(--surface-glass)';
    toast.style.backdropFilter = 'blur(12px)';
    toast.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
    toast.style.transform = 'translateX(100px)';
    toast.style.opacity = '0';
    toast.style.borderRadius = 'var(--radius-default)';
    
    let icon = '🧠';
    let borderColor = 'var(--primary)';
    if (type === 'success') {
        icon = '<i class="bi bi-check-circle-fill text-success fs-5"></i>';
        borderColor = 'var(--accent-success)';
    } else if (type === 'error') {
        icon = '<i class="bi bi-exclamation-triangle-fill text-danger fs-5"></i>';
        borderColor = 'var(--accent-error)';
    } else {
        icon = '<i class="bi bi-info-circle-fill text-info fs-5"></i>';
        borderColor = 'var(--primary)';
    }
    
    toast.style.borderColor = borderColor;
    
    toast.innerHTML = `
        <div class="d-flex align-items-center gap-2 w-100">
            <div>${icon}</div>
            <div class="text-on-surface fw-semibold small" style="font-size: 0.82rem; margin-top: 2px;">${message}</div>
            <button class="btn-close ms-auto" style="font-size: 0.65rem;" onclick="this.parentElement.parentElement.remove()"></button>
        </div>
    `;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.transform = 'translateX(0)';
        toast.style.opacity = '1';
    }, 10);
    
    setTimeout(() => {
        toast.style.transform = 'translateX(100px)';
        toast.style.opacity = '0';
        setTimeout(() => { toast.remove(); }, 300);
    }, 4000);
}

// Expose on window for global / inline HTML backward compatibility
if (typeof window !== 'undefined') {
    window.esc = esc;
    window.formatRelativeTime = formatRelativeTime;
    window.showToast = showToast;
    window.showSystemModal = showSystemModal;
    window.showConfirmModal = showConfirmModal;
    window.showInfoModal = showInfoModal;
}
