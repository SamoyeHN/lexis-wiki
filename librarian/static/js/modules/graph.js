/**
 * Lexis Wiki Dashboard - 2D Force Network Galaxy Map Module
 * Interactive Canvas visualization of knowledge nodes, dependencies, and connections.
 */

import { esc } from './utils.js';

// Canvas and Force Simulation State
let canvas = null;
let ctx = null;
let nodes = [];
let links = [];
let scale = 1;
let transformX = 0;
let transformY = 0;
let dragNode = null;
let hoveredNode = null;
let isDraggingCanvas = false;
let dragStart = { x: 0, y: 0 };
let hasAutoZoomed = false;
let currentGraphCategory = 'all';

function ensureCanvas() {
    if (!canvas && typeof document !== 'undefined') {
        canvas = document.getElementById('wiki-canvas');
        if (canvas) {
            ctx = canvas.getContext('2d');
        }
    }
    return Boolean(canvas && ctx);
}

// ------------------ DYNAMIC CANVAS FORCE NETWORK GALAXY MAP ------------------

function setGraphCategoryFilter(cat) {
    currentGraphCategory = cat;
    document.querySelectorAll('.bento-card [id^="graph-filter-"]').forEach(btn => btn.classList.remove('active'));
    const activeBtn = document.getElementById(`graph-filter-${cat}`);
    if (activeBtn) activeBtn.classList.add('active');
    drawGraph();
}

function exportGraphImage() {
    if (!canvas) return;
    const link = document.createElement('a');
    link.download = `linguistic_galaxy_map_${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
}

function resizeCanvas() {
    if (!ensureCanvas()) return;
    const box = canvas.parentElement;
    if (!box) return;
    canvas.width = box.clientWidth;
    canvas.height = box.clientHeight;
}

function refreshGraph() {
    hasAutoZoomed = false;
    fetch('/api/wiki-graph')
        .then(res => res.json())
        .then(data => {
            nodes = data.nodes || [];
            links = data.links || [];
            
            // Update badge count
            const badge = document.getElementById('galaxy-node-count-badge');
            if (badge) badge.innerText = `${nodes.length} nodes`;

            // Position nodes in radial galaxy distribution
            nodes.forEach((n, idx) => {
                const angle = (idx / (nodes.length || 1)) * Math.PI * 2;
                const radius = n.group === 'source' ? 50 : 160 + (Math.random() - 0.5) * 80;
                n.x = (canvas.width / 2) + Math.cos(angle) * radius;
                n.y = (canvas.height / 2) + Math.sin(angle) * radius;
                n.vx = (Math.random() - 0.5) * 2;
                n.vy = (Math.random() - 0.5) * 2;
            });
            
            runGraphSimulation();
        })
        .catch(err => { console.error('Galaxy graph load failed:', err); });
}

function runGraphSimulation() {
    function step() {
        // Apply electrostatic charges and spring dampening
        for (let i = 0; i < nodes.length; i++) {
            let n1 = nodes[i];
            for (let j = i + 1; j < nodes.length; j++) {
                let n2 = nodes[j];
                let dx = n2.x - n1.x;
                let dy = n2.y - n1.y;
                let dist = Math.sqrt(dx*dx + dy*dy) || 1;
                if (dist < 200) {
                    let force = (200 - dist) * 0.06;
                    n1.vx -= (dx / dist) * force;
                    n1.vy -= (dy / dist) * force;
                    n2.vx += (dx / dist) * force;
                    n2.vy += (dy / dist) * force;
                }
            }
        }

        links.forEach(l => {
            let source = nodes.find(n => n.id === l.source);
            let target = nodes.find(n => n.id === l.target);
            if (source && target) {
                let dx = target.x - source.x;
                let dy = target.y - source.y;
                let dist = Math.sqrt(dx*dx + dy*dy) || 1;
                let force = (dist - 90) * 0.035;
                source.vx += (dx / dist) * force;
                source.vy += (dy / dist) * force;
                target.vx -= (dx / dist) * force;
                target.vy -= (dy / dist) * force;
            }
        });

        nodes.forEach(n => {
            if (n !== dragNode) {
                n.x += n.vx;
                n.y += n.vy;
                n.vx *= 0.82;
                n.vy *= 0.82;
            }
        });

        drawGraph();
        if (dragNode || nodes.some(n => Math.abs(n.vx) > 0.05)) {
            requestAnimationFrame(step);
        } else {
            if (!hasAutoZoomed) {
                zoomToFit();
                hasAutoZoomed = true;
            }
        }
    }
    requestAnimationFrame(step);
}

function drawGraph() {
    if (!ctx || !canvas) return;

    const isLight = document.documentElement.getAttribute('data-bs-theme') === 'light';
    const computedStyles = getComputedStyle(document.documentElement);
    const colorPrimary = computedStyles.getPropertyValue('--primary').trim() || '#38bdf8';
    const colorWarning = computedStyles.getPropertyValue('--accent-warning').trim() || '#fbbf24';
    const colorPink = computedStyles.getPropertyValue('--color-pink-primary').trim() || '#ec4899';
    const colorSuccess = computedStyles.getPropertyValue('--secondary').trim() || '#34d399';
    const colorPurple = computedStyles.getPropertyValue('--color-purple-primary').trim() || '#a855f7';

    // Clear & Draw Celestial Deep Canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Background Galaxy Grid / Subtle Starlight Dust
    if (!isLight) {
        ctx.save();
        ctx.fillStyle = 'rgba(255, 255, 255, 0.03)';
        for (let i = 0; i < 30; i++) {
            const sx = (Math.sin(i * 99 + 1) * 0.5 + 0.5) * canvas.width;
            const sy = (Math.cos(i * 33 + 1) * 0.5 + 0.5) * canvas.height;
            ctx.beginPath();
            ctx.arc(sx, sy, 1.2, 0, Math.PI * 2);
            ctx.fill();
        }
        ctx.restore();
    }

    ctx.save();
    ctx.translate(canvas.width / 2 + transformX, canvas.height / 2 + transformY);
    ctx.scale(scale, scale);

    // Filter active nodes based on category selection
    const isNodeVisible = (n) => {
        if (currentGraphCategory === 'all') return true;
        return n.group === currentGraphCategory;
    };

    // 1. Draw Galaxy Constellation Links with Gradient Glow
    links.forEach(l => {
        let source = nodes.find(n => n.id === l.source);
        let target = nodes.find(n => n.id === l.target);
        if (source && target && isNodeVisible(source) && isNodeVisible(target)) {
            const isHoverLink = (hoveredNode && (hoveredNode.id === source.id || hoveredNode.id === target.id));

            ctx.save();
            ctx.beginPath();
            ctx.moveTo(source.x, source.y);
            ctx.lineTo(target.x, target.y);

            if (isHoverLink) {
                ctx.strokeStyle = colorPrimary;
                ctx.lineWidth = 2.5;
                ctx.shadowColor = colorPrimary;
                ctx.shadowBlur = 8;
            } else {
                ctx.strokeStyle = isLight ? 'rgba(15, 23, 42, 0.12)' : 'rgba(255, 255, 255, 0.12)';
                ctx.lineWidth = 1.2;
            }
            ctx.stroke();
            ctx.restore();
        }
    });

    // 2. Draw Stellar Nodes with Radial Outer Halos
    nodes.forEach(n => {
        if (!isNodeVisible(n)) return;

        let size = 7;
        let color = colorPrimary;
        if (n.group === 'source') { size = 13; color = colorWarning; }
        else if (n.group === 'grammar') { size = 8; color = colorPink; }
        else if (n.group === 'summaries') { size = 9; color = colorSuccess; }
        else if (n.group === 'concepts') { size = 6; color = colorPurple; }

        const isHovered = (n === hoveredNode);
        if (isHovered) size += 3;

        ctx.save();

        // Glow Halo
        ctx.shadowColor = color;
        ctx.shadowBlur = isHovered ? 20 : (n.group === 'source' ? 14 : 8);

        // Orbit Ring for Source Stars
        if (n.group === 'source') {
            ctx.beginPath();
            ctx.arc(n.x, n.y, size + 6, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(251, 191, 36, 0.35)';
            ctx.lineWidth = 1;
            ctx.stroke();
        }

        // Core Node Circle
        ctx.beginPath();
        ctx.arc(n.x, n.y, size, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();

        // White Inner Highlight
        ctx.beginPath();
        ctx.arc(n.x - size * 0.3, n.y - size * 0.3, size * 0.35, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.45)';
        ctx.fill();

        // Hover ring
        if (isHovered) {
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
        ctx.restore();

        // Node Label Typography
        ctx.save();
        ctx.fillStyle = isLight ? '#0f172a' : '#f8fafc';
        ctx.font = isHovered ? '600 11px Outfit' : (n.group === 'source' ? '600 10px Outfit' : '500 8.5px Outfit');
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText(n.label, n.x, n.y + size + 5);
        ctx.restore();
    });

    ctx.restore();
}

function setupGraphHandlers() {
    canvas.addEventListener('mousedown', (e) => {
        const rect = canvas.getBoundingClientRect();
        const mouseX = (e.clientX - rect.left - canvas.width/2 - transformX) / scale;
        const mouseY = (e.clientY - rect.top - canvas.height/2 - transformY) / scale;

        dragNode = nodes.find(n => {
            let dist = Math.sqrt((n.x - mouseX)**2 + (n.y - mouseY)**2);
            return dist < 15;
        });

        if (!dragNode) {
            isDraggingCanvas = true;
            dragStart.x = e.clientX - transformX;
            dragStart.y = e.clientY - transformY;
        }
    });

    canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        const mouseX = (e.clientX - rect.left - canvas.width/2 - transformX) / scale;
        const mouseY = (e.clientY - rect.top - canvas.height/2 - transformY) / scale;

        hoveredNode = nodes.find(n => {
            let dist = Math.sqrt((n.x - mouseX)**2 + (n.y - mouseY)**2);
            return dist < 12;
        });

        if (dragNode) {
            dragNode.x = mouseX;
            dragNode.y = mouseY;
            drawGraph();
        } else if (isDraggingCanvas) {
            // Prevent graph from flying away by limiting pan coordinates
            const maxPan = 1000;
            transformX = Math.max(-maxPan, Math.min(maxPan, e.clientX - dragStart.x));
            transformY = Math.max(-maxPan, Math.min(maxPan, e.clientY - dragStart.y));
            drawGraph();
        } else {
            drawGraph();
        }
    });

    canvas.addEventListener('mouseup', () => {
        if (dragNode) {
            dragNode = null;
            runGraphSimulation();
        }
        isDraggingCanvas = false;
    });

    // Double-click handler to open beautiful side-drawer preview (Option 3)
    canvas.addEventListener('dblclick', () => {
        if (hoveredNode) {
            previewNode(hoveredNode);
        }
    });
}

function previewNode(n) {
    if (!n) return;
    const headerEl = document.getElementById('node-preview-header');
    const contentEl = document.getElementById('node-preview-content');
    
    let typeBadgeColor = 'bg-secondary';
    let typeLabel = n.group.toUpperCase();
    if (n.group === 'vocabulary') typeBadgeColor = 'bg-success-subtle text-success border border-success-subtle';
    else if (n.group === 'grammar') typeBadgeColor = 'bg-danger-subtle text-danger border border-danger-subtle';
    else if (n.group === 'summaries') typeBadgeColor = 'bg-info-subtle text-info border border-info-subtle';
    else if (n.group === 'concepts') typeBadgeColor = 'bg-purple-subtle text-purple border border-purple-subtle';

    // Determine matching file name to let user jump to workspace
    let workspaceBtnHtml = '';
    let baseName = '';
    if (n.group === 'source') {
        baseName = n.id.split(':').pop().replace(/\.[^/.]+$/, "");
    } else {
        baseName = n.id.split(':').pop().split('_')[0];
    }
    
    const currentRawFiles = (typeof rawFiles !== 'undefined' && Array.isArray(rawFiles)) ? rawFiles : ((typeof window !== 'undefined' && Array.isArray(window.rawFiles)) ? window.rawFiles : []);
    const rawFile = currentRawFiles.find(f => f.name.includes(baseName) || f.stem === baseName);
    if (rawFile) {
        workspaceBtnHtml = `<button class="btn btn-sm btn-outline-info rounded-pill py-1 px-2.5 font-monospace fw-semibold" style="font-size: 0.68rem;" onclick="bootstrap.Offcanvas.getInstance(document.getElementById('offcanvasNodePreview')).hide(); openUnitWorkspace('${esc(rawFile.name)}');"><i class="bi bi-folder2-open"></i> Open Workspace</button>`;
    }

    headerEl.innerHTML = `
        <div class="d-flex align-items-center justify-content-between mb-2">
            <span class="badge ${typeBadgeColor} rounded-pill font-monospace" style="font-size: 0.65rem; padding: 0.25rem 0.5rem;">${typeLabel}</span>
            ${workspaceBtnHtml}
        </div>
        <h4 class="fw-bold mb-1 text-on-surface">${n.label}</h4>
        <div class="text-muted small font-monospace text-truncate" style="font-size: 0.68rem;" title="${n.path || ''}">${n.path || ''}</div>
    `;

    const myOffcanvas = new bootstrap.Offcanvas(document.getElementById('offcanvasNodePreview'));
    myOffcanvas.show();

    if (n.group === 'source') {
        contentEl.innerHTML = '<div class="text-center py-4"><span class="spinner-border spinner-border-sm me-2"></span>Reading raw source...</div>';
        const filename = n.id.split(':').pop();
        const unitName = filename.replace(/\.[^/.]+$/, "");
        fetch(`/wiki/${unitName}/sources/${filename}`)
            .then(res => res.ok ? res.text() : 'Failed to load raw source content.')
            .then(text => {
                contentEl.innerHTML = `<pre class="p-3 rounded font-monospace" style="background: var(--logger-bg); border: 1px solid var(--card-border); font-size: 0.75rem; white-space: pre-wrap; word-break: break-all;">${text}</pre>`;
            })
            .catch(() => { contentEl.innerHTML = 'Unable to fetch source content.'; });
    } else if (n.path) {
        let cleanPath = n.path;
        if (cleanPath.startsWith('wiki/')) {
            cleanPath = cleanPath.substring(5);
        } else if (cleanPath.startsWith('/wiki/')) {
            cleanPath = cleanPath.substring(6);
        }
        const renderFn = typeof renderMarkdownNode === 'function' ? renderMarkdownNode : (typeof window !== 'undefined' && window.renderMarkdownNode ? window.renderMarkdownNode : null);
        if (renderFn) renderFn(`/wiki/${cleanPath}`, contentEl);
    } else {
        contentEl.innerHTML = '<p class="text-muted small">No direct preview is available for this virtual node.</p>';
    }
}

function zoomIn() { scale = Math.min(4.0, scale * 1.25); drawGraph(); }
function zoomOut() { scale = Math.max(0.25, scale / 1.25); drawGraph(); }
function zoomReset() { zoomToFit(); }

function zoomToFit() {
    if (!ensureCanvas()) return;
    if (!nodes || nodes.length === 0) return;
    
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    
    nodes.forEach(n => {
        if (n.x < minX) minX = n.x;
        if (n.x > maxX) maxX = n.x;
        if (n.y < minY) minY = n.y;
        if (n.y > maxY) maxY = n.y;
    });
    
    const graphW = maxX - minX;
    const graphH = maxY - minY;
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    
    const viewW = canvas.width;
    const viewH = canvas.height;
    
    const padding = 0.75;
    let newScale = 1;
    if (graphW > 0 && graphH > 0) {
        const scaleX = (viewW * padding) / graphW;
        const scaleY = (viewH * padding) / graphH;
        newScale = Math.min(scaleX, scaleY);
        newScale = Math.max(0.15, Math.min(2.0, newScale));
    }
    
    scale = newScale;
    transformX = -scale * centerX;
    transformY = -scale * centerY;
    
    drawGraph();
}


// Bindings for window global scope (backward compatibility with inline HTML onclick handlers)
if (typeof window !== 'undefined') {
    window.setGraphCategoryFilter = setGraphCategoryFilter;
    window.exportGraphImage = exportGraphImage;
    window.resizeCanvas = resizeCanvas;
    window.refreshGraph = refreshGraph;
    window.runGraphSimulation = runGraphSimulation;
    window.drawGraph = drawGraph;
    window.setupGraphHandlers = setupGraphHandlers;
    window.zoomIn = zoomIn;
    window.zoomOut = zoomOut;
    window.zoomReset = zoomReset;
    window.zoomToFit = zoomToFit;
}

export {
    ensureCanvas,
    setGraphCategoryFilter,
    exportGraphImage,
    resizeCanvas,
    refreshGraph,
    runGraphSimulation,
    drawGraph,
    setupGraphHandlers,
    zoomIn,
    zoomOut,
    zoomReset,
    zoomToFit
};
