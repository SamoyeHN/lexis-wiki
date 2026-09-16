    // Helper to safely escape string values for use in inline JS inside HTML attributes (e.g. onclick)
    function esc(str) {
        if (!str) return '';
        return str.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/"/g, '\\"');
    }

    // State Machine
    let appConfig = {};
    let rawFiles = [];
    let compiledUnits = {};
    let currentUnit = null; // Name of the active Raw File Focus Workspace
    let activeJobId = null;
    let activePollTimers = {};
    let activePreviewPath = null;

    // canvas variables for 2D network force graph
    let canvas = document.getElementById('wiki-canvas');
    let ctx = canvas.getContext('2d');
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

    // Workspace subtab control
    let activeSubTab = 'raw';

    // Filter/Sort Ribbon State
    let libraryFilter = 'all';
    let librarySortOption = 'recent';
    let libraryViewMode = 'grid';

    function setLibraryFilter(filter) {
        libraryFilter = filter;
        document.querySelectorAll('.filter-ribbon-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.getElementById(`filter-${filter}`).classList.add('active');
        fetchRawFiles();
    }

    function onLibrarySort() {
        librarySortOption = document.getElementById('library-sort').value;
        fetchRawFiles();
    }

    function setViewMode(mode) {
        libraryViewMode = mode;
        const gridBtn = document.getElementById('view-mode-grid');
        const listBtn = document.getElementById('view-mode-list');
        if (mode === 'grid') {
            gridBtn.classList.add('text-info');
            gridBtn.classList.remove('text-muted');
            listBtn.classList.add('text-muted');
            listBtn.classList.remove('text-info');
        } else {
            listBtn.classList.add('text-info');
            listBtn.classList.remove('text-muted');
            gridBtn.classList.add('text-muted');
            gridBtn.classList.remove('text-info');
        }
        fetchRawFiles();
    }

    // ------------------ INITIAL LIFE-CYCLE ------------------
    window.addEventListener('load', () => {
        fetchConfig();
        fetchCompiledUnits();
        setupGraphHandlers();
        
        // Stop playing video/audio when handout preview drawer is hidden
        const offcanvasEl = document.getElementById('offcanvasHandoutPreview');
        if (offcanvasEl) {
            offcanvasEl.addEventListener('hidden.bs.offcanvas', () => {
                const previewFrame = document.getElementById('quiz-preview-iframe-offcanvas');
                if (previewFrame) {
                    previewFrame.src = '';
                }
            });
        }

        // Polling background tasks
        setInterval(fetchRecentJobs, 3000);
        fetchRecentJobs();

        setTimeout(() => {
            syncHeaderHeight();
            resizeCanvas();
            refreshGraph();
        }, 120);
    });

    // Handle viewport resize for Force Graph & Dynamic Header Height
    window.addEventListener('resize', () => {
        syncHeaderHeight();
        resizeCanvas();
        drawGraph();
    });

    // ------------------ WORKSPACE STATE CONTROL ------------------
    function syncHeaderHeight() {
        const header = document.querySelector('header.fixed-top');
        if (header) {
            const h = header.offsetHeight;
            if (h > 0) {
                document.documentElement.style.setProperty('--header-height', `${h}px`);
            }
        }
    }

    // Attach ResizeObserver to automatically adapt if header size shifts
    if (window.ResizeObserver) {
        const headerEl = document.querySelector('header.fixed-top');
        if (headerEl) {
            new ResizeObserver(() => syncHeaderHeight()).observe(headerEl);
        }
    }

    let savedLibraryScrollY = 0;

    function openUnitWorkspace(filename) {
        if (currentUnit !== filename) {
            activePreviewPath = null;
            selectedCompileCategory = null;
        }
        // Save current vertical scroll position of the library view before locking viewport
        savedLibraryScrollY = window.scrollY || document.documentElement.scrollTop || 0;

        currentUnit = filename;
        const file = rawFiles.find(f => f.name === filename);
        // Measure and sync dynamic header height
        syncHeaderHeight();
        document.body.classList.add('in-workspace');

        // Reset document window scroll immediately so top header and workspace dock cleanly at (0, 0)
        window.scrollTo(0, 0);
        document.documentElement.scrollTop = 0;
        document.body.scrollTop = 0;

        // Sync visual UI crumbs & layout
        document.getElementById('view-library').classList.add('d-none');
        document.getElementById('view-workspace').classList.remove('d-none');

        document.getElementById('header-brand-normal').classList.replace('d-flex', 'd-none');
        document.getElementById('header-brand-workspace').classList.replace('d-none', 'd-flex');

        const displayName = file ? (file.title || file.stem || filename.replace(/\.[^/.]+$/, "")) : filename.replace(/\.[^/.]+$/, "");
        document.getElementById('breadcrumb-unit-name').innerText = displayName.replace(/_/g, ' ');
        document.getElementById('workspace-unit-title').innerText = displayName.replace(/_/g, ' ');
        document.getElementById('workspace-unit-size').innerText = file ? `${(file.size / 1024).toFixed(1)} KB` : '...';

        // Fill Workspace Raw File Text previewer accordion via loadWorkspaceSourceDoc
        const unitName = filename.replace(/\.[^/.]+$/, "");
        loadWorkspaceSourceDoc(`/wiki/${unitName}/sources/${filename}`, displayName);

        // Fetch fresh compiled + raw data then update viewport (avoids stale compiledUnits state)
        fetchCompiledUnits();
    }

    let activeSourceDocPath = null;

    function loadWorkspaceSourceDoc(docUrl, label) {
        activeSourceDocPath = docUrl;
        const rawContentEl = document.getElementById('workspace-raw-file-content');
        if (!rawContentEl) return;
        rawContentEl.textContent = 'Loading source content...';

        fetch(docUrl)
            .then(res => res.ok ? res.text() : 'No source preview is stored recursively inside sources/ folder.')
            .then(text => { 
                if (activeSourceDocPath === docUrl) {
                    rawContentEl.textContent = text; 
                }
            })
            .catch(() => { 
                if (activeSourceDocPath === docUrl) {
                    rawContentEl.textContent = 'Unable to fetch source markdown content.'; 
                }
            });

        // Update active class on source document pills
        document.querySelectorAll('.source-doc-pill').forEach(pill => {
            if (pill.getAttribute('data-url') === docUrl) {
                pill.classList.add('active');
                pill.style.background = 'var(--primary)';
                pill.style.borderColor = 'var(--primary)';
                pill.style.color = '#fff';
            } else {
                pill.classList.remove('active');
                pill.style.background = 'rgba(255, 255, 255, 0.05)';
                pill.style.borderColor = 'var(--card-border)';
                pill.style.color = 'var(--on-surface-variant)';
            }
        });
    }

    function closeUnitWorkspace() {
        currentUnit = null;
        activePreviewPath = null;
        document.body.classList.remove('in-workspace');
        document.getElementById('view-workspace').classList.add('d-none');
        document.getElementById('view-library').classList.remove('d-none');
        
        document.getElementById('header-brand-workspace').classList.replace('d-flex', 'd-none');
        document.getElementById('header-brand-normal').classList.replace('d-none', 'd-flex');

        // Restore previously saved library scroll position gracefully
        window.scrollTo(0, savedLibraryScrollY);
        
        // Reset subtab views
        switchWorkspaceSubTab('raw');
        fetchCompiledUnits();
    }

    function switchWorkspaceSubTab(subtab) {
        // Handle handout subtab redirect gracefully since it's now in the right column and offcanvas preview
        if (subtab === 'handout') {
            const generatorCard = document.querySelector('.btn-quiz-arch')?.closest('.bento-card');
            if (generatorCard) {
                generatorCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
                generatorCard.style.outline = '2px solid var(--primary)';
                setTimeout(() => { generatorCard.style.outline = 'none'; }, 1500);
            }
            return;
        }

        activeSubTab = subtab;
        
        // Update nav-btns
        document.querySelectorAll('.workspace-tab-btn').forEach(btn => btn.classList.remove('active'));
        const tabBtn = document.getElementById(`tab-btn-${subtab}`);
        if (tabBtn) tabBtn.classList.add('active');

        // Toggle contents panels
        document.querySelectorAll('.workspace-subtab-content').forEach(p => p.classList.add('d-none'));
        const contentPanel = document.getElementById(`subtab-${subtab}`);
        if (contentPanel) contentPanel.classList.remove('d-none');

        // Prevent container scrolling when using mindmap pan/zoom
        const container = document.getElementById('workspace-viewport-container');
        if (container) {
            if (subtab === 'mindmap') {
                container.style.overflowY = 'hidden';
            } else {
                container.style.overflowY = 'auto';
            }
        }
    }

    function updateWorkspaceViewport(filename) {
        const file = rawFiles.find(f => f.name === filename);
        const baseName = file ? (file.stem || filename.replace(/\.[^/.]+$/, "")) : filename.replace(/\.[^/.]+$/, "");
        
        // ------------------ TRANSCRIBING STATE OVERRIDE ------------------
        const isTranscribing = file && file.is_transcribing;
        const tabsEl = document.querySelector('.workspace-tabs');
        const transcribingPanel = document.getElementById('subtab-transcribing');
        
        if (isTranscribing) {
            if (tabsEl) tabsEl.classList.add('d-none');
            if (transcribingPanel) {
                transcribingPanel.classList.remove('d-none');
                
                // Find matching video import job
                const matchedJob = recentJobs.find(j => 
                    j.type === 'video-import' && 
                    (j.unit_name === baseName || j.unit_name === filename || j.target.includes(baseName) || j.target.includes(filename))
                );
                
                const progressNum = document.getElementById('transcribe-progress-num');
                const logsEl = document.getElementById('transcribe-logs');
                const statusBadge = document.getElementById('transcribe-status-badge');
                
                if (matchedJob) {
                    if (progressNum) progressNum.innerText = `${matchedJob.progress}%`;
                    if (statusBadge) {
                        statusBadge.innerText = matchedJob.status.toUpperCase();
                        if (matchedJob.status === 'completed') {
                            statusBadge.className = 'badge bg-success-subtle text-success font-monospace';
                        } else if (matchedJob.status === 'failed') {
                            statusBadge.className = 'badge bg-danger-subtle text-danger font-monospace';
                        } else {
                            statusBadge.className = 'badge bg-info-subtle text-info font-monospace';
                        }
                    }
                    if (logsEl) {
                        logsEl.innerHTML = '';
                        matchedJob.logs.forEach(l => {
                            let color = '#38bdf8';
                            if (l.includes('[ERROR]') || l.includes('failed')) color = '#ff8080';
                            if (l.includes('[WARNING]')) color = '#ffcc80';
                            if (l.includes('Success!') || l.includes('completed successfully')) color = '#80ff80';
                            logsEl.innerHTML += `<div style="color: ${color}; margin-bottom: 2px;">${l}</div>`;
                        });
                        logsEl.scrollTop = logsEl.scrollHeight;
                    }
                } else {
                    // Fallback if job is not in the recent list yet or was completed/evicted
                    if (progressNum) progressNum.innerText = '⏳';
                    if (logsEl) logsEl.innerHTML = '<div class="text-muted">Task queued. Waiting for Whisper model...</div>';
                    if (statusBadge) {
                        statusBadge.innerText = 'PENDING';
                        statusBadge.className = 'badge bg-warning-subtle text-warning font-monospace';
                    }
                }
            }
            
            // Hide all other workspace content panels
            document.querySelectorAll('.workspace-subtab-content').forEach(p => {
                if (p.id !== 'subtab-transcribing') {
                    p.classList.add('d-none');
                }
            });
            
            // Prevent compiling while transcribing
            const compileBtn = document.getElementById('workspace-compile-btn');
            if (compileBtn) {
                compileBtn.disabled = true;
                compileBtn.innerHTML = `<span class="spinner-border spinner-border-sm me-1" style="width: 0.8rem; height: 0.8rem;"></span> TRANSCRIBING...`;
            }
            return; // Exit early to bypass normal panel content syncs
        } else {
            if (tabsEl) tabsEl.classList.remove('d-none');
            if (transcribingPanel) transcribingPanel.classList.add('d-none');
            
            // Restore compile button via unified state manager
            updateCompileButtonUI();
        }
        // -----------------------------------------------------------------
        
        // Vocab, grammar and concept summary path lookups
        const vocabName = `${baseName}_vocabulary.md`;
        const grammarName = `${baseName}_grammar.md`;
        const summaryName = `${baseName}_summary.md`;

        const vocabPath = compiledUnits.vocabulary && compiledUnits.vocabulary.find(v => v.endsWith(vocabName) || v === vocabName);
        const grammarPath = compiledUnits.grammar && compiledUnits.grammar.find(g => g.endsWith(grammarName) || g === grammarName);
        
        const summariesList = compiledUnits.concepts_by_source && (compiledUnits.concepts_by_source[filename] || compiledUnits.concepts_by_source[file.stem] || compiledUnits.concepts_by_source[baseName]);
        const summaryPath = summariesList && summariesList.length > 0 ? summariesList[0] : null;

        const mindmapName = `${baseName}_mindmap.html`;
        const mindmapPath = compiledUnits.quizzes && compiledUnits.quizzes.find(q => q.toLowerCase().endsWith(mindmapName.toLowerCase()));

        // Dynamically toggle active classes on the Active Source target pills (V, G, S, M)
        const pillV = document.getElementById('ws-pill-v');
        const pillG = document.getElementById('ws-pill-g');
        const pillS = document.getElementById('ws-pill-s');
        const pillM = document.getElementById('ws-pill-m');

        if (pillV) {
            pillV.className = `unit-target-pill ${vocabPath ? 'vocab-done' : ''}`;
            pillV.title = `Vocabulary ${vocabPath ? '(Compiled)' : '(Uncompiled)'}`;
        }
        if (pillG) {
            pillG.className = `unit-target-pill ${grammarPath ? 'grammar-done' : ''}`;
            pillG.title = `Grammar ${grammarPath ? '(Compiled)' : '(Uncompiled)'}`;
        }
        if (pillS) {
            pillS.className = `unit-target-pill ${summaryPath ? 'summary-done' : ''}`;
            pillS.title = `Summary ${summaryPath ? '(Compiled)' : '(Uncompiled)'}`;
        }
        if (pillM) {
            pillM.className = `unit-target-pill ${mindmapPath ? 'mindmap-done' : ''}`;
            pillM.title = `Mind Map ${mindmapPath ? '(Compiled)' : '(Uncompiled)'}`;
        }

        // 1. Vocabulary pane sync
        const vocabEmpty = document.getElementById('subtab-vocab-empty');
        const vocabWrapper = document.getElementById('subtab-vocab-wrapper');
        const vocabViewer = document.getElementById('subtab-vocab-viewer');
        if (vocabPath) {
            vocabEmpty.classList.add('d-none');
            vocabWrapper.classList.remove('d-none');
            renderMarkdownNode(`/wiki/${vocabPath}`, vocabViewer);
        } else {
            vocabEmpty.classList.remove('d-none');
            vocabWrapper.classList.add('d-none');
        }

        // 2. Grammar pane sync
        const grammarEmpty = document.getElementById('subtab-grammar-empty');
        const grammarWrapper = document.getElementById('subtab-grammar-wrapper');
        const grammarViewer = document.getElementById('subtab-grammar-viewer');
        if (grammarPath) {
            grammarEmpty.classList.add('d-none');
            grammarWrapper.classList.remove('d-none');
            renderMarkdownNode(`/wiki/${grammarPath}`, grammarViewer);
        } else {
            grammarEmpty.classList.remove('d-none');
            grammarWrapper.classList.add('d-none');
        }

        // 3. Summary pane sync
        const summaryEmpty = document.getElementById('subtab-summary-empty');
        const summaryWrapper = document.getElementById('subtab-summary-wrapper');
        const summaryViewer = document.getElementById('subtab-summary-viewer');
        if (summaryPath) {
            summaryEmpty.classList.add('d-none');
            summaryWrapper.classList.remove('d-none');
            renderMarkdownNode(`/wiki/${summaryPath}`, summaryViewer);
        } else {
            summaryEmpty.classList.remove('d-none');
            summaryWrapper.classList.add('d-none');
        }

        // 3.5. Mind Map sync

        const mindmapEmpty = document.getElementById('subtab-mindmap-empty');
        const mindmapViewer = document.getElementById('subtab-mindmap-viewer');
        const mindmapFrame = document.getElementById('mindmap-frame');

        if (mindmapPath) {
            mindmapEmpty.classList.add('d-none');
            mindmapViewer.classList.remove('d-none');
            // Prevent redundant reloading if iframe src is already matching
            const targetSrc = `/${mindmapPath}`;
            if (mindmapFrame.getAttribute('src') !== targetSrc) {
                mindmapFrame.src = targetSrc;
            }
            // Ensure the container itself matches the light mindmap background
            const parentContainer = document.getElementById('subtab-mindmap');
            if (parentContainer) {
                parentContainer.style.background = '#f8fafc';
                parentContainer.style.borderRadius = '0.5rem';
            }
        } else {
            mindmapEmpty.classList.remove('d-none');
            mindmapViewer.classList.add('d-none');
            mindmapFrame.src = '';
            const parentContainer = document.getElementById('subtab-mindmap');
            if (parentContainer) {
                parentContainer.style.background = 'transparent';
            }
        }

        // 4. Local unit quizzes sync
        const quizListEl = document.getElementById('workspace-quizzes-list');
        quizListEl.innerHTML = '';
        
        // Filter quizzes belonging to this unit, excluding any mindmaps (HTML or JSON)
        const unitQuizzes = compiledUnits.quizzes && compiledUnits.quizzes.filter(q => {
            const qFile = q.split('/').pop().toLowerCase();
            const parts = q.replace(/\\/g, '/').split('/');
            const wikiIdx = parts.indexOf('wiki');
            const fileUnitName = (wikiIdx !== -1 && wikiIdx + 1 < parts.length) ? parts[wikiIdx + 1] : '';
            const matchesUnit = fileUnitName ? (fileUnitName.toLowerCase() === baseName.toLowerCase()) : qFile.startsWith(baseName.toLowerCase());
            return matchesUnit && !qFile.includes('mindmap');
        });

        document.getElementById('workspace-quizzes-count').innerText = unitQuizzes ? unitQuizzes.length : 0;

        if (!unitQuizzes || unitQuizzes.length === 0) {
            quizListEl.innerHTML = `<div class="text-muted small py-2">No quizzes compiled for this unit yet. Choose a quiz type above and click "Generate Quiz"!</div>`;
        } else {
            unitQuizzes.forEach(q => {
                const btn = document.createElement('div');
                btn.className = 'wiki-link-item mb-1';
                // Stylize beautifully based on quiz architectures
                let iconClass = 'bi-file-earmark-code';
                if (q.includes('listening')) iconClass = 'bi-headphones';
                if (q.includes('reading')) iconClass = 'bi-book';
                if (q.includes('video')) iconClass = 'bi-film';
                if (q.includes('translation')) iconClass = 'bi-translate';
                if (q.includes('vocabulary')) iconClass = 'bi-spellcheck';

                btn.innerHTML = `<i class="bi ${iconClass} me-2 text-info"></i><span class="text-truncate">${q.split('/').pop()}</span>`;
                btn.onclick = () => loadHandoutPreview(q);
                quizListEl.appendChild(btn);
            });
            // Automatically load first quiz preview in background ONLY if there isn't already an active preview loaded
            if (!activePreviewPath && unitQuizzes.length > 0) {
                loadHandoutPreview(unitQuizzes[0], false);
            }
        }

        // Populate dynamic source inputs inside hidden selectors to safeguard existing backend scripts
        const srcSelect = document.getElementById('quiz-source-select');
        if (srcSelect) {
            srcSelect.innerHTML = `<option value="${filename}" selected>${filename}</option>`;
        }

        // Option A (Canvas First): Render Source Document Switcher pills
        const switcherContainer = document.getElementById('workspace-source-switcher');
        const pillsContainer = document.getElementById('workspace-source-pills');
        if (switcherContainer && pillsContainer) {
            const unitSources = [];
            // 1. Primary Text Source
            const primaryUrl = `/wiki/${baseName}/sources/${filename}`;
            const primaryLabel = file ? (file.title || file.stem || filename) : filename;
            unitSources.push({
                url: primaryUrl,
                label: `Primary Text: ${primaryLabel}`,
                isPrimary: true,
                filename: filename
            });

            // 2. Transcripts from unit media
            if (compiledUnits.media) {
                compiledUnits.media.forEach(mPath => {
                    const cleanPath = mPath.replace(/\\/g, '/');
                    const parts = cleanPath.split('/');
                    const wikiIdx = parts.indexOf('wiki');
                    const mUnit = (wikiIdx !== -1 && wikiIdx + 1 < parts.length) ? parts[wikiIdx + 1] : '';
                    if (mUnit.toLowerCase() === baseName.toLowerCase() && cleanPath.endsWith('.md')) {
                        const mFileName = cleanPath.split('/').pop();
                        const mTitle = mFileName.replace(/\.[^/.]+$/, "").replace(/_/g, " ");
                        unitSources.push({
                            url: `/wiki/${mUnit}/sources/media/${mFileName}`,
                            label: `Transcript: ${mTitle}`,
                            isPrimary: false,
                            filename: mFileName
                        });
                    }
                });
            }

            // Also check supplemental files for companion markdown transcripts if any
            if (compiledUnits.supplemental) {
                compiledUnits.supplemental.forEach(sPath => {
                    const cleanPath = sPath.replace(/\\/g, '/');
                    const parts = cleanPath.split('/');
                    const wikiIdx = parts.indexOf('wiki');
                    const sUnit = (wikiIdx !== -1 && wikiIdx + 1 < parts.length) ? parts[wikiIdx + 1] : '';
                    if (sUnit.toLowerCase() === baseName.toLowerCase() && cleanPath.endsWith('.md')) {
                        const sFileName = cleanPath.split('/').pop();
                        const sTitle = sFileName.replace(/\.[^/.]+$/, "").replace(/_/g, " ");
                        const sUrl = cleanPath.startsWith('/') ? cleanPath : `/${cleanPath}`;
                        if (!unitSources.some(s => s.url === sUrl || s.filename === sFileName)) {
                            unitSources.push({
                                url: sUrl,
                                label: `Supplementary: ${sTitle}`,
                                isPrimary: false,
                                filename: sFileName
                            });
                        }
                    }
                });
            }

            if (unitSources.length > 1) {
                switcherContainer.classList.remove('d-none');
                pillsContainer.innerHTML = '';
                unitSources.forEach(src => {
                    const pillBtn = document.createElement('button');
                    pillBtn.type = 'button';
                    pillBtn.className = 'btn btn-sm source-doc-pill d-flex align-items-center gap-1.5 px-2.5 py-1 rounded-pill border';
                    pillBtn.setAttribute('data-url', src.url);
                    pillBtn.style.fontSize = '0.72rem';
                    pillBtn.style.cursor = 'pointer';
                    pillBtn.style.transition = 'all 0.15s ease';

                    const isCurrentActive = (activeSourceDocPath === src.url) || (!activeSourceDocPath && src.isPrimary);
                    if (isCurrentActive) {
                        pillBtn.classList.add('active');
                        pillBtn.style.background = 'var(--primary)';
                        pillBtn.style.borderColor = 'var(--primary)';
                        pillBtn.style.color = '#fff';
                    } else {
                        pillBtn.style.background = 'rgba(255, 255, 255, 0.05)';
                        pillBtn.style.borderColor = 'var(--card-border)';
                        pillBtn.style.color = 'var(--on-surface-variant)';
                    }

                    const icon = src.isPrimary ? 'bi-file-earmark-text' : 'bi-file-earmark-play';
                    pillBtn.innerHTML = `<i class="bi ${icon} me-1"></i><span class="text-truncate" style="max-width: 260px;" title="${esc(src.label)}">${esc(src.label)}</span>`;
                    pillBtn.onclick = () => {
                        loadWorkspaceSourceDoc(src.url, src.label);
                        switchWorkspaceSubTab('raw');
                    };
                    pillsContainer.appendChild(pillBtn);
                });
            } else {
                switcherContainer.classList.add('d-none');
            }
        }
    }

    function renderMarkdownNode(url, targetElement) {
        targetElement.innerHTML = '<div class="text-center py-4"><span class="spinner-border spinner-border-sm me-2"></span>Reading markdown...</div>';
        fetch(url)
            .then(res => res.ok ? res.text() : 'Failed to load markdown.')
            .then(mdContent => {
                let frontmatterHtml = '';
                let bodyContent = mdContent;
                
                const fmMatch = mdContent.match(/^---\r?\n([\s\S]*?)\r?\n---/);
                if (fmMatch) {
                    bodyContent = mdContent.slice(fmMatch[0].length);
                    frontmatterHtml = `<pre class="p-3 rounded mb-3 text-on-surface-variant" style="background: var(--logger-bg); border: 1px solid var(--card-border); font-size: 0.75rem; font-family: var(--font-mono); color: var(--logger-text);">${fmMatch[0]}</pre>`;
                }

                let lines = bodyContent.split(/\r?\n/);
                let processedLines = lines.map(line => {
                    let trimmed = line.trim();
                    if (trimmed.startsWith('# ')) {
                        return `<h1 class="h4 fw-bold border-bottom pb-2 mb-3 mt-2">${trimmed.substring(2)}</h1>`;
                    } else if (trimmed.startsWith('## ')) {
                        return `<h2 class="h5 fw-bold text-info mt-3 mb-2">${trimmed.substring(3)}</h2>`;
                    } else if (trimmed.startsWith('### ')) {
                        return `<h3 class="h6 fw-bold text-success mt-2 mb-2">${trimmed.substring(4)}</h3>`;
                    } else if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
                        return `<div class="d-flex align-items-start gap-1.5 mb-1 ps-2"><span class="text-info fw-bold lh-1 me-1">- </span><div class="small">${trimmed.substring(2)}</div></div>`;
                    } else if (trimmed === '') {
                        return '';
                    } else {
                        return `<p class="mb-2 text-on-surface-variant small" style="line-height: 1.55;">${line}</p>`;
                    }
                });
                let html = processedLines.join('\n');
                html = html
                    .replace(/\*\*(.*?)\*\*/g, '<strong class="fw-bold text-on-surface">$1</strong>')
                    .replace(/\*(.*?)\*/g, '<em class="fst-italic">$1</em>')
                    .replace(/`([^`]+)`/g, '<code class="font-monospace p-0.5 rounded text-danger bg-dark border" style="font-size: 0.75rem;">$1</code>')
                    .replace(/\[\[(.*?)\]\]/g, '<span class="text-info text-decoration-underline fw-semibold" style="cursor: pointer;">$1</span>');

                targetElement.innerHTML = `<div class="workspace-markdown-card">${frontmatterHtml}${html}</div>`;
            })
            .catch(err => {
                targetElement.innerHTML = `<div class="workspace-markdown-card"><span class="text-danger">Failed to parse document: ${err}</span></div>`;
            });
    }

    function loadHandoutPreview(filePath, showOffcanvas = true) {
        activePreviewPath = filePath;
        const previewFrame = document.getElementById('quiz-preview-iframe-offcanvas');
        const mdContainer = document.getElementById('markdown-preview-container');
        const title = document.getElementById('offcanvasHandoutPreviewLabel');
        const actions = document.getElementById('preview-actions-offcanvas');

        // Normalize leading slashes
        let cleanPath = filePath;
        if (cleanPath.startsWith('//')) {
            cleanPath = cleanPath.substring(1);
        } else if (!cleanPath.startsWith('/')) {
            cleanPath = '/' + cleanPath;
        }

        if (cleanPath.endsWith('.md')) {
            if (previewFrame) {
                previewFrame.classList.add('d-none');
                previewFrame.src = '';
            }
            if (mdContainer) {
                mdContainer.classList.remove('d-none');
                renderMarkdownNode(cleanPath, mdContainer);
            }
        } else {
            if (mdContainer) {
                mdContainer.classList.add('d-none');
                mdContainer.innerHTML = '';
            }
            if (previewFrame) {
                previewFrame.classList.remove('d-none');
                previewFrame.src = cleanPath;
            }
        }

        const iconEl = document.getElementById('offcanvasHandoutPreviewIcon');
        if (iconEl) {
            let iconClass = 'bi-file-earmark-text';
            const lower = cleanPath.toLowerCase();
            if (lower.includes('listening')) iconClass = 'bi-headphones';
            else if (lower.includes('reading')) iconClass = 'bi-book';
            else if (lower.includes('video')) iconClass = 'bi-film';
            else if (lower.includes('translation')) iconClass = 'bi-translate';
            else if (lower.includes('vocabulary')) iconClass = 'bi-spellcheck';
            else if (lower.endsWith('.md')) iconClass = 'bi-file-earmark-markdown';
            
            iconEl.className = `bi ${iconClass} text-info`;
        }

        if (title) {
            title.textContent = cleanPath.split('/').pop();
        }
        if (actions) {
            let buttonsHtml = `
                <a href="${cleanPath}" target="_blank" class="btn btn-sm btn-outline-primary py-1 px-2.5" style="font-size: 0.75rem;">
                    <i class="bi bi-box-arrow-up-right me-1"></i> Open Tab
                </a>
            `;

            if (cleanPath.endsWith('.html')) {
                const isVideoQuiz = cleanPath.toLowerCase().includes('_video_quiz') || cleanPath.toLowerCase().includes('_video_');
                if (isVideoQuiz) {
                    buttonsHtml += `
                        <button class="btn btn-sm btn-outline-warning py-1 px-2.5" style="font-size: 0.75rem;" onclick="bundleStandaloneExe('${esc(cleanPath)}')">
                            <i class="bi bi-file-earmark-zip me-1"></i> Bundle EXE
                        </button>
                    `;
                } else {
                    buttonsHtml += `
                        <a href="${cleanPath}" download="${cleanPath.split('/').pop()}" class="btn btn-sm btn-outline-success py-1 px-2.5" style="font-size: 0.75rem;">
                            <i class="bi bi-download me-1"></i> Save .HTML
                        </a>
                    `;
                }
            }

            actions.innerHTML = `
                <div class="d-flex gap-1.5 flex-wrap">
                    ${buttonsHtml}
                </div>
            `;
        }

        if (showOffcanvas) {
            const offcanvasEl = document.getElementById('offcanvasHandoutPreview');
            if (offcanvasEl) {
                const bsOffcanvas = bootstrap.Offcanvas.getOrCreateInstance(offcanvasEl);
                bsOffcanvas.show();
            }
        }
    }

    // ------------------ EXPORT & DOWNLOAD UTILITY FUNCTIONS ------------------
    function copyActiveNode(category) {
        if (!currentUnit) {
            showToast('No active unit selected.', 'error');
            return;
        }
        const baseName = currentUnit.replace(/\.[^/.]+$/, "");
        const filePath = `/wiki/${baseName}/extractions/${baseName}_${category}.md`;
        
        fetch(filePath)
            .then(res => {
                if (!res.ok) throw new Error('File not found');
                return res.text();
            })
            .then(text => {
                navigator.clipboard.writeText(text).then(() => {
                    showToast('Copied markdown to clipboard!', 'success');
                }).catch(err => {
                    showToast('Failed to copy: ' + err, 'error');
                });
            })
            .catch(err => {
                showToast('Unable to copy markdown file.', 'error');
            });
    }

    function getActiveSubTabCategory() {
        if (activeSubTab === 'vocab') return 'vocabulary';
        if (activeSubTab === 'grammar') return 'grammar';
        if (activeSubTab === 'summary') return 'summary';
        if (activeSubTab === 'raw') return 'raw';
        if (activeSubTab === 'mindmap') return 'mindmap';
        return activeSubTab;
    }

    function copyActiveWorkspaceTab() {
        const cat = getActiveSubTabCategory();
        if (cat === 'raw') {
            const sourceView = document.getElementById('workspace-raw-file-content');
            if (sourceView) {
                navigator.clipboard.writeText(sourceView.innerText || sourceView.textContent)
                    .then(() => showToast('Copied raw source content!', 'success'))
                    .catch(err => showToast('Failed to copy: ' + err, 'error'));
            } else {
                showToast('No raw source content to copy.', 'error');
            }
            return;
        }
        if (cat === 'mindmap') {
            copyMindmap();
            return;
        }
        copyActiveNode(cat);
    }

    function downloadActiveWorkspaceTab() {
        const cat = getActiveSubTabCategory();
        if (cat === 'raw') {
            downloadRawSource();
            return;
        }
        if (cat === 'mindmap') {
            downloadMindmap();
            return;
        }
        downloadActiveNode(cat);
    }

    function downloadRawSource() {
        if (!currentUnit) {
            showToast('No active unit selected.', 'error');
            return;
        }
        const baseName = currentUnit.replace(/\.[^/.]+$/, "");
        // Preserve the real extension so .md / .txt / .html sources all save with the correct type.
        const fileName = currentUnit;
        // The raw/Source file lives in the unit's sources/ directory (same path the viewer uses).
        const filePath = `/wiki/${baseName}/sources/${fileName}`;

        // Pick the correct MIME type from the real extension (no more hardcoded markdown).
        const lower = fileName.toLowerCase();
        let mime = 'text/plain;charset=utf-8';
        if (lower.endsWith('.md')) {
            mime = 'text/markdown;charset=utf-8';
        } else if (lower.endsWith('.html') || lower.endsWith('.htm')) {
            mime = 'text/html;charset=utf-8';
        }

        fetch(filePath)
            .then(res => {
                if (!res.ok) throw new Error('File not found');
                return res.text();
            })
            .then(text => {
                const blob = new Blob([text], { type: mime });
                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = fileName;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(a.href);
                showToast(`Saved ${fileName}`, 'success');
            })
            .catch(err => showToast(`Export failed: ${err.message}`, 'error'));
    }

    function copyMindmap() {
        if (!currentUnit) {
            showToast('No active unit selected.', 'error');
            return;
        }
        const baseName = currentUnit.replace(/\.[^/.]+$/, "");

        // Preferred: read the live data object injected into the Mind Map iframe
        // (same-origin). This yields the full tree even when branches are collapsed,
        // which body.innerText would miss.
        let data = null;
        try {
            const frame = document.getElementById('mindmap-frame');
            data = frame && frame.contentWindow && frame.contentWindow.mindmapData;
        } catch (e) {
            data = null;
        }

        const doCopy = (d) => {
            if (!d || !d.branches) {
                showToast('Could not read the Mind Map content.', 'error');
                return;
            }
            navigator.clipboard.writeText(serializeMindmap(d)).then(() => {
                showToast('Copied Mind Map content to clipboard!', 'success');
            }).catch(err => {
                showToast('Failed to copy Mind Map: ' + err, 'error');
            });
        };

        if (data) {
            doCopy(data);
            return;
        }

        // Fallback: fetch the standalone HTML and pull out the injected JSON.
        const htmlPath = `/wiki/${baseName}/extractions/${baseName}_mindmap.html`;
        fetch(htmlPath)
            .then(res => {
                if (!res.ok) throw new Error('Mind Map not found');
                return res.text();
            })
            .then(html => {
                const m = html.match(/const mindmapData\s*=\s*(\{[\s\S]*?\})\s*;/);
                if (!m) throw new Error('Mind Map data not found');
                doCopy(JSON.parse(m[1]));
            })
            .catch(err => showToast('Unable to copy Mind Map: ' + err.message, 'error'));
    }

    function serializeMindmap(d) {
        const lines = [];
        if (d.title) lines.push(d.title);
        if (d.overall_cefr_level) lines.push(`CEFR ${d.overall_cefr_level}`);
        if (lines.length) lines.push('');
        if (d.root_name) lines.push(d.root_name);
        (d.branches || []).forEach(branch => {
            lines.push(`• ${branch.branch_name}`);
            (branch.sub_branches || []).forEach(sub => {
                const name = (sub.sub_branch_name && sub.sub_branch_name.trim()) ? sub.sub_branch_name.trim() : '';
                const leaves = sub.leaves || [];
                if (name) {
                    lines.push(`  - ${name}`);
                    leaves.forEach(leaf => lines.push(`    • ${leaf}`));
                } else {
                    leaves.forEach(leaf => lines.push(`  • ${leaf}`));
                }
            });
            (branch.leaves || []).forEach(leaf => lines.push(`  • ${leaf}`));
        });
        return lines.join('\n');
    }

    function downloadMindmap() {
        if (!currentUnit) {
            showToast('No active unit selected.', 'error');
            return;
        }
        const baseName = currentUnit.replace(/\.[^/.]+$/, "");
        // The Mind Map is a self-contained standalone HTML: the JSON data is injected
        // into it at compile time (see _render_mindmap), so the .html is the full artifact.
        const filePath = `/wiki/${baseName}/extractions/${baseName}_mindmap.html`;
        const dlName = `${baseName}_mindmap.html`;

        fetch(filePath)
            .then(res => {
                if (!res.ok) throw new Error('Mind Map HTML not found');
                return res.text();
            })
            .then(html => {
                const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = dlName;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(a.href);
                showToast(`Saved ${dlName}`, 'success');
            })
            .catch(err => showToast(`Export failed: ${err.message}`, 'error'));
    }

    function printActiveWorkspaceTab() {
        const cat = getActiveSubTabCategory();
        if (cat === 'vocab') printActiveNode('subtab-vocab-viewer');
        else if (cat === 'grammar') printActiveNode('subtab-grammar-viewer');
        else if (cat === 'summary') printActiveNode('subtab-summary-viewer');
        else if (cat === 'raw') printActiveNode('workspace-raw-file-content');
        else showToast('Printing is not supported for this view.', 'info');
    }

    function downloadActiveNode(category) {
        if (!currentUnit) {
            showToast('No active unit selected.', 'error');
            return;
        }
        const baseName = currentUnit.replace(/\.[^/.]+$/, "");
        const filePath = `/wiki/${baseName}/extractions/${baseName}_${category}.md`;
        
        fetch(filePath)
            .then(res => {
                if (!res.ok) throw new Error('File not found');
                return res.text();
            })
            .then(text => {
                const blob = new Blob([text], { type: 'text/markdown;charset=utf-8' });
                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = `${baseName}_${category}.md`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(a.href);
                showToast(`Downloaded ${baseName}_${category}.md`, 'success');
            })
            .catch(err => showToast(`Export failed: ${err.message}`, 'error'));
    }

    function printActiveNode(containerId) {
        const el = document.getElementById(containerId);
        if (!el) return;
        const printWin = window.open('', '_blank');
        printWin.document.write(`
            <!DOCTYPE html>
            <html>
            <head>
                <title>Lexis Wiki Printable Export</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
                <style>
                    body { font-family: sans-serif; padding: 2rem; color: #000; background: #fff; }
                    pre { background: #f8fafc; padding: 1rem; border-radius: 4px; border: 1px solid #e2e8f0; }
                </style>
            </head>
            <body>
                <div>${el.innerHTML}</div>
                <script>
                    window.onload = function() { window.print(); window.close(); }
                <\/script>
            </body>
            </html>
        `);
        printWin.document.close();
    }

    function bundleStandaloneExe(htmlPath) {
        showToast('Building standalone EXE bundle...', 'info');
        fetch('/api/export-exe', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ html_path: htmlPath })
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                showToast(data.message || 'Standalone EXE generated successfully!', 'success');
                if (data.download_url) {
                    const a = document.createElement('a');
                    a.href = data.download_url;
                    a.download = data.download_url.split('/').pop();
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                }
            } else {
                showToast(`Packaging failed: ${data.error}`, 'error');
            }
        })
        .catch(err => showToast(`Request failed: ${err.message}`, 'error'));
    }

    // Global tracking of currently selected compilation target
    let selectedCompileCategory = null; // null = Compile All
    let isCompiling = false;

    function updateCompileButtonUI() {
        const btn = document.getElementById('workspace-compile-btn');
        if (btn) {
            if (isCompiling) {
                btn.disabled = true;
                btn.innerHTML = `<span class="spinner-border spinner-border-sm me-1" style="width: 0.8rem; height: 0.8rem;"></span> COMPILING...`;
            } else {
                btn.disabled = false;
                if (!selectedCompileCategory) {
                    btn.innerHTML = `<i class="bi bi-play-fill me-1"></i> COMPILE ALL`;
                } else {
                    btn.innerHTML = `<i class="bi bi-play-fill me-1"></i> COMPILE ${selectedCompileCategory.toUpperCase()}`;
                }
            }
        }
        // Update active class in dropdown menu items
        document.querySelectorAll('.compile-category-item').forEach(el => {
            const cat = el.getAttribute('data-category') || null;
            if ((!cat && !selectedCompileCategory) || (cat && cat === selectedCompileCategory)) {
                el.classList.add('active');
            } else {
                el.classList.remove('active');
            }
        });
    }

    function selectCompileCategory(category) {
        selectedCompileCategory = category ? category.trim() : null;
        updateCompileButtonUI();
    }

    function triggerWorkspaceCompileDirect() {
        // Triggers the active selection remembered by the button
        triggerWorkspaceCompile(selectedCompileCategory);
    }

    function triggerWorkspaceCompile(category) {
        if (!currentUnit) return;

        if (isCompiling) {
            showInfoModal("Job in Progress", "A compilation job is already running. Please wait for the current build to finish before starting a new build.");
            return;
        }

        isCompiling = true;
        updateCompileButtonUI();

        // Expand logs panel automatically
        const colLog = document.getElementById('collapseWorkspaceTerminal');
        if (colLog && !colLog.classList.contains('show')) {
            new bootstrap.Collapse(colLog, { show: true });
        }

        const term = document.getElementById('workspace-compiler-terminal');
        const catTarget = (category !== undefined) ? category : selectedCompileCategory;
        term.innerHTML = `<div class="log-entry text-info">🔨 Building ${catTarget ? catTarget : 'everything'} for '${currentUnit}'...</div>`;

        const payload = { filename: currentUnit };
        if (catTarget) {
            payload.categories = [catTarget];
        }

        fetch('/api/compile', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                activeJobId = data.job_id;
                selectedCompileJobId = data.job_id;
                pollJobStatus(data.job_id, 'compiler');
                fetchRecentJobs();
            } else {
                term.innerHTML += `<div class="log-entry text-danger">[ERROR] Compilation dispatch failed: ${data.error}</div>`;
                isCompiling = false;
                updateCompileButtonUI();
            }
        })
        .catch(err => {
            if (term) term.innerHTML += `<div class="log-entry text-danger">[ERROR] Dispatch failed: ${err.message}</div>`;
            isCompiling = false;
            updateCompileButtonUI();
        });
    }

    // ------------------ SETTINGS TAB SWITCHING ------------------
    function switchSettingsTab(tabId) {
        document.querySelectorAll('.settings-tab-btn').forEach(btn => btn.classList.remove('active'));
        const tabBtn = document.querySelector(`.settings-tab-btn[data-tab="${tabId}"]`);
        if (tabBtn) tabBtn.classList.add('active');

        document.querySelectorAll('.settings-tab-content').forEach(tab => tab.classList.remove('active'));
        const contentTab = document.getElementById(tabId);
        if (contentTab) contentTab.classList.add('active');
    }

    // ------------------ REUSABLE BOOTSTRAP 5 SYSTEM MODAL ------------------
    let appSystemModalResolver = null;

    function showSystemModal({ title = 'Notice', message = '', icon = 'bi-info-circle-fill', iconColor = 'text-info', iconBg = 'rgba(14, 165, 233, 0.12)', confirmText = 'OK', cancelText = null, isConfirm = false }) {
        return new Promise((resolve) => {
            appSystemModalResolver = resolve;
            
            document.getElementById('appSystemModalLabel').innerText = title;
            document.getElementById('appSystemModalBody').innerHTML = message;
            
            const iconEl = document.getElementById('appSystemModalIcon');
            iconEl.className = `bi ${icon} ${iconColor} fs-5`;
            
            const iconBgEl = document.getElementById('appSystemModalIconBg');
            iconBgEl.style.background = iconBg;
            
            const cancelBtn = document.getElementById('appSystemModalCancelBtn');
            if (isConfirm || cancelText) {
                cancelBtn.classList.remove('d-none');
                cancelBtn.innerText = cancelText || 'Cancel';
            } else {
                cancelBtn.classList.add('d-none');
            }
            
            const confirmBtn = document.getElementById('appSystemModalConfirmBtn');
            confirmBtn.innerText = confirmText;
            if (iconColor.includes('danger')) {
                confirmBtn.className = 'btn btn-sm btn-danger px-3.5 py-1.5 rounded-pill font-monospace fw-semibold';
            } else if (iconColor.includes('warning')) {
                confirmBtn.className = 'btn btn-sm btn-warning text-dark px-3.5 py-1.5 rounded-pill font-monospace fw-semibold';
            } else {
                confirmBtn.className = 'btn btn-sm btn-primary px-3.5 py-1.5 rounded-pill font-monospace fw-semibold';
            }
            
            const modalEl = document.getElementById('appSystemModal');
            const modalInstance = bootstrap.Modal.getOrCreateInstance(modalEl);
            
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

    function showConfirmModal(title, message, confirmText = 'Overwrite') {
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

    function showInfoModal(title, message) {
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

    // ------------------ LEVEL-2 EXPERT RE-AUDIT (BATCH) ------------------
    let lastReAuditResult = null;

    function openReAuditModal() {
        const modal = document.getElementById('re-audit-modal');
        const inst = bootstrap.Modal.getOrCreateInstance(modal);
        if (lastReAuditResult) showReAuditSummary(lastReAuditResult);
        inst.show();
    }

    function setReAuditRunning(running) {
        const btn = document.getElementById('re-audit-run-btn');
        const spinner = document.getElementById('re-audit-run-spinner');
        const icon = document.getElementById('re-audit-run-icon');
        const label = document.getElementById('re-audit-run-label');
        if (btn) btn.disabled = running;
        if (spinner) spinner.classList.toggle('d-none', !running);
        if (icon) icon.classList.toggle('d-none', running);
        if (label) label.textContent = running ? 'RE-AUDITING…' : 'RE-AUDIT ALL';
    }

    function startReAuditAll() {
        if (!confirm('Re-audit ALL quiz handouts with the Level-2 expert judge?\n\nThis is read-only and only calls the judge model (the generator is not re-run).')) return;
        setReAuditRunning(true);
        showReAuditSummary(null, true);
        fetch('/api/re-audit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ workers: 2 })
        })
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                pollJobStatus(data.job_id, 're-audit');
                fetchRecentJobs();
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

    function showReAuditSummary(summary, running) {
        const stats = document.getElementById('re-audit-stats');
        const body = document.getElementById('re-audit-body');
        if (!summary) {
            stats.innerHTML = '';
            body.innerHTML = running
                ? '<tr><td colspan="7" class="text-center text-muted py-4"><span class="spinner-border spinner-border-sm me-2"></span>Re-auditing in the background… (watch the Recent Jobs panel).</td></tr>'
                : '<tr><td colspan="7" class="text-center text-muted py-4">No re-audit run yet. Click <strong>RE-AUDIT ALL</strong> to run the Level-2 expert judge over every quiz handout.</td></tr>';
            return;
        }
        lastReAuditResult = summary;

        const esc = (s) => String(s == null ? '' : s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
        const acc = (summary.avg_blind_solve_accuracy != null) ? Math.round(summary.avg_blind_solve_accuracy * 100) + '%' : '—';
        const statCell = (val, color, label) =>
            `<div class="col d-flex flex-column align-items-center border rounded-3 py-2" style="border-color: var(--card-border) !important;">
                <span class="fs-5 fw-bold">${val}</span><span class="small ${color}">${label}</span>
             </div>`;
        stats.innerHTML =
            statCell(summary.passed, 'text-success', 'PASSED') +
            statCell(summary.failed, 'text-danger', 'FAILED') +
            statCell(summary.errors, 'text-warning', 'ERRORS') +
            statCell(acc, 'text-info', 'AVG BLIND-SOLVE');

        const rows = (summary.results || []).map(r => {
            const accR = (r.blind_solve_accuracy != null) ? Math.round(r.blind_solve_accuracy * 100) + '%' : '—';
            const prev = (r.prev_passed == null) ? '<span class="text-muted">n/a</span>'
                : (r.prev_passed ? '<span class="badge bg-success-subtle text-success">PASS</span>' : '<span class="badge bg-danger-subtle text-danger">FAIL</span>');
            const now = r.error ? '<span class="badge bg-warning-subtle text-warning">ERROR</span>'
                : (r.passed ? '<span class="badge bg-success-subtle text-success">PASS</span>' : '<span class="badge bg-danger-subtle text-danger">FAIL</span>');
            let change;
            if (r.prev_passed == null) change = '<span class="text-muted">first audit</span>';
            else if (r.prev_passed && !r.passed) change = 'PASS → FAIL';
            else if (!r.prev_passed && r.passed) change = 'FAIL → PASS';
            else change = '<span class="text-muted">unchanged</span>';
            return `<tr>
                <td class="font-monospace small">${esc(r.unit)} <span class="badge bg-secondary-subtle text-secondary">${esc(r.template)}</span></td>
                <td class="text-center">${r.item_count == null ? '—' : r.item_count}</td>
                <td class="text-center fw-semibold">${accR}</td>
                <td class="text-center">${r.overall_quality_score == null ? '—' : r.overall_quality_score}</td>
                <td class="text-center">${prev}</td>
                <td class="text-center">${now}</td>
                <td class="small">${change}${r.error ? `<div class="text-warning mt-1">${esc(r.error)}</div>` : ''}</td>
            </tr>`;
        }).join('');
        body.innerHTML = rows || '<tr><td colspan="7" class="text-center text-muted py-3">No results.</td></tr>';
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
        if (label) label.textContent = loading ? 'EVALUATING…' : 'RE-EVALUATE LOGS';
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
            if (audit.post_cure_score != null && audit.pre_cure_score != null) {
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

    // ------------------ SETTINGS OVERLAY MODAL ------------------
    function openSettingsModal() {
        const modal = document.getElementById('settings-modal');
        const modalInstance = bootstrap.Modal.getOrCreateInstance(modal);
        modalInstance.show();
    }
    function closeSettingsModal() {
        const modal = document.getElementById('settings-modal');
        const modalInstance = bootstrap.Modal.getInstance(modal);
        if (modalInstance) modalInstance.hide();
    }
    function saveGlobalConfigurationAndClose() {
        // Collect model parameters
        const api_type = document.getElementById('config-api-type').value;
        const model = document.getElementById('config-active-model').value;
        const api_url = document.getElementById('config-api-url').value;
        const api_key = document.getElementById('config-api-key').value;

        const compile_defaults = {
            vocabulary: parseInt(document.getElementById('config-vocab-count').value),
            grammar: parseInt(document.getElementById('config-grammar-count').value),
            concepts: parseInt(document.getElementById('config-concept-count').value),
            max_parallel: parseInt(document.getElementById('config-max-p').value)
        };

        const quiz_defaults = {
            reading: parseInt(document.getElementById('config-quiz-reading').value) || 0,
            vocabulary: parseInt(document.getElementById('config-quiz-vocabulary').value) || 0,
            translation: parseInt(document.getElementById('config-quiz-translation').value) || 0,
            listening: parseInt(document.getElementById('config-quiz-listening').value) || 0,
            video: parseInt(document.getElementById('config-quiz-video').value) || 0
        };

        const tts_engine = document.getElementById('config-tts-engine').value;
        const tts_url = document.getElementById('config-tts-url').value;
        const tts_voice_a = document.getElementById('config-tts-voice-a').value;
        const tts_voice_b = document.getElementById('config-tts-voice-b').value;
        const tts_model = document.getElementById('config-tts-model').value;
        const target_language = document.getElementById('config-target-language').value;
        const enable_expert_audit = document.getElementById('config-enable-expert-audit').checked;
        const judge_model = document.getElementById('config-judge-model').value.trim();

        fetch('/api/config/update', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                api_type, model, api_url, api_key,
                compile_defaults, quiz_defaults,
                tts_engine, tts_url, tts_voice_a, tts_voice_b, tts_model, target_language,
                enable_expert_audit, judge_model
            })
        })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                showToast(data.error, 'error');
            } else {
                showToast(data.message || 'Configuration saved successfully!', 'success');
                closeSettingsModal();
                fetchConfig();
            }
        });
    }

    // ------------------ ORIGINAL CORE API PORT INTEGRATION ------------------
    function fetchConfig() {
        fetch('/api/config')
            .then(res => res.json())
            .then(data => {
                appConfig = data.config || {};
                
                document.getElementById('headbar-model-name').innerText = appConfig.model || 'No model';

                // Synchronize modal options
                document.getElementById('config-api-type').value = appConfig.api_type || 'ollama';
                document.getElementById('config-api-url').value = appConfig.api_url || '';
                document.getElementById('config-api-key').value = appConfig.api_key || '';

                const cDefaults = appConfig.compile_defaults || {};
                const vCount = cDefaults.vocabulary || 20;
                const gCount = cDefaults.grammar || 5;
                const cCount = cDefaults.concepts || 3;
                const mParallel = cDefaults.max_parallel || 3;

                document.getElementById('config-vocab-count').value = vCount;
                document.getElementById('config-vocab-count-val').innerText = vCount;
                document.getElementById('config-grammar-count').value = gCount;
                document.getElementById('config-grammar-count-val').innerText = gCount;
                document.getElementById('config-concept-count').value = cCount;
                document.getElementById('config-concept-count-val').innerText = cCount;
                document.getElementById('config-max-p').value = mParallel;
                document.getElementById('config-max-p-val').innerText = mParallel;

                const qDefaults = appConfig.quiz_defaults || {};
                const qVocab = qDefaults.vocabulary || 10;
                const qReading = qDefaults.reading || 5;
                const qTranslation = qDefaults.translation || 5;
                const qListening = qDefaults.listening || 5;
                const qVideo = qDefaults.video ?? 5;

                document.getElementById('config-quiz-vocabulary').value = qVocab;
                document.getElementById('config-quiz-vocabulary-val').innerText = qVocab;
                document.getElementById('config-quiz-reading').value = qReading;
                document.getElementById('config-quiz-reading-val').innerText = qReading;
                document.getElementById('config-quiz-translation').value = qTranslation;
                document.getElementById('config-quiz-translation-val').innerText = qTranslation;
                document.getElementById('config-quiz-listening').value = qListening;
                document.getElementById('config-quiz-listening-val').innerText = qListening;
                document.getElementById('config-quiz-video').value = qVideo;
                document.getElementById('config-quiz-video-val').innerText = qVideo;

                document.getElementById('config-tts-engine').value = appConfig.tts_engine || 'kokoro';
                document.getElementById('config-tts-url').value = appConfig.tts_url || '';
                document.getElementById('config-tts-model').value = appConfig.tts_model || 'tts-1';
                document.getElementById('config-target-language').value = appConfig.target_language || 'Chinese';

                // Level 2 Expert Audit options
                document.getElementById('config-enable-expert-audit').checked = !!appConfig.enable_expert_audit;
                document.getElementById('config-judge-model').value = appConfig.judge_model || '';


                // Load API type first (sets up model field visibility)
                onApiTypeChange(false);

                // Populate voice dropdowns based on engine, do NOT auto-fill URL — we already loaded saved config above
                onTTSEngineChange(appConfig.tts_voice_a, appConfig.tts_voice_b, false);

                // Fetch models AFTER all fields are set so active model can be highlighted correctly
                fetchModels();
                selectQuizArch('vocabulary');
            })
            .catch(() => { document.getElementById('headbar-model-name').innerText = 'Config unavailable'; });
    }

    let endpointInputTimer = null;
    function onEndpointInput() {
        if (endpointInputTimer) clearTimeout(endpointInputTimer);
        endpointInputTimer = setTimeout(() => {
            fetchModels(true);
        }, 500);
    }

    function fetchModels(userTriggered = false) {
        // Send the engine/endpoint SELECTED in the dropdown so the backend lists models
        // for THAT engine (not just the saved one). This is what lets the list update live
        // when the user switches API Engine or modifies URL/Token before saving.
        const apiType = document.getElementById('config-api-type').value;
        const apiUrl = (document.getElementById('config-api-url') || {}).value || '';
        const apiKey = (document.getElementById('config-api-key') || {}).value || '';
        const params = new URLSearchParams();
        params.set('api_type', apiType);
        if (apiUrl) params.set('api_url', apiUrl);
        if (apiKey) params.set('api_key', apiKey);

        const modelSelect = document.getElementById('config-active-model');
        const refreshIcon = document.getElementById('model-refresh-icon');
        if (refreshIcon) refreshIcon.classList.add('spin-icon');

        // Check if the current form endpoint matches the saved server configuration
        const savedApiType = (appConfig && appConfig.api_type) || '';
        const savedApiUrl = (appConfig && appConfig.api_url) || '';
        const savedApiKey = (appConfig && appConfig.api_key) || '';
        const endpointChanged = (savedApiType !== apiType || savedApiUrl !== apiUrl || savedApiKey !== apiKey);

        fetch('/api/models?' + params.toString())
            .then(res => res.json())
            .then(data => {
                if (refreshIcon) refreshIcon.classList.remove('spin-icon');

                // If endpoint changed or user triggered a switch, previously-saved model belongs
                // to a different endpoint and must NOT linger in the Active Model list.
                const savedActive = (appConfig && appConfig.model) || '';
                const activeModel = endpointChanged ? '' : savedActive;
                const offline = !!(data && data.offline);

                modelSelect.innerHTML = '';

                // ENGINE OFFLINE: do NOT inject fake default model names.
                if (offline) {
                    const opt = document.createElement('option');
                    opt.value = '';
                    opt.selected = true;
                    opt.disabled = true;
                    opt.innerText = `(Offline - endpoint unreachable)`;
                    modelSelect.appendChild(opt);
                    document.getElementById('headbar-model-name').innerText = activeModel || 'LLM offline';
                    return;
                }

                // Engine reachable.
                let modelsList = data.models || [];

                if (modelsList.length === 0) {
                    const opt = document.createElement('option');
                    opt.value = '';
                    opt.selected = true;
                    opt.disabled = true;
                    opt.innerText = `(No models found)`;
                    modelSelect.appendChild(opt);
                    document.getElementById('headbar-model-name').innerText = activeModel || 'No models';
                    return;
                }

                // If saved active model exists on this endpoint, preserve it.
                // Otherwise default to the first available model from this endpoint.
                const targetSelected = (activeModel && modelsList.includes(activeModel)) ? activeModel : modelsList[0];
                const judgeSelect = document.getElementById('config-judge-model');
                const savedJudgeModel = (appConfig && appConfig.judge_model) || '';
                if (judgeSelect) {
                    judgeSelect.innerHTML = '<option value="">(Same as Active Generation Model)</option>';
                }

                modelsList.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m;
                    opt.innerText = m;
                    if (m === targetSelected) {
                        opt.selected = true;
                    }
                    modelSelect.appendChild(opt);

                    if (judgeSelect) {
                        const jOpt = document.createElement('option');
                        jOpt.value = m;
                        jOpt.innerText = m;
                        if (m === savedJudgeModel) {
                            jOpt.selected = true;
                        }
                        judgeSelect.appendChild(jOpt);
                    }
                });

                document.getElementById('headbar-model-name').innerText = targetSelected || 'Ready';
            })
            .catch(() => {
                if (refreshIcon) refreshIcon.classList.remove('spin-icon');
                document.getElementById('headbar-model-name').innerText = 'LLM offline';
                modelSelect.innerHTML = '';
                const opt = document.createElement('option');
                opt.value = '';
                opt.selected = true;
                opt.disabled = true;
                opt.innerText = '(Endpoint unreachable)';
                modelSelect.appendChild(opt);
            });
    }

    function fetchRawFiles() {
        fetch('/api/raw-files')
            .then(res => res.json())
            .then(data => {
                rawFiles = data.files;
                document.getElementById('stat-raw-files-count').innerText = `${rawFiles.length} file${rawFiles.length === 1 ? '' : 's'}`;

                // Redraw Document Library Grid on Homepage Welcome screen
                const grid = document.getElementById('document-library-grid');
                const gridScrollTop = grid ? grid.scrollTop : 0;
                grid.innerHTML = '';

                // Apply search, filter, and sort on the local files list
                let filesToRender = [...rawFiles];

                // 2. Category Filter ("all", "compiled", "raw")
                if (libraryFilter === 'compiled') {
                    filesToRender = filesToRender.filter(f => {
                        const baseName = f.stem || f.name.replace(/\.[^/.]+$/, "");
                        const vocabName = `${baseName}_vocabulary.md`;
                        const grammarName = `${baseName}_grammar.md`;
                        const summariesList = compiledUnits.concepts_by_source && (compiledUnits.concepts_by_source[f.name] || compiledUnits.concepts_by_source[f.stem] || compiledUnits.concepts_by_source[baseName]);
                        const hasVocab = compiledUnits.vocabulary && compiledUnits.vocabulary.some(v => v.endsWith(vocabName) || v === vocabName);
                        const hasGrammar = compiledUnits.grammar && compiledUnits.grammar.some(g => g.endsWith(grammarName) || g === grammarName);
                        const hasSummary = summariesList && summariesList.length > 0;
                        return hasVocab || hasGrammar || hasSummary;
                    });
                } else if (libraryFilter === 'raw') {
                    filesToRender = filesToRender.filter(f => {
                        const baseName = f.stem || f.name.replace(/\.[^/.]+$/, "");
                        const vocabName = `${baseName}_vocabulary.md`;
                        const grammarName = `${baseName}_grammar.md`;
                        const summariesList = compiledUnits.concepts_by_source && (compiledUnits.concepts_by_source[f.name] || compiledUnits.concepts_by_source[f.stem] || compiledUnits.concepts_by_source[baseName]);
                        const hasVocab = compiledUnits.vocabulary && compiledUnits.vocabulary.some(v => v.endsWith(vocabName) || v === vocabName);
                        const hasGrammar = compiledUnits.grammar && compiledUnits.grammar.some(g => g.endsWith(grammarName) || g === grammarName);
                        const hasSummary = summariesList && summariesList.length > 0;
                        return !(hasVocab || hasGrammar || hasSummary);
                    });
                }

                // 3. Sorting ("recent", "name", "size")
                if (librarySortOption === 'recent') {
                    filesToRender.sort((a, b) => b.mtime - a.mtime);
                } else if (librarySortOption === 'name') {
                    filesToRender.sort((a, b) => (a.title || a.name).localeCompare(b.title || b.name));
                } else if (librarySortOption === 'size') {
                    filesToRender.sort((a, b) => b.size - a.size);
                }

                // Prepend Dedicated permanent "+ Create New" Card as the first card in the grid
                const createNewCol = document.createElement('div');
                createNewCol.className = libraryViewMode === 'grid' ? 'col-sm-6 col-md-4' : 'col-12';
                createNewCol.innerHTML = `
                    <div class="create-new-card h-100" onclick="triggerFileInput()">
                        <i class="bi bi-plus-lg fs-3"></i>
                        <span class="fw-semibold">Add New Source</span>
                    </div>
                `;
                grid.appendChild(createNewCol);

                // Render matching notebooks
                filesToRender.forEach(f => {
                    const baseName = f.stem || f.name.replace(/\.[^/.]+$/, "");
                    const displayName = f.title || baseName.replace(/_/g, " ");
                    
                    const vocabName = `${baseName}_vocabulary.md`;
                    const grammarName = `${baseName}_grammar.md`;
                    const summariesList = compiledUnits.concepts_by_source && (compiledUnits.concepts_by_source[f.name] || compiledUnits.concepts_by_source[f.stem] || compiledUnits.concepts_by_source[baseName]);

                    const hasVocab = compiledUnits.vocabulary && compiledUnits.vocabulary.some(v => v.endsWith(vocabName) || v === vocabName);
                    const hasGrammar = compiledUnits.grammar && compiledUnits.grammar.some(g => g.endsWith(grammarName) || g === grammarName);
                    const hasSummary = summariesList && summariesList.length > 0;

                    const vocabClass = hasVocab ? 'vocab-done' : '';
                    const grammarClass = hasGrammar ? 'grammar-done' : '';
                    const summaryClass = hasSummary ? 'summary-done' : '';

                    const mindmapName = `${baseName}_mindmap.html`;
                    const hasMindmap = compiledUnits.quizzes && compiledUnits.quizzes.some(q => q.toLowerCase().endsWith(mindmapName.toLowerCase()));
                    const mindmapClass = hasMindmap ? 'mindmap-done' : '';

                    const card = document.createElement('div');
                    card.className = libraryViewMode === 'grid' ? 'col-sm-6 col-md-4' : 'col-12';
                    const dateStr = "Jun 29, 2026"; // Consistent date metadata

                    if (libraryViewMode === 'grid') {
                        card.innerHTML = `
                            <div class="source-card d-flex flex-column justify-content-between h-100 position-relative ${f.is_transcribing ? 'transcribing-card' : ''}" onclick="openUnitWorkspace('${esc(f.name)}')">
                                <div>
                                    <!-- Cover Thumbnail and Action menu -->
                                    <div class="d-flex align-items-start justify-content-between mb-3">
                                        <div class="notebook-thumbnail d-flex align-items-center justify-content-center ${f.is_transcribing ? 'transcribing-spinner' : ''}">
                                            <div class="notebook-spine"></div>
                                            <i class="bi ${f.is_transcribing ? 'bi-soundwave text-info' : 'bi-book-fill'}" style="font-size: 1.4rem; color: #ffffff; opacity: 0.95;"></i>
                                        </div>
                                        
                                        <!-- Three-dot Action Menu -->
                                        <div class="dropdown notebook-action-dropdown" onclick="event.stopPropagation()">
                                            <button class="btn btn-link text-muted p-0 border-0" type="button" data-bs-toggle="dropdown" aria-expanded="false" style="font-size: 1.1rem; line-height: 1;">
                                                <i class="bi bi-three-dots-vertical"></i>
                                            </button>
                                            <ul class="dropdown-menu dropdown-menu-end shadow-sm">
                                                <li><a class="dropdown-item py-1.5 small text-on-surface" href="#" onclick="openUnitWorkspace('${esc(f.name)}'); return false;"><i class="bi bi-folder2-open me-2 text-info"></i>Open Workspace</a></li>
                                                <li><a class="dropdown-item py-1.5 small text-on-surface" href="#" onclick="openSourceEditorForUnit('${esc(f.name)}'); return false;"><i class="bi bi-pencil-square me-2 text-warning"></i>Edit Syllabus</a></li>
                                                ${f.is_transcribing ? '' : `<li><a class="dropdown-item py-1.5 small text-on-surface" href="#" onclick="currentUnit='${esc(f.name)}'; triggerWorkspaceCompile(); return false;"><i class="bi bi-cpu me-2 text-warning"></i>Run Compiler</a></li>`}
                                                ${f.is_transcribing ? '' : `<li><a class="dropdown-item py-1.5 small text-on-surface" href="#" onclick="openUnitWorkspace('${esc(f.name)}'); switchWorkspaceSubTab('handout'); return false;"><i class="bi bi-patch-question-fill me-2 text-danger"></i>Generate Quiz</a></li>`}
                                            </ul>
                                        </div>
                                    </div>
                                    
                                    ${f.is_transcribing ? `
                                    <div class="mb-1.5 d-flex align-items-center gap-1 text-info fw-semibold font-monospace" style="font-size: 0.62rem; letter-spacing: 0.05em;">
                                        <span class="spinner-grow spinner-grow-sm text-info me-1" style="width: 0.5rem; height: 0.5rem;"></span>
                                        TRANSCRIBING...
                                    </div>
                                    ` : ''}

                                    <h6 class="fw-bold mb-1 text-truncate" title="${displayName}">${displayName}</h6>
                                    <!-- Date & Source Metadata -->
                                    <div class="d-flex align-items-center gap-2 mb-3" style="font-size: 0.7rem; color: var(--on-surface-variant);">
                                        <span>${dateStr}</span>
                                        <span class="text-muted">•</span>
                                        <span>${f.is_transcribing ? 'Whisper Active' : `${(f.size / 1024).toFixed(1)} KB`}</span>
                                    </div>
                                </div>
                                
                                <div class="d-flex align-items-center justify-content-between pt-2 border-top" style="border-color: var(--card-border) !important;">
                                    <span class="text-muted small" style="font-size: 0.65rem;">${f.is_transcribing ? 'Processing audio...' : 'Linguistic Nodes'}</span>
                                    <div class="d-flex gap-1">
                                        ${f.is_transcribing ? `
                                        <span class="spinner-border spinner-border-sm text-info flex-shrink-0" role="status"></span>
                                        ` : `
                                        <span class="unit-target-pill ${vocabClass}" title="Vocabulary unit">V</span>
                                        <span class="unit-target-pill ${grammarClass}" title="Grammar structures">G</span>
                                        <span class="unit-target-pill ${summaryClass}" title="Unit summary concepts">S</span>
                                        <span class="unit-target-pill ${mindmapClass}" title="Mind Map">M</span>
                                        `}
                                    </div>
                                </div>
                            </div>
                        `;
                    } else {
                        // List mode
                        card.innerHTML = `
                            <div class="source-card d-flex align-items-center justify-content-between py-2.5 px-3 ${f.is_transcribing ? 'transcribing-card' : ''}" onclick="openUnitWorkspace('${esc(f.name)}')">
                                <div class="d-flex align-items-center gap-3 text-truncate" style="flex: 1;">
                                    <div class="notebook-thumbnail d-flex align-items-center justify-content-center ${f.is_transcribing ? 'transcribing-spinner' : ''}" style="width: 32px; height: 40px; border-radius: 4px;">
                                        <div class="notebook-spine" style="width: 4px;"></div>
                                        <i class="bi ${f.is_transcribing ? 'bi-soundwave text-info' : 'bi-book-fill'}" style="font-size: 0.95rem; color: #ffffff; opacity: 0.95;"></i>
                                    </div>
                                    <div class="text-truncate">
                                        <h6 class="fw-bold mb-0 text-truncate" style="font-size: 0.9rem;" title="${displayName}">${displayName}</h6>
                                        <div class="d-flex align-items-center gap-2" style="font-size: 0.68rem; color: var(--on-surface-variant);">
                                            <span>${dateStr}</span>
                                            <span class="text-muted">•</span>
                                            <span>${f.is_transcribing ? 'Transcribing local video...' : `${(f.size / 1024).toFixed(1)} KB`}</span>
                                        </div>
                                    </div>
                                </div>
                                <div class="d-flex align-items-center gap-3">
                                    ${f.is_transcribing ? `
                                    <span class="badge bg-info-subtle text-info border border-info-subtle py-0.5 px-2 rounded-pill font-monospace" style="font-size: 0.6rem;"><i class="bi bi-cpu animate-pulse me-1"></i>WHISPER</span>
                                    ` : `
                                    <div class="d-flex gap-1">
                                        <span class="unit-target-pill ${vocabClass}" title="Vocabulary unit">V</span>
                                        <span class="unit-target-pill ${grammarClass}" title="Grammar structures">G</span>
                                        <span class="unit-target-pill ${summaryClass}" title="Unit summary concepts">S</span>
                                        <span class="unit-target-pill ${mindmapClass}" title="Mind Map">M</span>
                                    </div>
                                    `}
                                    <!-- Three-dot Action Menu -->
                                    <div class="dropdown notebook-action-dropdown" onclick="event.stopPropagation()">
                                        <button class="btn btn-link text-muted p-0 border-0" type="button" data-bs-toggle="dropdown" aria-expanded="false" style="font-size: 1.1rem; line-height: 1;">
                                            <i class="bi bi-three-dots-vertical"></i>
                                        </button>
                                        <ul class="dropdown-menu dropdown-menu-end shadow-sm">
                                            <li><a class="dropdown-item py-1.5 small text-on-surface" href="#" onclick="openUnitWorkspace('${esc(f.name)}'); return false;"><i class="bi bi-folder2-open me-2 text-info"></i>Open Workspace</a></li>
                                            <li><a class="dropdown-item py-1.5 small text-on-surface" href="#" onclick="openSourceEditorForUnit('${esc(f.name)}'); return false;"><i class="bi bi-pencil-square me-2 text-warning"></i>Edit Syllabus</a></li>
                                            ${f.is_transcribing ? '' : `<li><a class="dropdown-item py-1.5 small text-on-surface" href="#" onclick="currentUnit='${esc(f.name)}'; triggerWorkspaceCompile(); return false;"><i class="bi bi-cpu me-2 text-warning"></i>Run Compiler</a></li>`}
                                            ${f.is_transcribing ? '' : `<li><a class="dropdown-item py-1.5 small text-on-surface" href="#" onclick="openUnitWorkspace('${esc(f.name)}'); switchWorkspaceSubTab('handout'); return false;"><i class="bi bi-patch-question-fill me-2 text-danger"></i>Generate Quiz</a></li>`}
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        `;
                    }
                    grid.appendChild(card);
                });

                // Restore scroll position after DOM rebuild (after all cards appended)
                if (grid) grid.scrollTop = gridScrollTop;

                // Reactively update focus pane if inside a workspace
                if (currentUnit) {
                    updateWorkspaceViewport(currentUnit);
                }

                // Update Workspace Sidebar Files List with Collapsible Groups (Option 1 - Grouped Units)
                const sidebarList = document.getElementById('workspace-sidebar-files-list');
                if (sidebarList) {
                    sidebarList.innerHTML = '';

                    // 1. Group files and compiled assets into Units
                    const unitsMap = {};
                    
                    function getUnitNameFromPath(path) {
                        if (!path) return '';
                        const cleanPath = path.replace(/\\/g, '/').replace(/^\//, '').replace(/^wiki\//, '');
                        const parts = cleanPath.split('/');
                        if (parts.length > 1) return parts[0];
                        const file = parts[0];
                        return file.replace(/(_vocabulary|_grammar|_expressions|_summary|_mindmap|_listening_quiz|_reading_quiz|_video_learning|_translation_quiz)?\.(md|html|txt)$/i, '');
                    }

                    // Map raw textbook/source files
                    rawFiles.forEach(f => {
                        const baseName = f.stem || f.name.replace(/\.[^/.]+$/, "");
                        unitsMap[baseName] = {
                            id: baseName,
                            displayName: baseName.replace(/_/g, ' '),
                            rawFile: f,
                            vocabulary: null,
                            grammar: null,
                            expressions: null,
                            summary: null,
                            mindmap: null,
                            quizzes: [],
                            supplemental: [],
                            media: []
                        };
                    });

                    // Map compiled assets
                    if (compiledUnits.vocabulary) {
                        compiledUnits.vocabulary.forEach(v => {
                            const u = getUnitNameFromPath(v);
                            if (!unitsMap[u]) {
                                unitsMap[u] = { id: u, displayName: u.replace(/_/g, ' '), rawFile: null, vocabulary: null, grammar: null, expressions: null, summary: null, mindmap: null, quizzes: [], supplemental: [], media: [] };
                            }
                            unitsMap[u].vocabulary = v;
                        });
                    }
                    if (compiledUnits.grammar) {
                        compiledUnits.grammar.forEach(g => {
                            const u = getUnitNameFromPath(g);
                            if (!unitsMap[u]) {
                                unitsMap[u] = { id: u, displayName: u.replace(/_/g, ' '), rawFile: null, vocabulary: null, grammar: null, expressions: null, summary: null, mindmap: null, quizzes: [], supplemental: [], media: [] };
                            }
                            unitsMap[u].grammar = g;
                        });
                    }
                    if (compiledUnits.expressions) {
                        compiledUnits.expressions.forEach(e => {
                            const u = getUnitNameFromPath(e);
                            if (!unitsMap[u]) {
                                unitsMap[u] = { id: u, displayName: u.replace(/_/g, ' '), rawFile: null, vocabulary: null, grammar: null, expressions: null, summary: null, mindmap: null, quizzes: [], supplemental: [], media: [] };
                            }
                            unitsMap[u].expressions = e;
                        });
                    }
                    if (compiledUnits.summaries) {
                        compiledUnits.summaries.forEach(s => {
                            const u = getUnitNameFromPath(s);
                            if (!unitsMap[u]) {
                                unitsMap[u] = { id: u, displayName: u.replace(/_/g, ' '), rawFile: null, vocabulary: null, grammar: null, expressions: null, summary: null, mindmap: null, quizzes: [], supplemental: [], media: [] };
                            }
                            unitsMap[u].summary = s;
                        });
                    }
                    if (compiledUnits.quizzes) {
                        compiledUnits.quizzes.forEach(q => {
                            const u = getUnitNameFromPath(q);
                            if (!unitsMap[u]) {
                                unitsMap[u] = { id: u, displayName: u.replace(/_/g, ' '), rawFile: null, vocabulary: null, grammar: null, expressions: null, summary: null, mindmap: null, quizzes: [], supplemental: [], media: [] };
                            }
                            if (q.toLowerCase().endsWith('_mindmap.html') || q.toLowerCase().endsWith('_mindmap.json')) {
                                unitsMap[u].mindmap = q;
                            } else {
                                unitsMap[u].quizzes.push(q);
                            }
                        });
                    }
                    if (compiledUnits.supplemental) {
                        compiledUnits.supplemental.forEach(s => {
                            const u = getUnitNameFromPath(s);
                            if (!unitsMap[u]) {
                                unitsMap[u] = { id: u, displayName: u.replace(/_/g, ' '), rawFile: null, vocabulary: null, grammar: null, expressions: null, summary: null, mindmap: null, quizzes: [], supplemental: [], media: [] };
                            }
                            unitsMap[u].supplemental.push(s);
                        });
                    }
                    if (compiledUnits.media) {
                        compiledUnits.media.forEach(m => {
                            const u = getUnitNameFromPath(m);
                            if (!unitsMap[u]) {
                                unitsMap[u] = { id: u, displayName: u.replace(/_/g, ' '), rawFile: null, vocabulary: null, grammar: null, expressions: null, summary: null, mindmap: null, quizzes: [], supplemental: [], media: [] };
                            }
                            if (!unitsMap[u].media) unitsMap[u].media = [];
                            unitsMap[u].media.push(m);
                        });
                    }

                    const unitsList = Object.values(unitsMap);
                    const activeMediaMap = compiledUnits.active_media || {};
                    
                    // Robust, normalized matching for individual Node/unit management (ignores case, extensions, and underscores)
                    const activeUnit = unitsList.find(unit => {
                        if (!currentUnit) return false;
                        const cleanCurrentUnit = currentUnit.replace(/\.[^/.]+$/, "").replace(/_/g, "").toLowerCase();
                        const cleanUnitId = unit.id.replace(/_/g, "").toLowerCase();
                        const cleanRawName = unit.rawFile ? unit.rawFile.name.replace(/\.[^/.]+$/, "").replace(/_/g, "").toLowerCase() : '';
                        return cleanUnitId === cleanCurrentUnit || cleanRawName === cleanCurrentUnit;
                    });
                    const unitsToRender = activeUnit ? [activeUnit] : [];

                    let activeFilesCount = 0;
                    if (activeUnit) {
                        if (activeUnit.rawFile) activeFilesCount++;
                        if (activeUnit.media) activeFilesCount += activeUnit.media.length;
                    }
                    document.getElementById('workspace-sidebar-files-count').innerText = `${activeFilesCount}`;

                    unitsToRender.forEach(unit => {
                        // If this unit corresponds to current workspace
                        const isCurrentUnit = (unit.id === currentUnit || (unit.rawFile && unit.rawFile.name === currentUnit));

                        // Create a flat container without collapsible details accordion header
                        const groupDetails = document.createElement('div');
                        groupDetails.className = 'sidebar-unit-group';

                        groupDetails.innerHTML = `
                            <div class="sidebar-unit-content px-1 py-1 flex-column gap-2">
                                
                                <!-- 1. SOURCES SECTION -->
                                <div class="unit-tree-section">
                                    <div class="text-uppercase text-secondary fw-bold mb-1.5 d-flex align-items-center" style="font-size: 0.58rem; letter-spacing: 0.05em;">
                                        <i class="bi bi-folder-fill me-1 text-warning"></i>Sources
                                    </div>
                                    <div class="ps-1">
                                        ${unit.rawFile ? `
                                        <a class="sidebar-unit-link py-1 px-2.5 d-flex align-items-center justify-content-between text-truncate ${isCurrentUnit && activeSubTab === 'raw' ? 'active' : ''}" onclick="openUnitWorkspace('${esc(unit.rawFile.name)}'); switchWorkspaceSubTab('raw')" style="min-width: 0;">
                                            <span class="text-truncate" title="${esc(unit.rawFile.title || unit.rawFile.name)}"><i class="bi bi-file-earmark-text-fill me-2 text-secondary"></i>${esc(unit.rawFile.title || unit.rawFile.name)}</span>
                                            <span class="text-muted ms-2 flex-shrink-0" style="font-size: 0.6rem;">${(unit.rawFile.size / 1024).toFixed(1)} KB</span>
                                        </a>
                                        ` : '<div class="text-muted ps-3 py-1" style="font-size: 0.65rem; font-style: italic;">No source file</div>'}
                                        
                                        ${unit.media && unit.media.length > 0 ? `
                                        <div class="mt-2.5 ps-1">
                                            <div class="text-uppercase text-secondary fw-bold mb-1.5 d-flex align-items-center" style="font-size: 0.52rem; letter-spacing: 0.05em; opacity: 0.8;">
                                                <i class="bi bi-camera-reels-fill me-1 text-info"></i>Media
                                            </div>
                                             ${unit.media.map(mediaPath => {
                                                 const mName = mediaPath.split('/').pop();
                                                 const isActive = activeMediaMap[unit.id] === mName;
                                                 const isMd = mName.toLowerCase().endsWith('.md');
                                                 const clickAction = isMd
                                                     ? `openUnitWorkspace('${unit.rawFile ? esc(unit.rawFile.name) : esc(unit.id)}'); loadWorkspaceSourceDoc('/wiki/${esc(mediaPath)}', '${esc(mName)}'); switchWorkspaceSubTab('raw')`
                                                     : `openUnitWorkspace('${unit.rawFile ? esc(unit.rawFile.name) : esc(unit.id)}'); loadHandoutPreview('/wiki/${esc(mediaPath)}')`;
                                                 return `
                                                 <div class="d-flex align-items-center gap-1 my-0.5" style="border-radius: var(--radius-sm); overflow: hidden;">
                                                     <input type="radio" name="active-media-${unit.id}" class="form-check-input my-0 ms-1" style="width: 0.72rem; height: 0.72rem; cursor: pointer; flex-shrink: 0;" ${isActive ? 'checked' : ''} onclick="event.stopPropagation(); selectActiveMedia('${esc(unit.id)}', '${esc(mName)}')" />
                                                     <a class="sidebar-unit-link flex-grow-1 py-1 px-2 border-0 m-0 d-flex align-items-center justify-content-between text-truncate" onclick="${clickAction}" style="min-width: 0;">
                                                         <span class="text-truncate"><i class="bi ${isMd ? 'bi-file-earmark-text-fill text-warning' : 'bi-file-earmark-play-fill text-info'} me-2"></i>${mName}</span>
                                                     </a>
                                                 </div>
                                                 `;
                                             }).join('')}
                                        </div>
                                        ` : ''}
                                    </div>
                                </div>
                                
                            </div>
                        `;
                        sidebarList.appendChild(groupDetails);
                    });
                }
            })
            .catch(err => { console.error('fetchRawFiles:', err); });
    }

    function formatRelativeTime(mtime) {
        if (!mtime) return '';
        const diffMs = Date.now() - (mtime * 1000);
        const diffMins = Math.floor(diffMs / 60000);
        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins}m ago`;
        const diffHours = Math.floor(diffMins / 60);
        if (diffHours < 24) return `${diffHours}h ago`;
        return new Date(mtime * 1000).toLocaleDateString(undefined, {month: 'short', day: 'numeric'});
    }

    function fetchCompiledUnits() {
        fetch('/api/compiled-units')
            .then(res => res.json())
            .then(data => {
                compiledUnits = data.units;
                
                fetchRawFiles();
            })
            .catch(err => { console.error('fetchCompiledUnits:', err); fetchRawFiles(); });
    }

    function selectActiveMedia(unitId, mediaName) {
        fetch('/api/select-active-media', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ unit: unitId, media: mediaName })
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                fetchCompiledUnits();
            } else {
                alert('Error selecting active media: ' + data.error);
            }
        })
        .catch(err => { alert('Network error: ' + err.message); });
    }

    // ------------------ BACKGROUND TASK MONITORING ------------------
    let selectedCompileJobId = null;
    let recentJobs = []; // Global store for active job monitoring
    function fetchRecentJobs() {
        fetch('/api/recent-jobs')
            .then(res => res.json())
            .then(data => {
                recentJobs = data.jobs || [];
                const list = document.getElementById('recent-jobs-list');
                if (!list) return;

                // Save scroll position before clearing DOM
                const scrollTop = list.scrollTop;

                if (recentJobs.length === 0) {
                    list.innerHTML = `
                        <div class="text-muted small text-center py-3">
                            <i class="bi bi-check2-circle fs-5 text-success opacity-50 mb-1 d-block"></i>
                            All systems ready. No active tasks.
                        </div>`;
                    const statusBadge = document.getElementById('task-center-status');
                    if (statusBadge) {
                        statusBadge.innerText = 'IDLE';
                        statusBadge.className = 'badge rounded-pill bg-secondary-subtle text-secondary border border-secondary-subtle font-monospace';
                    }
                } else {
                    list.innerHTML = '';
                    let activeCount = 0;
                    
                    // Restore scroll position after DOM rebuild
                    requestAnimationFrame(() => { list.scrollTop = scrollTop; });
                    // Show last 4 jobs in reverse (newest first)
                    recentJobs.slice(-4).reverse().forEach(job => {
                        let iconHtml = '';
                        let typeLabel = '';
                        let actionBtnHtml = '';
                        
                        if (job.status === 'running') {
                            activeCount++;
                        }

                        // Determine Icon & Label based on job type
                        if (job.type === 'compile') {
                            iconHtml = job.status === 'running' 
                                ? `<span class="spinner-border spinner-border-sm text-warning me-2 flex-shrink-0" role="status"></span>`
                                : `<i class="bi bi-cpu text-warning me-2 flex-shrink-0" style="font-size: 0.9rem;"></i>`;
                            typeLabel = 'Compile Unit';
                        } else if (job.type === 'quiz') {
                            iconHtml = job.status === 'running'
                                ? `<span class="spinner-border spinner-border-sm text-danger me-2 flex-shrink-0" role="status"></span>`
                                : `<i class="bi bi-patch-question text-danger me-2 flex-shrink-0" style="font-size: 0.9rem;"></i>`;
                            typeLabel = 'Generate Quiz';
                            // Action buttons removed as requested to delegate navigation entirely to the file tree
                        } else if (job.type === 'video-import') {
                            iconHtml = job.status === 'running'
                                ? `<span class="spinner-border spinner-border-sm text-info me-2 flex-shrink-0" role="status"></span>`
                                : `<i class="bi bi-film text-info me-2 flex-shrink-0" style="font-size: 0.9rem;"></i>`;
                            typeLabel = 'Video Import';
                            // Action buttons removed as requested to delegate navigation entirely to the file tree
                        } else if (job.type === 'audit-batch') {
                            iconHtml = job.status === 'running'
                                ? `<span class="spinner-border spinner-border-sm text-success me-2 flex-shrink-0" role="status"></span>`
                                : `<i class="bi bi-clipboard2-check-fill text-success me-2 flex-shrink-0" style="font-size: 0.9rem;"></i>`;
                            typeLabel = 'Expert Re-audit';
                        } else {
                            iconHtml = `<i class="bi bi-gear me-2 flex-shrink-0"></i>`;
                            typeLabel = 'System Task';
                        }

                        // Determine Badge styling
                        let statusBadgeHtml = '';
                        if (job.status === 'completed') {
                            statusBadgeHtml = `<span class="badge bg-success-subtle text-success border border-success-subtle py-0.5 px-1.5" style="font-size: 0.58rem;">DONE</span>`;
                        } else if (job.status === 'failed') {
                            statusBadgeHtml = `<span class="badge bg-danger-subtle text-danger border border-danger-subtle py-0.5 px-1.5" style="font-size: 0.58rem;">FAIL</span>`;
                        } else if (job.type === 'quiz' && (job.stage === 'auditing' || job.stage === 'correcting')) {
                            statusBadgeHtml = `<span class="badge bg-info-subtle text-info border border-info-subtle py-0.5 px-1.5" style="font-size: 0.58rem;">AUDITING</span>`;
                        } else {
                            statusBadgeHtml = `<span class="badge bg-warning-subtle text-warning border border-warning-subtle py-0.5 px-1.5" style="font-size: 0.58rem;">${job.progress}%</span>`;
                        }

                        // Get latest user-friendly log entry for display (the last log line)
                        let latestLog = '';
                        if (job.logs && job.logs.length > 0) {
                            latestLog = job.logs[job.logs.length - 1];
                            // Clean up standard debug prefixes
                            latestLog = latestLog.replace(/^\[[A-Z\-]+\]\s*/i, '');
                        }

                        const item = document.createElement('div');
                        item.className = 'p-2 rounded-2 d-flex flex-column gap-1';
                        item.style.background = 'rgba(255, 255, 255, 0.02)';
                        item.style.border = '1px solid rgba(255, 255, 255, 0.05)';
                        item.innerHTML = `
                            <div class="d-flex align-items-center justify-content-between">
                                <div class="d-flex align-items-center text-truncate" style="max-width: 75%;">
                                    ${iconHtml}
                                    <div class="text-truncate">
                                        <span class="fw-semibold text-on-surface" style="font-size: 0.72rem;">${job.target}</span>
                                        <span class="text-muted d-block" style="font-size: 0.62rem;">${typeLabel}${actionBtnHtml}</span>
                                    </div>
                                </div>
                                <div>
                                    ${statusBadgeHtml}
                                </div>
                            </div>
                            ${job.status === 'running' ? `
                            <!-- Primary Task Progress Bar (retained at 100% when drafting finishes) -->
                            <div class="mt-1">
                                <div class="d-flex align-items-center justify-content-between px-0.5 mb-1" style="line-height: 1;">
                                    <span class="font-monospace text-muted" style="font-size: 0.6rem;">
                                        <i class="bi bi-cpu me-1 ${job.progress >= 100 ? 'text-success' : 'text-primary'}"></i>${job.type === 'quiz' ? 'Draft Generation' : 'Task Progress'}
                                    </span>
                                    <span class="font-monospace fw-semibold ${job.progress >= 100 ? 'text-success' : 'text-primary'}" style="font-size: 0.6rem;">
                                        ${job.progress}%
                                    </span>
                                </div>
                                <div class="progress" style="height: 4px; background: rgba(255, 255, 255, 0.08); border-radius: 2px;">
                                    <div class="progress-bar ${job.progress >= 100 ? 'bg-success' : ((job.type === 'compile' ? 'bg-warning' : 'bg-primary') + ' progress-bar-striped progress-bar-animated')}" role="progressbar" style="width: ${job.progress}%; border-radius: 2px;"></div>
                                </div>
                            </div>
                            <!-- Level 2 Expert Audit Progress Bar (shown below when auditing/correcting) -->
                            ${job.type === 'quiz' && (job.stage === 'auditing' || job.stage === 'correcting' || (job.audit_progress && job.audit_progress > 0)) ? `
                            <div class="mt-1">
                                <div class="d-flex align-items-center justify-content-between px-0.5 mb-1" style="line-height: 1;">
                                    <span class="font-monospace text-info" style="font-size: 0.6rem;">
                                        <i class="bi bi-shield-check me-1"></i>Level 2 Expert Audit
                                    </span>
                                    <span class="font-monospace text-info fw-semibold" style="font-size: 0.6rem;">
                                        ${job.audit_progress || 0}%
                                    </span>
                                </div>
                                <div class="progress" style="height: 4px; background: rgba(255, 255, 255, 0.08); border-radius: 2px;">
                                    <div class="progress-bar bg-info progress-bar-striped progress-bar-animated" role="progressbar" style="width: ${job.audit_progress || 0}%; border-radius: 2px;"></div>
                                </div>
                            </div>
                            ` : ''}
                            ` : ''}
                            ${latestLog ? `
                            <div class="text-muted text-truncate font-monospace mt-0.5" style="font-size: 0.6rem; opacity: 0.75; padding-left: 1.25rem;">
                                ${latestLog}
                            </div>
                            ` : ''}
                        `;
                        list.appendChild(item);

                        // If compiles are running inside active unit, propagate log updates
                        if (job.type === 'compile' && currentUnit && (job.target === currentUnit || job.target.startsWith(currentUnit + ' ('))) {
                            selectedCompileJobId = job.id;

                            const term = document.getElementById('workspace-compiler-terminal');
                            if (term) {
                                term.innerHTML = '';
                                job.logs.forEach(log => {
                                    let color = 'var(--logger-text)';
                                    if (log.startsWith('❌') || log.startsWith('[ERROR]')) color = '#ff8080';
                                    if (log.startsWith('✨') || log.startsWith('✅') || log.startsWith('[COMPILE] Finished')) color = '#80ff80';
                                    term.innerHTML += `<div class="log-entry" style="color: ${color}; margin-bottom: 2px;">${log}</div>`;
                                });
                                term.scrollTop = term.scrollHeight;
                            }
                        }
                    });

                    // Update main status badge for Task Center
                    const mainStatusBadge = document.getElementById('task-center-status');
                    if (mainStatusBadge) {
                        if (activeCount > 0) {
                            mainStatusBadge.innerText = `${activeCount} RUNNING`;
                            mainStatusBadge.className = 'badge rounded-pill bg-warning-subtle text-warning border border-warning-subtle animate-pulse';
                        } else {
                            mainStatusBadge.innerText = 'IDLE';
                            mainStatusBadge.className = 'badge rounded-pill bg-secondary-subtle text-secondary border border-secondary-subtle';
                        }
                    }
                }
                
                // If the active unit is transcribing, refresh compiled units and raw files to check if finished
                if (currentUnit) {
                    const activeFile = rawFiles.find(f => f.name === currentUnit);
                    if (activeFile && activeFile.is_transcribing) {
                        fetchCompiledUnits();
                    }
                }
            });
    }

    function pollJobStatus(jobId, service) {
        if (activePollTimers[jobId]) clearTimeout(activePollTimers[jobId]);

        fetch('/api/job-status', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ job_id: jobId })
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                const job = data.job;
                if (service === 'compiler') {
                    fetchRecentJobs();
                } else if (service === 'quiz') {
                    fetchRecentJobs();
                }

                if (job.status === 'completed' || job.status === 'failed') {
                    delete activePollTimers[jobId];
                    
                    if (job.status === 'completed' && job.type === 'video-import' && job.saved_files && job.saved_files.length > 0) {
                        const p = job.saved_files[0];
                        const parts = p.split('/');
                        const wikiIdx = parts.indexOf('wiki');
                        let folderName = null;
                        if (wikiIdx !== -1 && wikiIdx + 1 < parts.length) {
                            folderName = parts[wikiIdx + 1];
                        }
                        
                        fetch('/api/compiled-units')
                            .then(res => res.json())
                            .then(data => {
                                compiledUnits = data.units;
                                fetch('/api/raw-files')
                                    .then(res => res.json())
                                    .then(raw_data => {
                                        rawFiles = raw_data.files;
                                        document.getElementById('stat-raw-files-count').innerText = `${rawFiles.length} file${rawFiles.length === 1 ? '' : 's'}`;
                                        
                                        // Update sidebar / grid
                                        const grid = document.getElementById('document-library-grid');
                                        if (grid) {
                                            // Trigger normal redraw
                                            fetchRawFiles();
                                        }
                                        
                                        let targetFile = null;
                                        if (folderName) {
                                            targetFile = rawFiles.find(f => f.stem === folderName || f.stem.toLowerCase() === folderName.toLowerCase());
                                        }
                                        
                                        if (targetFile) {
                                            openUnitWorkspace(targetFile.name);
                                            switchWorkspaceSubTab('raw');
                                        } else {
                                            const finishedFilename = p.split('/').pop();
                                            if (finishedFilename) {
                                                openUnitWorkspace(finishedFilename);
                                                switchWorkspaceSubTab('raw');
                                            }
                                        }
                                    });
                            });
                    } else {
                        fetchCompiledUnits();
                    }
                    
                    if (service === 'compiler') {
                        isCompiling = false;
                        updateCompileButtonUI();
                    }

                    if (job.type === 'audit-batch') {
                        setReAuditRunning(false);
                        if (job.status === 'completed' && job.result) {
                            showReAuditSummary(job.result);
                            const raModal = document.getElementById('re-audit-modal');
                            bootstrap.Modal.getOrCreateInstance(raModal).show();
                        }
                    }
                    
                    if (job.status === 'failed') {
                        const lastLog = (job.logs && job.logs.length > 0) ? job.logs[job.logs.length - 1] : "Job encountered an unexpected error.";
                        let errorMsg = lastLog;
                        if (lastLog.includes("faster-whisper is not installed")) {
                            errorMsg = "<strong>faster-whisper is not installed.</strong><br><br>To transcribe local video/audio files, please run in your terminal:<br><pre class='p-2 bg-dark rounded border mt-2'>pip install faster-whisper</pre>";
                        }
                        showInfoModal("Task Execution Failed", errorMsg);
                    }

                    if (service === 'quiz' && job.status === 'completed' && job.saved_files.length > 0) {
                        loadHandoutPreview(job.saved_files[0]);
                        document.getElementById('quiz-progress-section').style.display = 'none';
                        switchWorkspaceSubTab('handout');
                    }
                } else {
                    activePollTimers[jobId] = setTimeout(() => pollJobStatus(jobId, service), 1500);
                }
            }
        });
    }

    function generateQuiz() {
        const template = document.getElementById('quiz-template-select').value;
        const count = document.getElementById('quiz-count-input').value;

        if (!currentUnit) {
            showInfoModal("No Workspace Selected", "Please select a unit workspace first.");
            return;
        }

        if (template === 'video') {
            fetch('/api/check-video-source', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: currentUnit })
            })
            .then(res => res.json())
            .then(data => {
                if (data.success && data.is_video) {
                    proceedWithQuizGeneration(currentUnit, count, template);
                } else {
                    showInfoModal("Video Handout Unavailable", "Cannot generate video handout: This raw unit source is not mapped to an active media file or transcript.");
                }
            });
        } else {
            proceedWithQuizGeneration(currentUnit, count, template);
        }
    }

    function proceedWithQuizGeneration(filename, count, template) {
        fetch('/api/generate-quiz', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename, count, template })
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                pollJobStatus(data.job_id, 'quiz');
                fetchRecentJobs();
            } else {
                showInfoModal("Quiz Generation Failed", "Failed to start quiz generation: " + (data.error || "Unknown error"));
            }
        });
    }

    // ------------------ DYNAMIC FORMS & PARAMETERS HELPERS ------------------
    function onApiTypeChange(triggerFetch = true) {
        const val = document.getElementById('config-api-type').value;
        if (triggerFetch) {
            fetchModels();
        }
    }

    const KOKORO_VOICES = [
        { value: "af_sarah", label: "US Female: Sarah" },
        { value: "af_bella", label: "US Female: Bella" },
        { value: "am_michael", label: "US Male: Michael" },
        { value: "bf_emma", label: "UK Female: Emma" }
    ];
    const EDGE_VOICES = [
        { value: "en-US-AriaNeural", label: "US Female: Aria" },
        { value: "en-US-JennyNeural", label: "US Female: Jenny" },
        { value: "en-GB-RyanNeural", label: "UK Male: Ryan" }
    ];

    function onTTSEngineChange(selectedVoiceA = null, selectedVoiceB = null, autoFillEndpoint = true) {
        const engine = document.getElementById('config-tts-engine').value;
        const voiceASelect = document.getElementById('config-tts-voice-a');
        const voiceBSelect = document.getElementById('config-tts-voice-b');
        if (!voiceASelect || !voiceBSelect) return;

        if (autoFillEndpoint) {
            const urlInput = document.getElementById('config-tts-url');
            if (urlInput) {
                urlInput.value = engine === 'kokoro' 
                    ? 'http://localhost:8880/v1/audio/speech' 
                    : 'http://localhost:5050/v1/audio/speech';
            }
        }

        const voices = engine === 'kokoro' ? KOKORO_VOICES : EDGE_VOICES;
        [voiceASelect, voiceBSelect].forEach((select, index) => {
            select.innerHTML = '';
            voices.forEach(v => {
                const opt = document.createElement('option');
                opt.value = v.value;
                opt.innerText = v.label;
                select.appendChild(opt);
            });
            const selVal = index === 0 ? selectedVoiceA : selectedVoiceB;
            if (selVal && voices.some(v => v.value === selVal)) {
                select.value = selVal;
            } else if (voices.length > index) {
                select.value = voices[index].value;
            }
        });
    }

    function selectQuizArch(val) {
        document.getElementById('quiz-template-select').value = val;
        ['translation', 'reading', 'vocabulary', 'listening', 'video'].forEach(arch => {
            const btn = document.getElementById('btn-arch-' + arch);
            if (btn) {
                if (arch === val) btn.classList.add('active');
                else btn.classList.remove('active');
            }
        });
        
        let defaultCount = 10;
        if (appConfig.quiz_defaults && appConfig.quiz_defaults[val] !== undefined) {
            defaultCount = appConfig.quiz_defaults[val];
        }
        updateQuizCount(defaultCount);
    }

    function updateQuizCount(val) {
        document.getElementById('quiz-count-slider').value = val;
        document.getElementById('quiz-count-badge').innerText = val;
        document.getElementById('quiz-count-input').value = val;
    }

    // ------------------ SOURCE MATERIAL & SYLLABUS EDITOR ------------------
    let editorTargetFilename = null;

    function openSourceEditorForNew() {
        editorTargetFilename = null;
        document.getElementById('sourceEditorModalLabel').innerText = "Add New Source & Syllabus";
        document.getElementById('editor-unit-name').value = "";
        document.getElementById('editor-unit-name').removeAttribute('readonly');
        document.getElementById('editor-passage-text').value = "";
        document.getElementById('editor-syllabus-vocab').value = "";
        document.getElementById('editor-syllabus-grammar').value = "";
        updateEditorCharCount();
        updateSyllabusVocabCount();
        updateSyllabusGrammarCount();

        const modalEl = document.getElementById('source-editor-modal');
        const modal = bootstrap.Modal.getOrCreateInstance(modalEl);
        modal.show();
    }

    function openSourceEditorForCurrentUnit() {
        if (!currentUnit) {
            showInfoModal("No Unit Selected", "Please select or open a unit workspace first.");
            return;
        }
        openSourceEditorForUnit(currentUnit);
    }

    function openSourceEditorForUnit(filename) {
        editorTargetFilename = filename;
        const stem = filename.replace(/\.[^/.]+$/, "");
        document.getElementById('sourceEditorModalLabel').innerText = `Edit Source & Syllabus: ${stem.replace(/_/g, ' ')}`;
        document.getElementById('editor-unit-name').value = stem;
        document.getElementById('editor-unit-name').setAttribute('readonly', 'true');
        document.getElementById('editor-passage-text').value = "Loading unit source text...";
        document.getElementById('editor-syllabus-vocab').value = "";
        document.getElementById('editor-syllabus-grammar').value = "";

        const modalEl = document.getElementById('source-editor-modal');
        const modal = bootstrap.Modal.getOrCreateInstance(modalEl);
        modal.show();

        // Fetch source content from unit's sources directory
        fetch(`/wiki/${stem}/sources/${filename}`)
            .then(res => {
                if (!res.ok) throw new Error("Could not load source file");
                return res.text();
            })
            .then(text => {
                const parsed = parseSourceContentForEditor(text);
                document.getElementById('editor-passage-text').value = parsed.passage;
                document.getElementById('editor-syllabus-vocab').value = parsed.vocab;
                document.getElementById('editor-syllabus-grammar').value = parsed.grammar;
                updateEditorCharCount();
                updateSyllabusVocabCount();
                updateSyllabusGrammarCount();
            })
            .catch(err => {
                document.getElementById('editor-passage-text').value = "";
                showInfoModal("Load Error", "Unable to load source text: " + err.message);
            });
    }

    function parseSourceContentForEditor(fullText) {
        let passage = fullText;
        let vocab = '';
        let grammar = '';

        // Extract Syllabus Vocabulary section
        const vocabRegex = /\n##\s+Syllabus\s+Vocabulary\s*([\s\S]*?)(?=\n##\s+Syllabus\s+Grammar|$)/i;
        const vocabMatch = passage.match(vocabRegex);
        if (vocabMatch) {
            vocab = vocabMatch[1].trim();
            passage = passage.replace(vocabMatch[0], '');
        }

        // Extract Syllabus Grammar section
        const grammarRegex = /\n##\s+Syllabus\s+Grammar\s*([\s\S]*?)(?=\n##\s+Syllabus\s+Vocabulary|$)/i;
        const grammarMatch = passage.match(grammarRegex);
        if (grammarMatch) {
            grammar = grammarMatch[1].trim();
            passage = passage.replace(grammarMatch[0], '');
        }

        return {
            passage: passage.trim(),
            vocab: vocab,
            grammar: grammar
        };
    }

    function updateEditorCharCount() {
        const text = document.getElementById('editor-passage-text').value || "";
        document.getElementById('editor-passage-count').innerText = `${text.length} chars`;
    }

    function updateSyllabusVocabCount() {
        const text = document.getElementById('editor-syllabus-vocab').value || "";
        const lines = text.split('\n').map(l => l.trim()).filter(l => l.length > 0 && !l.startsWith('#'));
        document.getElementById('editor-vocab-badge').innerText = `${lines.length} items`;
    }

    function updateSyllabusGrammarCount() {
        const text = document.getElementById('editor-syllabus-grammar').value || "";
        const lines = text.split('\n').map(l => l.trim()).filter(l => l.length > 0 && !l.startsWith('#'));
        document.getElementById('editor-grammar-badge').innerText = `${lines.length} topics`;
    }

    function saveSourceAndSyllabus(triggerCompile = false) {
        let rawUnitName = document.getElementById('editor-unit-name').value.trim();
        if (!rawUnitName) {
            showInfoModal("Unit Name Required", "Please enter a valid unit name (e.g. 'Book 1 Unit 2').");
            return;
        }

        let unitStem = rawUnitName.replace(/[\/\\:*?"<>|]/g, '').trim().replace(/\s+/g, '_');
        if (unitStem.endsWith('.md')) unitStem = unitStem.slice(0, -3);
        const filename = `${unitStem}.md`;

        const passage = document.getElementById('editor-passage-text').value.trim();
        const vocab = document.getElementById('editor-syllabus-vocab').value.trim();
        const grammar = document.getElementById('editor-syllabus-grammar').value.trim();

        if (!passage && !vocab) {
            showInfoModal("Content Required", "Please provide lesson passage text or syllabus vocabulary.");
            return;
        }

        // Build markdown content
        let fullMarkdown = passage;
        if (!fullMarkdown.startsWith('---')) {
            const cleanTitle = unitStem.replace(/_/g, ' ');
            fullMarkdown = `---\ntitle: "${cleanTitle}"\ncategory: ["source"]\n---\n\n# ${cleanTitle}\n\n` + fullMarkdown;
        }

        if (vocab) {
            fullMarkdown += `\n\n## Syllabus Vocabulary\n` + vocab;
        }
        if (grammar) {
            fullMarkdown += `\n\n## Syllabus Grammar\n` + grammar;
        }
        fullMarkdown += '\n';

        const saveBtn = document.getElementById('editor-save-btn');
        const saveCompileBtn = document.getElementById('editor-save-compile-btn');
        saveBtn.disabled = true;
        saveCompileBtn.disabled = true;

        fetch('/api/upload-file', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filename: filename,
                content: fullMarkdown
            })
        })
        .then(res => res.json())
        .then(data => {
            saveBtn.disabled = false;
            saveCompileBtn.disabled = false;
            if (data.success) {
                const modalEl = document.getElementById('source-editor-modal');
                const modal = bootstrap.Modal.getInstance(modalEl);
                if (modal) modal.hide();

                showToast(`Successfully saved ${filename}!`, 'success');
                fetchCompiledUnits();
                fetchRawFiles();
                openUnitWorkspace(filename);

                if (triggerCompile) {
                    setTimeout(() => {
                        triggerWorkspaceCompileDirect();
                    }, 300);
                }
            } else {
                showInfoModal("Save Failed", "Failed to save source file: " + (data.error || "Unknown error"));
            }
        })
        .catch(err => {
            saveBtn.disabled = false;
            saveCompileBtn.disabled = false;
            showInfoModal("Network Error", "Failed to save source file: " + err.message);
        });
    }

    // File selection/Drop triggers
    function triggerFileInput() { document.getElementById('file-input').click(); }
    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        // Reset file input value immediately so that subsequent selects of the same file trigger the change event
        e.target.value = "";

        // Check if file is a media file to prevent reading binary data as text
        const filename = file.name;
        const ext = filename.slice((filename.lastIndexOf(".") - 1 >>> 0) + 2).toLowerCase();
        const mediaExts = ["mp4", "mp3", "mkv", "avi", "wav", "aac", "mov", "flv", "m4a"];
        if (mediaExts.includes(ext)) {
            uploadLocalMediaAndTranscribe(file);
            return;
        }

        uploadFile(file);
    }

    function uploadFile(file) {
        const filename = file.name;
        fetch('/api/check-upload-collision', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename, is_media: false, current_unit: currentUnit })
        })
        .then(res => res.json())
        .then(async collisionData => {
            if (collisionData.success && collisionData.collision) {
                let confirmMsg = "";
                if (collisionData.collision_type === "exact_match") {
                    confirmMsg = `A file named '${collisionData.conflicting_file}' already exists. Do you want to overwrite it?`;
                } else if (collisionData.collision_type === "case_mismatch") {
                    confirmMsg = `A file with mismatching casing '${collisionData.conflicting_file}' already exists. Overwriting will replace and delete the old file to prevent duplicates. Do you want to proceed?`;
                } else {
                    confirmMsg = `A file with a different extension '${collisionData.conflicting_file}' already exists. Overwriting will replace and delete the old file to prevent duplicates. Do you want to proceed?`;
                }
                const userConfirmed = await showConfirmModal("Overwrite Existing File?", confirmMsg, "Overwrite");
                if (!userConfirmed) {
                    return;
                }
            }
            
            proceedWithRawFileUpload(file);
        })
        .catch(err => {
            console.error("Collision check failed:", err);
            proceedWithRawFileUpload(file);
        });
    }

    function proceedWithRawFileUpload(file) {
        const filename = file.name;
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/api/upload-file', true);
        xhr.setRequestHeader('X-Filename', encodeURIComponent(filename));
        xhr.setRequestHeader('Content-Type', 'application/octet-stream');

        xhr.onload = function() {
            if (xhr.status === 200) {
                try {
                    const res = JSON.parse(xhr.responseText);
                    if (res.success) {
                        if (res.filename) {
                            openUnitWorkspace(res.filename);
                        }
                        fetchCompiledUnits();
                    } else {
                        alert("Upload failed: " + res.error);
                    }
                } catch(err) {
                    alert("Failed to parse response: " + err.message);
                }
            } else {
                alert("Upload failed with status " + xhr.status);
            }
        };

        xhr.onerror = function() {
            alert("Upload network error.");
        };

        xhr.send(file);
    }

    function switchMediaTab(tab) {
        const webBtn = document.getElementById('tab-web-import');
        const localBtn = document.getElementById('tab-local-import');
        const webPane = document.getElementById('media-web-pane');
        const localPane = document.getElementById('media-local-pane');

        if (tab === 'web') {
            // Web Video becomes active (same gradient as COMPILE SOURCE button)
            webBtn.style.background = 'linear-gradient(135deg, #0ea5e9, #0284c7)';
            webBtn.style.color = '#ffffff';
            webBtn.classList.remove('text-muted');
            webBtn.classList.add('text-white');
            localBtn.style.background = 'transparent';
            localBtn.style.color = '';
            localBtn.classList.remove('text-white');
            localBtn.classList.add('text-muted');
            webPane.style.display = 'block';
            localPane.style.display = 'none';
        } else {
            // Local Video becomes active (same gradient as COMPILE SOURCE button)
            localBtn.style.background = 'linear-gradient(135deg, #0ea5e9, #0284c7)';
            localBtn.style.color = '#ffffff';
            localBtn.classList.remove('text-muted');
            localBtn.classList.add('text-white');
            webBtn.style.background = 'transparent';
            webBtn.style.color = '';
            webBtn.classList.remove('text-white');
            webBtn.classList.add('text-muted');
            webPane.style.display = 'none';
            localPane.style.display = 'block';
            fetchMediaFiles();
        }
    }

    function fetchMediaFiles() {
        fetch('/api/list-media')
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    const select = document.getElementById('video-local-select');
                    select.innerHTML = '<option value="">-- Choose file --</option>';
                    if (data.files && data.files.length > 0) {
                        data.files.forEach(f => {
                            const sizeMB = (f.size / (1024 * 1024)).toFixed(1);
                            const opt = document.createElement('option');
                            opt.value = f.name;
                            opt.innerText = `${f.name} (${sizeMB} MB)`;
                            select.appendChild(opt);
                        });
                    } else {
                        const opt = document.createElement('option');
                        opt.value = '';
                        opt.innerText = 'No media files in raw/media/';
                        select.appendChild(opt);
                    }
                    updateLocalVideoBtnIcon();
                }
            })
            .catch(() => {});
    }

    function importVideoLink() {
        const url = document.getElementById('video-url-input').value.trim();
        if (!url) {
            showInfoModal("URL Required", "Please enter a valid YouTube/Bilibili video URL or Video ID.");
            return;
        }

        fetch('/api/import-video', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url, current_unit: currentUnit })
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                document.getElementById('video-url-input').value = '';
                pollJobStatus(data.job_id, 'compiler');
                showToast("Video import task queued successfully", "success");
            } else {
                showInfoModal("Import Error", "Video import task failed: " + (data.error || "Unknown error"));
            }
        })
        .catch(err => { showInfoModal("Network Error", "Import request failed: " + err.message); });
    }

    // ------------------ SIDEBAR UNIT GROUP COLLAPSE/EXPAND ------------------
    function toggleSidebarUnitGroup(el, event) {
        if (event) {
            event.preventDefault();
            event.stopPropagation();
        }
        
        // Determine what element type was clicked and find the group/content/arrow accordingly
        let group = null;
        let content = null;
        let icon = null;
        
        if (el.classList.contains('sidebar-unit-summary-arrow')) {
            // Clicked on the arrow wrapper span
            icon = el.querySelector('i');
            group = el.closest('.sidebar-unit-group');
        } else if (el.classList.contains('sidebar-unit-header') || el.classList.contains('sidebar-unit-summary')) {
            // Clicked on the header div (this in onclick context)
            group = el;
            content = el.nextElementSibling;
        } else if (el.tagName === 'I' && el.classList.contains('bi-chevron')) {
            // Clicked directly on the <i> icon
            icon = el;
            group = el.closest('.sidebar-unit-group');
        } else {
            // Fallback: try to find group from clicked element
            group = el.closest('.sidebar-unit-group');
        }
        
        if (!group) return;
        if (!content) content = group.querySelector('.sidebar-unit-content');
        if (!icon) icon = group.querySelector('.sidebar-unit-summary-arrow i');
        
        // Toggle content display
        const isHidden = content.classList.contains('d-none');
        if (isHidden) {
            content.classList.remove('d-none');
        } else {
            content.classList.add('d-none');
        }
        
        // Update arrow direction
        if (icon) {
            icon.className = isHidden ? 'bi bi-chevron-down' : 'bi bi-chevron-right';
        }
    }

    function transcribeLocalVideo() {
        const filename = document.getElementById('video-local-select').value;
        if (!filename) {
            showInfoModal("Media Selection Required", "Please select a local media file from the dropdown first.");
            return;
        }

        fetch('/api/import-video', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: filename, current_unit: currentUnit })
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                pollJobStatus(data.job_id, 'compiler');
                fetchMediaFiles(); // Refresh local list
                fetchCompiledUnits(); // Refresh library to show placeholder instantly!
                showToast("Transcription task queued successfully", "success");
            } else {
                showInfoModal("Transcription Error", "Local transcription failed to start: " + (data.error || "Unknown error"));
            }
        })
        .catch(err => { showInfoModal("Network Error", "Transcription request failed: " + err.message); });
    }

    function uploadLocalMediaAndTranscribe(fileOrEvent) {
        let file;
        let inputElement = null;
        if (fileOrEvent && fileOrEvent.target && fileOrEvent.target.files) {
            file = fileOrEvent.target.files[0];
            inputElement = fileOrEvent.target;
        } else {
            file = fileOrEvent;
        }
        if (!file) return;

        const filename = file.name;
        
        // Reset file input value immediately so that subsequent selects of the same file trigger the change event
        if (inputElement) {
            inputElement.value = "";
        }

        // Perform collision check first
        fetch('/api/check-upload-collision', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename, is_media: true, current_unit: currentUnit })
        })
        .then(res => res.json())
        .then(async collisionData => {
            if (collisionData.success && collisionData.collision) {
                let confirmMsg = "";
                if (collisionData.collision_type === "exact_match") {
                    confirmMsg = `A media file named '${collisionData.conflicting_file}' already exists. Do you want to overwrite it?`;
                } else if (collisionData.collision_type === "case_mismatch") {
                    confirmMsg = `A media file with mismatching casing '${collisionData.conflicting_file}' already exists. Overwriting will replace and delete the old file. Do you want to proceed?`;
                } else {
                    confirmMsg = `A media file with a different extension '${collisionData.conflicting_file}' already exists. Overwriting will replace and delete the old file. Do you want to proceed?`;
                }
                const userConfirmed = await showConfirmModal("Overwrite Media File?", confirmMsg, "Overwrite");
                if (!userConfirmed) {
                    return;
                }
            }
            
            // Proceed with actual upload
            proceedWithMediaUpload(file, event);
        })
        .catch(err => {
            console.error("Collision check failed:", err);
            // Proceed anyway on error as fallback
            proceedWithMediaUpload(file, event);
        });
    }

    function proceedWithMediaUpload(file, event) {
        const filename = file.name;
        
        // Disable file input and show loading status in dropdown or alert
        const select = document.getElementById('video-local-select');
        const originalHTML = select.innerHTML;
        select.innerHTML = `<option value="">Uploading ${filename}... 0%</option>`;
        select.disabled = true;

        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/api/upload-media', true);
        xhr.setRequestHeader('X-Filename', encodeURIComponent(filename));
        xhr.setRequestHeader('X-Current-Unit', encodeURIComponent(currentUnit || ''));
        xhr.setRequestHeader('Content-Type', 'application/octet-stream');

        xhr.upload.onprogress = function(e) {
            if (e.lengthComputable) {
                const percent = Math.round((e.loaded / e.total) * 100);
                select.innerHTML = `<option value="">Uploading ${filename}... ${percent}%</option>`;
            }
        };

        xhr.onload = function() {
            select.disabled = false;
            select.innerHTML = originalHTML;
            // Clear input
            if (event && event.target) {
                event.target.value = "";
            }
            if (xhr.status === 200) {
                try {
                    const res = JSON.parse(xhr.responseText);
                    if (res.success) {
                        // Start video-import job on the uploaded file
                        fetch('/api/import-video', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ url: res.filepath, current_unit: currentUnit })
                        })
                        .then(r => r.json())
                        .then(data => {
                            if (data.success) {
                                pollJobStatus(data.job_id, 'compiler');
                                fetchMediaFiles(); // Refresh local list
                                fetchCompiledUnits(); // Refresh library to show placeholder instantly!
                            } else {
                                alert("Failed to start transcription: " + data.error);
                            }
                        })
                        .catch(err => { alert('Transcription request failed: ' + err.message); });
                    } else {
                        alert("Upload failed: " + res.error);
                    }
                } catch(err) {
                    alert("Failed to parse response: " + err.message);
                }
            } else {
                alert("Upload failed with status " + xhr.status);
            }
        };

        xhr.onerror = function() {
            select.disabled = false;
            select.innerHTML = originalHTML;
            if (event && event.target) {
                event.target.value = "";
            }
            alert("Upload network error.");
        };

        xhr.send(file);
    }

    function handleLocalVideoAction() {
        const filename = document.getElementById('video-local-select').value;
        if (filename) {
            transcribeLocalVideo();
        } else {
            document.getElementById('local-video-file-input').click();
        }
    }

    function updateLocalVideoBtnIcon() {
        const filename = document.getElementById('video-local-select').value;
        const btn = document.getElementById('video-local-action-btn');
        if (!btn) return;
        if (filename) {
            btn.innerHTML = '<i class="bi bi-lightning-charge-fill"></i>';
            btn.title = "Transcribe Selected";
        } else {
            btn.innerHTML = '<i class="bi bi-upload"></i>';
            btn.title = "Upload File";
        }
    }

    // ------------------ DYNAMIC CANVAS FORCE NETWORK GALAXY MAP ------------------
    let currentGraphCategory = 'all';

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
        const box = canvas.parentElement;
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
        
        const rawFile = rawFiles.find(f => f.name.includes(baseName) || f.stem === baseName);
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
            renderMarkdownNode(`/wiki/${cleanPath}`, contentEl);
        } else {
            contentEl.innerHTML = '<p class="text-muted small">No direct preview is available for this virtual node.</p>';
        }
    }

    function zoomIn() { scale = Math.min(4.0, scale * 1.25); drawGraph(); }
    function zoomOut() { scale = Math.max(0.25, scale / 1.25); drawGraph(); }
    function zoomReset() { zoomToFit(); }

    function zoomToFit() {
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

    // Theme logic
    function toggleTheme() {
        const cur = document.documentElement.getAttribute('data-bs-theme') || 'dark';
        const target = cur === 'dark' ? 'light' : 'dark';
        
        // Trigger micro-animation on the icon
        const icon = document.getElementById('theme-icon');
        if (icon) {
            icon.classList.remove('theme-toggle-active');
            void icon.offsetWidth; // Force layout reflow to restart animation keyframes
            icon.classList.add('theme-toggle-active');
        }

        document.documentElement.setAttribute('data-bs-theme', target);
        localStorage.setItem('lexis-theme', target);
        updateThemeUI(target);
        drawGraph();
    }
    function updateThemeUI(theme) {
        const icon = document.getElementById('theme-icon');
        const text = document.getElementById('theme-text');
        if (theme === 'light') {
            if (icon) icon.className = 'bi bi-moon-stars-fill theme-toggle-active';
            if (text) text.innerText = 'Dark Mode';
        } else {
            if (icon) icon.className = 'bi bi-sun-fill theme-toggle-active';
            if (text) text.innerText = 'Light Mode';
        }
    }
    window.addEventListener('DOMContentLoaded', () => {
        const activeTheme = document.documentElement.getAttribute('data-bs-theme') || 'dark';
        updateThemeUI(activeTheme);
    });

    // 🧠 LEXIS ASK FLOATING DRAWER CONTROLLER
    let isAskOpen = false;
    function toggleLexisAsk(open) {
        const expanded = document.getElementById('lexis-ask-expanded');
        const collapsed = document.getElementById('lexis-ask-collapsed');
        if (open) {
            expanded.classList.remove('d-none');
            collapsed.classList.add('d-none');
            isAskOpen = true;
            fetch('/api/ask/stats')
                .then(res => res.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('lexis-ask-stats-badge').innerHTML = `<i class="bi bi-search me-1"></i>Searching ${data.node_count} nodes...`;
                    }
                });
        } else {
            expanded.classList.add('d-none');
            collapsed.classList.remove('d-none');
            isAskOpen = false;
        }
    }

    function submitLexisAsk(e) {
        e.preventDefault();
        const input = document.getElementById('lexis-ask-input');
        const chatLog = document.getElementById('lexis-ask-chat-log');
        const question = input.value.trim();
        if (!question) return;

        const userMsg = document.createElement('div');
        userMsg.className = 'align-self-end p-2 px-3 rounded text-white shadow-sm';
        userMsg.style.background = 'var(--primary)';
        userMsg.style.maxWidth = '85%';
        userMsg.innerText = question;
        chatLog.appendChild(userMsg);
        
        input.value = '';
        chatLog.scrollTop = chatLog.scrollHeight;

        const thinkingMsg = document.createElement('div');
        thinkingMsg.className = 'align-self-start p-2 px-3 rounded text-muted';
        thinkingMsg.style.background = 'rgba(255,255,255,0.02)';
        thinkingMsg.style.border = '1px solid var(--card-border)';
        thinkingMsg.style.maxWidth = '85%';
        thinkingMsg.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Thinking...';
        chatLog.appendChild(thinkingMsg);
        chatLog.scrollTop = chatLog.scrollHeight;

        fetch('/api/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        })
        .then(res => res.json())
        .then(data => {
            if (chatLog.contains(thinkingMsg)) chatLog.removeChild(thinkingMsg);
            const msg = document.createElement('div');
            msg.className = 'align-self-start p-3 rounded shadow-sm';
            msg.style.background = 'rgba(255, 255, 255, 0.03)';
            msg.style.border = '1px solid var(--card-border)';
            msg.style.maxWidth = '90%';
            msg.style.color = 'var(--on-surface)';
            
            if (data.success) {
                let formatted = data.answer
                    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
                    .replace(/\n/g, '<br>')
                    .replace(/\*\*(.*?)\*\"/g, '<strong class="fw-bold text-on-surface">$1</strong>')
                    .replace(/`([^`]+)`/g, '<code class="font-monospace p-1 rounded small" style="background: var(--input-bg); color: var(--accent-error); border: 1px solid var(--input-border);">$1</code>')
                    .replace(/\[\[(.*?)\]\]/g, '<span class="text-info text-decoration-underline fw-semibold" style="cursor: pointer;">$1</span>');
                msg.innerHTML = formatted;
            } else {
                msg.innerHTML = `<span class="text-danger"><i class="bi bi-exclamation-triangle-fill"></i> ${data.error}</span>`;
            }
            chatLog.appendChild(msg);
            chatLog.scrollTop = chatLog.scrollHeight;
        })
        .catch(err => {
            if (chatLog.contains(thinkingMsg)) chatLog.removeChild(thinkingMsg);
            const errMsg = document.createElement('div');
            errMsg.className = 'align-self-start p-2 px-3 rounded text-danger small';
            errMsg.style.border = '1px solid var(--card-border)';
            errMsg.innerText = 'Request failed: ' + err.message;
            chatLog.appendChild(errMsg);
            chatLog.scrollTop = chatLog.scrollHeight;
        });
    }

    function showToast(message, type = 'success') {
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

    document.addEventListener('click', (e) => {
        if (isAskOpen) {
            const container = document.getElementById('lexis-ask-container');
            if (container && !container.contains(e.target)) {
                toggleLexisAsk(false);
            }
        }
    });
