    // Note: esc, formatRelativeTime, showToast, modals, graph, hero_board, and glossary are owned by static/js/modules/*.js
    // State Machine
    let appConfig = {};
    let rawFiles = [];
    let compiledUnits = {};
    let currentUnit = null; // Name of the active Raw File Focus Workspace
    let activeJobId = null;
    let activePollTimers = {};
    let activePreviewPath = null;

    // Workspace subtab control
    let activeSubTab = 'raw';

    // Filter/Sort Ribbon State
    let libraryFilter = 'all';
    let librarySortOption = 'recent';
    let libraryViewMode = 'grid';
    let librarySearchQuery = '';
    let selectedSeries = 'all';
    let selectedBook = 'all';

    function getSeriesDisplayName(s) {
        if (!s || s === 'all') return 'All Series';
        if (s === 'standalone') return 'Standalone';
        if (Array.isArray(rawFiles)) {
            const match = rawFiles.find(f => f.series === s);
            if (match && match.series_display) return match.series_display;
        }
        if (s.includes('New Horizon College English')) return 'New Horizon · 4e';
        if (s.includes('New Practical English')) return 'Practical English';
        return s;
    }

    function setSeriesFilter(series) {
        selectedSeries = series;
        selectedBook = 'all';
        renderSeriesPillsBar();
        fetchRawFiles();
    }

    function setBookFilter(book) {
        selectedBook = book;
        renderSeriesPillsBar();
        fetchRawFiles();
    }

    function renderSeriesPillsBar() {
        const bar = document.getElementById('series-pills-bar');
        if (!bar) return;

        const seriesCatalog = {};
        let standaloneCount = 0;
        if (Array.isArray(rawFiles)) {
            rawFiles.forEach(f => {
                if (f.series) {
                    if (!seriesCatalog[f.series]) {
                        seriesCatalog[f.series] = {
                            books: new Set(),
                            display: f.series_display || f.series,
                            clean: f.series_clean || f.series,
                            edition: f.edition || null,
                            count: 0
                        };
                    }
                    seriesCatalog[f.series].count++;
                    if (f.book !== undefined && f.book !== null) {
                        seriesCatalog[f.series].books.add(f.book);
                    }
                } else {
                    standaloneCount++;
                }
            });
        }

        const seriesList = Object.keys(seriesCatalog).sort();

        let html = `
            <div class="dropdown">
                <button class="btn btn-sm btn-outline-secondary rounded-pill dropdown-toggle px-2.5 py-1 d-flex align-items-center" type="button" data-bs-toggle="dropdown" aria-expanded="false" style="font-size: 0.75rem; background: var(--input-bg); border-color: var(--card-border); color: var(--on-surface);">
                    <i class="bi bi-journal-bookmark-fill text-info me-2"></i>
                    <span class="fw-semibold text-truncate" style="max-width: 170px;">${esc(getSeriesDisplayName(selectedSeries))}</span>
                </button>
                <ul class="dropdown-menu dropdown-menu-end shadow-sm" style="font-size: 0.78rem; min-width: 210px;">
                    <li><a class="dropdown-item py-1.5 d-flex align-items-center ${selectedSeries === 'all' ? 'active' : ''}" href="#" onclick="setSeriesFilter('all'); return false;"><i class="bi bi-journals me-2 text-info" style="width: 16px; text-align: center;"></i><span>All Series</span></a></li>
        `;

        seriesList.forEach(s => {
            const meta = seriesCatalog[s] || {};
            const dispName = meta.display || getSeriesDisplayName(s);
            html += `<li><a class="dropdown-item py-1.5 d-flex align-items-center text-truncate ${selectedSeries === s ? 'active' : ''}" href="#" onclick="setSeriesFilter('${esc(s).replace(/'/g, "\\'")}'); return false;" title="${esc(s)}"><span class="text-truncate d-flex align-items-center"><i class="bi bi-journal-bookmark-fill me-2 text-info" style="width: 16px; text-align: center;"></i>${esc(dispName)}</span></a></li>`;
        });

        if (standaloneCount > 0) {
            html += `<li><hr class="dropdown-divider my-1"></li>`;
            html += `<li><a class="dropdown-item py-1.5 d-flex align-items-center ${selectedSeries === 'standalone' ? 'active' : ''}" href="#" onclick="setSeriesFilter('standalone'); return false;"><i class="bi bi-file-earmark-text-fill me-2 text-secondary" style="width: 16px; text-align: center;"></i><span>Standalone (${standaloneCount})</span></a></li>`;
        }

        html += `
                </ul>
            </div>
        `;

        if (selectedSeries !== 'all' && selectedSeries !== 'standalone' && seriesCatalog[selectedSeries]) {
            const books = Array.from(seriesCatalog[selectedSeries].books).sort((a, b) => {
                const na = Number(a), nb = Number(b);
                if (!isNaN(na) && !isNaN(nb)) return na - nb;
                return String(a).localeCompare(String(b));
            });

            html += `
                <div class="d-flex align-items-center gap-1 p-0.5 rounded-pill" style="background: var(--input-bg); border: 1px solid var(--card-border); height: 28px;">
                    <button class="filter-ribbon-btn ${selectedBook === 'all' ? 'active' : ''}" style="font-size: 0.72rem; padding: 2px 10px;" onclick="setBookFilter('all')">All Books</button>
            `;

            books.forEach(b => {
                const isActive = String(selectedBook) === String(b);
                html += `<button class="filter-ribbon-btn ${isActive ? 'active' : ''}" style="font-size: 0.72rem; padding: 2px 10px;" onclick="setBookFilter('${esc(String(b))}');">Book ${esc(String(b))}</button>`;
            });

            html += `</div>`;

            const activeBook = (selectedBook !== 'all' ? selectedBook : (books[0] || '1'));
            html += `
                <button class="btn btn-sm btn-outline-info rounded-pill px-2.5 py-1 d-inline-flex align-items-center ms-1" onclick="openBookGlossaryModal('${esc(selectedSeries)}', '${esc(String(activeBook))}'); return false;" title="Open Book ${esc(String(activeBook))} Glossary" style="font-size: 0.72rem; height: 28px;">
                    <i class="bi bi-book-half me-1.5 text-info"></i>
                    <span class="fw-semibold">Book ${esc(String(activeBook))} Glossary</span>
                </button>
            `;
        }

        bar.innerHTML = html;
    }

    function onLibrarySearchInput(val) {
        librarySearchQuery = (val || '').trim().toLowerCase();
        const clearBtn = document.getElementById('library-search-clear');
        if (clearBtn) {
            if (librarySearchQuery) clearBtn.classList.remove('d-none');
            else clearBtn.classList.add('d-none');
        }
        fetchRawFiles();
    }

    function clearLibrarySearch() {
        const input = document.getElementById('library-search-input');
        if (input) input.value = '';
        librarySearchQuery = '';
        const clearBtn = document.getElementById('library-search-clear');
        if (clearBtn) clearBtn.classList.add('d-none');
        fetchRawFiles();
    }

    function onLibrarySearchKeydown(event) {
        if (event.key === 'Escape') {
            clearLibrarySearch();
            const input = document.getElementById('library-search-input');
            if (input) input.blur();
        }
    }

    // Global keyboard shortcut: '/' or 'Ctrl+K' focuses search input
    document.addEventListener('keydown', (e) => {
        if ((e.key === '/' || (e.ctrlKey && e.key.toLowerCase() === 'k')) && !['INPUT', 'TEXTAREA'].includes(document.activeElement?.tagName)) {
            const input = document.getElementById('library-search-input');
            if (input) {
                e.preventDefault();
                input.focus();
                input.select();
            }
        }
    });

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
                const mdContainer = document.getElementById('markdown-preview-container');
                if (mdContainer) {
                    mdContainer.querySelectorAll('video, audio').forEach(el => {
                        try { el.pause(); el.removeAttribute('src'); el.load(); } catch (_) {}
                    });
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
                    const cleanPath = mPath.replace(/\\/g, '/').replace(/^\/+/, '');
                    const parts = cleanPath.split('/');
                    const wikiIdx = parts.indexOf('wiki');
                    const mUnit = (wikiIdx !== -1 && wikiIdx + 1 < parts.length) ? parts[wikiIdx + 1] : (parts.length > 1 ? parts[0] : '');
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
                    const cleanPath = sPath.replace(/\\/g, '/').replace(/^\/+/, '');
                    const parts = cleanPath.split('/');
                    const wikiIdx = parts.indexOf('wiki');
                    const sUnit = (wikiIdx !== -1 && wikiIdx + 1 < parts.length) ? parts[wikiIdx + 1] : (parts.length > 1 ? parts[0] : '');
                    if (sUnit.toLowerCase() === baseName.toLowerCase() && cleanPath.endsWith('.md')) {
                        const sFileName = cleanPath.split('/').pop();
                        const sTitle = sFileName.replace(/\.[^/.]+$/, "").replace(/_/g, " ");
                        const sUrl = cleanPath.startsWith('wiki/') ? `/${cleanPath}` : `/wiki/${cleanPath}`;
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
                    let leadingSpaces = (line.match(/^(\s*)/) || ['', ''])[1].replace(/\t/g, '    ').length;
                    let indentRem = leadingSpaces > 0 ? Math.min(leadingSpaces * 0.45, 6) : 0;
                    let indentStyle = indentRem > 0 ? `padding-left: ${indentRem.toFixed(2)}rem;` : '';

                    if (trimmed.startsWith('# ')) {
                        return `<h1 class="h4 fw-bold border-bottom pb-2 mb-3 mt-2">${trimmed.substring(2)}</h1>`;
                    } else if (trimmed.startsWith('## ')) {
                        let h2Body = trimmed.substring(3);
                        let hwMatch = h2Body.match(/^\[\[(.*?)\]\](.*)$/);
                        if (hwMatch) {
                            let hw = hwMatch[1];
                            let tail = hwMatch[2].trim();
                            if (tail) {
                                let gramTags = [];
                                let tailNoGram = tail.replace(/`\[([^\]]+)\]`/g, (m) => {
                                    gramTags.push(m);
                                    return '';
                                }).trim();
                                let cefrBadge = '';
                                let tailNoCefr = tailNoGram.replace(/\b(A1|A2|B1|B2|C1|C2)\b/g, (m) => {
                                    cefrBadge = `<span class="lex-pill lex-pill-cefr">${m}</span>`;
                                    return '';
                                }).trim();
                                let posBadge = tailNoCefr ? `<span class="lex-pill lex-pill-pos">${tailNoCefr}</span>` : '';
                                let attrParts = [posBadge, cefrBadge, ...gramTags].filter(Boolean);
                                return `<h2 class="h5 fw-bold text-info mt-3 mb-2 d-flex align-items-center flex-wrap" style="gap: 0.5rem;"><span class="me-1">[[${hw}]]</span> ${attrParts.join(' ')}</h2>`;
                            }
                        }
                        return `<h2 class="h5 fw-bold text-info mt-3 mb-2">${h2Body}</h2>`;
                    } else if (trimmed.startsWith('### ')) {
                        return `<h3 class="h6 fw-bold text-success mt-2 mb-2">${trimmed.substring(4)}</h3>`;
                    } else if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
                        let bulletPadding = indentRem > 0 ? `style="padding-left: ${(0.5 + indentRem).toFixed(2)}rem;"` : 'class="ps-2"';
                        return `<div class="d-flex align-items-start gap-1.5 mb-1" ${bulletPadding}><span class="text-info fw-bold lh-1 me-1">- </span><div class="small">${trimmed.substring(2)}</div></div>`;
                    } else if (trimmed === '' || trimmed === '&nbsp;') {
                        return '';
                    } else if (trimmed.startsWith('✗')) {
                        return `<div class="mb-1 small" style="line-height: 1.55; ${indentStyle}"><span class="text-danger fw-bold me-1">✗</span><span class="text-on-surface-variant">${trimmed.substring(1).trim()}</span></div>`;
                    } else if (trimmed.startsWith('✓')) {
                        return `<div class="mb-1 small" style="line-height: 1.55; ${indentStyle}"><span class="text-success fw-bold me-1">✓</span><span class="text-on-surface-variant">${trimmed.substring(1).trim()}</span></div>`;
                    } else {
                        let mbClass = indentRem > 0 ? 'mb-1' : 'mb-2';
                        return `<p class="${mbClass} text-on-surface-variant small" style="line-height: 1.55; ${indentStyle}">${trimmed}</p>`;
                    }
                });
                let html = processedLines.join('\n');
                html = html
                    // Convert backtick-wrapped bracket tags like `[countable, uncountable]` into soft grammar pills
                    .replace(/`\[([^\]]+)\]`/g, (_, inner) => {
                        return inner.split(/\s*,\s*/).map(tag => `<span class="lex-pill lex-pill-grammar">${tag.trim()}</span>`).join(' ');
                    })
                    // Convert single-bracket register tags like [formal], [spoken], [American English], [especially British English] into soft register pills (never match [[wikilinks]])
                    .replace(/(?<!\[)\[((?:(?:especially\s+)?(?:British|American|Australian)\s+English|(?:formal|informal|spoken|written|literary|humorous|old-fashioned|technical|law|medical|business|computing)(?:\s+(?:formal|informal|spoken|written|literary|humorous))?))\](?!\])/gi, (match, inner) => {
                        const clean = inner.trim();
                        if (!clean) return match;
                        return `<span class="lex-pill lex-pill-register">${clean}</span>`;
                    })
                    .replace(/\*\*(.*?)\*\*/g, '<strong class="fw-bold text-on-surface">$1</strong>')
                    .replace(/\*(.*?)\*/g, '<em class="fst-italic">$1</em>')
                    .replace(/`([^`]+)`/g, '<code class="lex-inline-code">$1</code>')
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
        } else if (/\.(mp4|webm|mkv|mov)$/i.test(cleanPath)) {
            if (previewFrame) {
                previewFrame.classList.add('d-none');
                previewFrame.src = '';
            }
            if (mdContainer) {
                mdContainer.classList.remove('d-none');
                mdContainer.innerHTML = `<div class="p-3 d-flex justify-content-center align-items-center" style="min-height: 60vh;"><video src="${cleanPath}" controls preload="metadata" style="width: 100%; max-height: 78vh; border-radius: 0.5rem; background: #000;"></video></div>`;
            }
        } else {
            if (mdContainer) {
                mdContainer.classList.add('d-none');
                mdContainer.innerHTML = '';
            }
            if (previewFrame) {
                previewFrame.classList.remove('d-none');
                const sep = cleanPath.includes('?') ? '&' : '?';
                previewFrame.src = cleanPath.endsWith('.html') ? `${cleanPath}${sep}v=${Date.now()}` : cleanPath;
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
            else if (lower.includes('cloze')) iconClass = 'bi-input-cursor-text';
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
                        <button class="btn btn-sm btn-outline-warning py-1 px-2.5" style="font-size: 0.75rem;" onclick="bundleStandaloneExe('${esc(cleanPath)}')" title="Bundle HTML handout + local video into a portable ZIP">
                            <i class="bi bi-file-earmark-zip me-1"></i> Bundle ZIP
                        </button>
                    `;
                } else {
                    buttonsHtml += `
                        <a href="${cleanPath}" download="${cleanPath.split('/').pop()}" class="btn btn-sm btn-outline-success py-1 px-2.5" style="font-size: 0.75rem;">
                            <i class="bi bi-download me-1"></i> Save .HTML
                        </a>
                    `;
                }
            } else if (cleanPath.endsWith('.md')) {
                buttonsHtml += `
                    <button class="btn btn-sm btn-outline-success py-1 px-2.5" style="font-size: 0.75rem;" onclick="exportMarkdownPathAsHtml('${esc(cleanPath)}')">
                        <i class="bi bi-filetype-html me-1"></i> Export .HTML
                    </button>
                `;
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

    function exportActiveWorkspaceTabHtml() {
        const cat = getActiveSubTabCategory();
        if (cat === 'mindmap') {
            downloadMindmap();
            return;
        }
        if (cat === 'raw') {
            exportRawSourceHtml();
            return;
        }
        exportActiveNodeHtml(cat);
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

    function triggerHtmlBlobDownload(htmlString, fileName) {
        const blob = new Blob([htmlString], { type: 'text/html;charset=utf-8' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = fileName;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(a.href);
        showToast(`Exported ${fileName}`, 'success');
    }

    function exportActiveNodeHtml(category) {
        if (!currentUnit) {
            showToast('No active unit selected.', 'error');
            return;
        }
        const baseName = currentUnit.replace(/\.[^/.]+$/, "");
        const filePath = `/wiki/${baseName}/extractions/${baseName}_${category}.md`;
        exportMarkdownPathAsHtml(filePath, `${baseName}_${category}.html`, category);
    }

    function exportMarkdownPathAsHtml(filePath, customOutName = '', categoryHint = '') {
        const rawFileName = filePath.split('/').pop() || 'handout.md';
        const stem = rawFileName.replace(/\.md$/i, '');
        const outFileName = customOutName || `${stem}.html`;
        let inferredCat = categoryHint;
        if (!inferredCat) {
            const lower = stem.toLowerCase();
            if (lower.endsWith('_vocabulary')) inferredCat = 'vocabulary';
            else if (lower.endsWith('_grammar')) inferredCat = 'grammar';
            else if (lower.endsWith('_summary')) inferredCat = 'summary';
            else inferredCat = 'source';
        }

        fetch(filePath)
            .then(res => {
                if (!res.ok) throw new Error('Markdown file not found. Run extraction first.');
                return res.text();
            })
            .then(mdContent => {
                const htmlDoc = buildStandaloneExtractionHtml(mdContent, stem.replace(/_/g, ' '), inferredCat);
                triggerHtmlBlobDownload(htmlDoc, outFileName);
            })
            .catch(err => showToast(`HTML export failed: ${err.message}`, 'error'));
    }

    function exportRawSourceHtml() {
        if (!currentUnit) {
            showToast('No active unit selected.', 'error');
            return;
        }
        const baseName = currentUnit.replace(/\.[^/.]+$/, "");
        const fileName = currentUnit;
        const filePath = `/wiki/${baseName}/sources/${fileName}`;

        fetch(filePath)
            .then(res => {
                if (!res.ok) throw new Error('Source file not found');
                return res.text();
            })
            .then(text => {
                // Strip internal <syllabus> block from student-facing source reading handout if present
                const cleanSource = text.replace(/<syllabus>[\s\S]*?<\/syllabus>/gi, '').trim();
                const htmlDoc = buildStandaloneExtractionHtml(cleanSource, `${baseName.replace(/_/g, ' ')} — Source Text`, 'source');
                triggerHtmlBlobDownload(htmlDoc, `${baseName}_source.html`);
            })
            .catch(err => showToast(`HTML export failed: ${err.message}`, 'error'));
    }

    function formatInlineLexisMarkdown(lineHtml) {
        return lineHtml
            .replace(/`\[([^\]]+)\]`/g, (_, inner) => {
                return inner.split(/\s*,\s*/).map(tag => `<span class="lex-pill lex-pill-grammar">${tag.trim()}</span>`).join(' ');
            })
            .replace(/(?<!\[)\[((?:(?:especially\s+)?(?:British|American|Australian)\s+English|(?:formal|informal|spoken|written|literary|humorous|old-fashioned|technical|law|medical|business|computing)(?:\s+(?:formal|informal|spoken|written|literary|humorous))?))\](?!\])/gi, (match, inner) => {
                const clean = inner.trim();
                if (!clean) return match;
                return `<span class="lex-pill lex-pill-register">${clean}</span>`;
            })
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`([^`]+)`/g, '<code class="lex-inline-code">$1</code>')
            .replace(/\[\[(.*?)\]\]/g, '<span class="lex-headword-link">$1</span>');
    }

    function buildStandaloneExtractionHtml(mdContent, fallbackTitle, category) {
        let bodyContent = mdContent || '';
        const meta = {
            title: fallbackTitle || 'Lexis Wiki Handout',
            cefr: '',
            itemCount: '',
            categoryLabel: (category || 'handout').charAt(0).toUpperCase() + (category || 'handout').slice(1)
        };

        const fmMatch = bodyContent.match(/^---\r?\n([\s\S]*?)\r?\n---/);
        if (fmMatch) {
            bodyContent = bodyContent.slice(fmMatch[0].length).trim();
            const fmText = fmMatch[1];
            const titleM = fmText.match(/^title:\s*"?([^"\r\n]+)"?/m);
            const cefrM = fmText.match(/^overall_cefr_level:\s*"?([^"\r\n]+)"?/m);
            const countM = fmText.match(/^item_count:\s*(\d+)/m);
            if (titleM && titleM[1].trim()) meta.title = titleM[1].trim();
            if (cefrM && cefrM[1].trim()) meta.cefr = cefrM[1].trim();
            if (countM && countM[1].trim()) meta.itemCount = countM[1].trim();
        }

        const lines = bodyContent.split(/\r?\n/);
        let docHeading = '';
        const sections = [];
        let currentCard = null;
        let introLines = [];

        const flushCard = () => {
            if (currentCard) {
                sections.push(`<section class="lex-entry-card">${currentCard.headerHtml}<div class="lex-entry-body">${currentCard.bodyLines.join('\n')}</div></section>`);
                currentCard = null;
            }
        };

        lines.forEach(rawLine => {
            const trimmed = rawLine.trim();
            const leadingSpaces = (rawLine.match(/^(\s*)/) || ['', ''])[1].replace(/\t/g, '    ').length;
            const indentRem = leadingSpaces > 0 ? Math.min(leadingSpaces * 0.4, 5) : 0;
            const indentStyle = indentRem > 0 ? `padding-left: ${indentRem.toFixed(2)}rem;` : '';

            if (trimmed.startsWith('# ')) {
                if (!docHeading) {
                    docHeading = trimmed.substring(2).trim();
                } else {
                    flushCard();
                    introLines.push(`<h2 class="lex-section-title">${formatInlineLexisMarkdown(esc(trimmed.substring(2).trim()))}</h2>`);
                }
                return;
            }

            if (trimmed.startsWith('## ')) {
                flushCard();
                const h2Body = trimmed.substring(3).trim();
                const hwMatch = h2Body.match(/^\[\[(.*?)\]\](.*)$/);
                let headerHtml = '';
                if (hwMatch) {
                    const hw = esc(hwMatch[1]);
                    const tail = hwMatch[2].trim();
                    let attrParts = [];
                    if (tail) {
                        const gramTags = [];
                        const tailNoGram = tail.replace(/`\[([^\]]+)\]`/g, (_, inner) => {
                            inner.split(/\s*,\s*/).forEach(t => {
                                if (t.trim()) gramTags.push(`<span class="lex-pill lex-pill-grammar">${esc(t.trim())}</span>`);
                            });
                            return '';
                        }).trim();
                        let cefrBadge = '';
                        const tailNoCefr = tailNoGram.replace(/\b(A1|A2|B1|B2|C1|C2)\b/g, (m) => {
                            cefrBadge = `<span class="lex-pill lex-pill-cefr">${m}</span>`;
                            return '';
                        }).trim();
                        const posBadge = tailNoCefr ? `<span class="lex-pill lex-pill-pos">${esc(tailNoCefr)}</span>` : '';
                        attrParts = [posBadge, cefrBadge, ...gramTags].filter(Boolean);
                    }
                    headerHtml = `<h2 class="lex-entry-title"><span class="lex-headword">${hw}</span>${attrParts.length ? ' <span class="lex-badges">' + attrParts.join(' ') + '</span>' : ''}</h2>`;
                } else {
                    headerHtml = `<h2 class="lex-entry-title"><span class="lex-headword">${formatInlineLexisMarkdown(esc(h2Body))}</span></h2>`;
                }
                currentCard = { headerHtml, bodyLines: [] };
                return;
            }

            if (trimmed === '' || trimmed === '&nbsp;') {
                return;
            }

            let renderedLine = '';
            if (trimmed.startsWith('### ')) {
                renderedLine = `<h3 class="lex-subheading">${formatInlineLexisMarkdown(esc(trimmed.substring(4)))}</h3>`;
            } else if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
                const bulletStyle = indentRem > 0 ? `style="padding-left: ${(0.25 + indentRem).toFixed(2)}rem;"` : '';
                renderedLine = `<div class="lex-bullet-row" ${bulletStyle}><span class="lex-bullet-dot">&bull;</span><div class="lex-bullet-text">${formatInlineLexisMarkdown(esc(trimmed.substring(2)))}</div></div>`;
            } else if (trimmed.startsWith('✗')) {
                renderedLine = `<div class="lex-contrast-row lex-wrong" style="${indentStyle}"><span class="lex-mark-wrong">&#10007;</span><span>${formatInlineLexisMarkdown(esc(trimmed.substring(1).trim()))}</span></div>`;
            } else if (trimmed.startsWith('✓')) {
                renderedLine = `<div class="lex-contrast-row lex-right" style="${indentStyle}"><span class="lex-mark-right">&#10003;</span><span>${formatInlineLexisMarkdown(esc(trimmed.substring(1).trim()))}</span></div>`;
            } else {
                renderedLine = `<p class="lex-paragraph" style="${indentStyle}">${formatInlineLexisMarkdown(esc(trimmed))}</p>`;
            }

            if (currentCard) {
                currentCard.bodyLines.push(renderedLine);
            } else {
                introLines.push(renderedLine);
            }
        });

        flushCard();

        const displayHeading = docHeading || `${meta.categoryLabel}: ${meta.title}`;
        const metaBadges = [];
        metaBadges.push(`<span class="meta-badge meta-cat">${esc(meta.categoryLabel)}</span>`);
        if (meta.cefr) metaBadges.push(`<span class="meta-badge meta-cefr">CEFR ${esc(meta.cefr)}</span>`);
        if (meta.itemCount) metaBadges.push(`<span class="meta-badge meta-count">${esc(meta.itemCount)} Items</span>`);

        const introCardHtml = introLines.length
            ? `<section class="lex-entry-card lex-prose-card"><div class="lex-entry-body">${introLines.join('\n')}</div></section>`
            : '';

        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${esc(displayHeading)}</title>
    <style>
        :root {
            --bg-page: #f8fafc;
            --bg-card: #ffffff;
            --text-main: #0f172a;
            --text-secondary: #334155;
            --text-muted: #64748b;
            --border-card: #e2e8f0;
            --accent-primary: #0284c7;
            --accent-navy: #0369a1;
            --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            --font-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            padding: 2.25rem 1.25rem 3.5rem;
            background-color: var(--bg-page);
            color: var(--text-main);
            font-family: var(--font-sans);
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
        }
        .handout-container {
            max-width: 880px;
            margin: 0 auto;
        }
        .handout-header {
            background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
            border: 1px solid var(--border-card);
            border-left: 5px solid var(--accent-primary);
            border-radius: 12px;
            padding: 1.4rem 1.75rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
        }
        .handout-title {
            margin: 0;
            font-size: 1.4rem;
            font-weight: 700;
            color: var(--text-main);
            letter-spacing: -0.015em;
        }
        .handout-meta {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        .meta-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .meta-cat {
            background: rgba(2, 132, 199, 0.1);
            color: #0284c7;
            border: 1px solid rgba(2, 132, 199, 0.25);
        }
        .meta-cefr {
            background: rgba(5, 150, 105, 0.1);
            color: #059669;
            border: 1px solid rgba(5, 150, 105, 0.25);
            font-family: var(--font-mono);
        }
        .meta-count {
            background: #f1f5f9;
            color: #475569;
            border: 1px solid #cbd5e1;
        }
        .lex-entry-card {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 10px;
            padding: 1.15rem 1.45rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
            break-inside: avoid;
            page-break-inside: avoid;
        }
        .lex-prose-card {
            padding: 1.5rem 1.75rem;
        }
        .lex-entry-title {
            margin: 0 0 0.65rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #f1f5f9;
            font-size: 1.12rem;
            font-weight: 700;
            color: var(--accent-navy);
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.45rem;
        }
        .lex-headword {
            color: var(--accent-navy);
        }
        .lex-badges {
            display: inline-flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.35rem;
        }
        .lex-pill {
            display: inline-flex;
            align-items: center;
            padding: 0.1rem 0.52rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            line-height: 1.35;
            white-space: nowrap;
            vertical-align: baseline;
        }
        .lex-pill-pos {
            background: rgba(2, 132, 199, 0.09);
            color: #0284c7;
            border: 1px solid rgba(2, 132, 199, 0.25);
            font-style: italic;
        }
        .lex-pill-cefr {
            background: rgba(5, 150, 105, 0.1);
            color: #059669;
            border: 1px solid rgba(5, 150, 105, 0.25);
            font-family: var(--font-mono);
            font-size: 0.68rem;
        }
        .lex-pill-grammar {
            background: rgba(109, 40, 217, 0.08);
            color: #6d28d9;
            border: 1px solid rgba(109, 40, 217, 0.24);
        }
        .lex-pill-register {
            background: rgba(217, 119, 6, 0.1);
            color: #b45309;
            border: 1px solid rgba(217, 119, 6, 0.28);
            font-size: 0.68rem;
            margin-right: 0.25rem;
        }
        .lex-inline-code {
            font-family: var(--font-mono);
            font-size: 0.8rem;
            padding: 0.08rem 0.38rem;
            border-radius: 4px;
            background: #f1f5f9;
            color: #0369a1;
            border: 1px solid #e2e8f0;
        }
        .lex-headword-link {
            color: #0284c7;
            font-weight: 600;
        }
        .lex-subheading {
            margin: 0.75rem 0 0.35rem;
            font-size: 0.95rem;
            font-weight: 700;
            color: #0f766e;
        }
        .lex-bullet-row {
            display: flex;
            align-items: flex-start;
            gap: 0.45rem;
            margin-bottom: 0.32rem;
            font-size: 0.92rem;
            color: var(--text-secondary);
        }
        .lex-bullet-dot {
            color: var(--accent-primary);
            font-weight: 700;
            line-height: 1.5;
            user-select: none;
        }
        .lex-bullet-text {
            flex: 1;
        }
        .lex-bullet-text strong, .lex-paragraph strong {
            color: var(--text-main);
            font-weight: 600;
        }
        .lex-contrast-row {
            display: flex;
            align-items: flex-start;
            gap: 0.4rem;
            margin: 0.22rem 0;
            font-size: 0.89rem;
            color: var(--text-secondary);
        }
        .lex-mark-wrong {
            color: #dc2626;
            font-weight: 700;
            flex-shrink: 0;
        }
        .lex-mark-right {
            color: #16a34a;
            font-weight: 700;
            flex-shrink: 0;
        }
        .lex-paragraph {
            margin: 0.35rem 0 0.55rem;
            font-size: 0.93rem;
            color: var(--text-secondary);
        }
        .lex-prose-card .lex-paragraph {
            font-size: 0.98rem;
            line-height: 1.72;
            margin-bottom: 0.85rem;
        }
        @media print {
            body { background: #ffffff; padding: 0; }
            .handout-container { max-width: 100%; }
            .handout-header, .lex-entry-card { box-shadow: none; }
        }
    </style>
</head>
<body>
    <main class="handout-container">
        <header class="handout-header">
            <h1 class="handout-title">${esc(displayHeading)}</h1>
            <div class="handout-meta">${metaBadges.join('')}</div>
        </header>
        ${introCardHtml}
        ${sections.join('\n')}
    </main>
</body>
</html>`;
    }

    function bundleStandaloneExe(htmlPath) {
        showToast('Bundling portable ZIP (HTML + Video)...', 'info');
        fetch('/api/export-exe', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ html_path: htmlPath })
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                showToast(data.message || 'Portable ZIP bundle ready!', 'success');
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

    function closeAllDropdowns() {
        document.querySelectorAll('.dropdown-menu.show').forEach(menu => {
            const toggle = menu.closest('.dropdown')?.querySelector('[data-bs-toggle="dropdown"]');
            if (toggle && typeof bootstrap !== 'undefined' && bootstrap.Dropdown) {
                const inst = bootstrap.Dropdown.getInstance(toggle) || (bootstrap.Dropdown.getOrCreateInstance ? bootstrap.Dropdown.getOrCreateInstance(toggle) : null);
                if (inst && typeof inst.hide === 'function') {
                    inst.hide();
                }
            }
            menu.classList.remove('show');
            if (toggle) {
                toggle.classList.remove('show');
                toggle.setAttribute('aria-expanded', 'false');
            }
        });
    }
    window.closeAllDropdowns = closeAllDropdowns;

    function selectCompileCategory(category) {
        selectedCompileCategory = category ? category.trim() : null;
        updateCompileButtonUI();
        closeAllDropdowns();
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

    function triggerUnitCompileAll(unitName) {
        closeAllDropdowns();
        if (!unitName) return;
        showToast(`Compiling all extractions for ${unitName}...`, 'info');
        fetch('/api/compile', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: unitName })
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                pollJobStatus(data.job_id, 'compiler');
                fetchRecentJobs();
                showToast(`Extraction compilation queued for ${unitName}`, 'success');
            } else {
                showInfoModal("Compilation Failed", "Failed to start extraction compilation: " + (data.error || "Unknown error"));
            }
        })
        .catch(err => {
            showInfoModal("Network Error", "Failed to trigger compilation: " + err.message);
        });
    }

    function triggerUnitGenerateAllQuizzes(unitName) {
        closeAllDropdowns();
        if (!unitName) return;
        showToast(`Generating all quizzes for ${unitName}...`, 'info');
        fetch('/api/generate-all-quizzes', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: unitName })
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                pollJobStatus(data.job_id, 'all-quizzes');
                fetchRecentJobs();
                showToast(`Quiz batch generation queued for ${unitName}`, 'success');
            } else {
                showInfoModal("Quiz Generation Failed", "Failed to start quiz batch generation: " + (data.error || "Unknown error"));
            }
        })
        .catch(err => {
            showInfoModal("Network Error", "Failed to trigger quiz batch generation: " + err.message);
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
    // ------------------ SETTINGS OVERLAY MODAL ------------------
    function openSettingsModal() {
        fetchConfig();
        const modal = document.getElementById('settings-modal');
        const modalInstance = bootstrap.Modal.getOrCreateInstance(modal);
        modalInstance.show();
    }
    function closeSettingsModal() {
        stopTTSVoicePreview();
        const modal = document.getElementById('settings-modal');
        const modalInstance = bootstrap.Modal.getInstance(modal);
        if (modalInstance) modalInstance.hide();
    }
    // Quiz Defaults tab: live "Total questions" counter.
    // Delegated on document so it works regardless of how the tile grid is rendered.
    function refreshQuizTotal() {
        const totalEl = document.getElementById('config-quiz-total-val');
        if (!totalEl) return;
        let total = 0;
        ['vocabulary', 'cloze', 'reading', 'translation', 'listening', 'video'].forEach(function (key) {
            const el = document.getElementById('config-quiz-' + key);
            if (el) total += parseInt(el.value, 10) || 0;
        });
        totalEl.textContent = 'Total: ' + total + (total === 1 ? ' question' : ' questions');
    }

    document.addEventListener('input', function (e) {
        const id = (e.target && e.target.id) || '';
        if (id.indexOf('config-quiz-') === 0) {
            refreshQuizTotal();
        }
    });

    function saveGlobalConfigurationAndClose() {
        stopTTSVoicePreview();
        // Collect model parameters
        const api_type = document.getElementById('config-api-type').value;
        const model = document.getElementById('config-active-model').value;
        const api_url = (document.getElementById('config-api-url').value || '').trim().replace(/\/+$/, '');
        const api_key = document.getElementById('config-api-key').value.trim();

        const compile_defaults = {
            vocabulary: parseInt(document.getElementById('config-vocab-count').value),
            grammar: parseInt(document.getElementById('config-grammar-count').value),
            concepts: parseInt(document.getElementById('config-concept-count').value),
            max_parallel: parseInt(document.getElementById('config-max-p').value)
        };

        const quiz_defaults = {
            vocabulary: {
                count: parseInt(document.getElementById('config-quiz-vocabulary').value) || 0
            },
            cloze: {
                count: parseInt(document.getElementById('config-quiz-cloze').value) || 0
            },
            reading: {
                count: parseInt(document.getElementById('config-quiz-reading').value) || 0
            },
            translation: {
                count: parseInt(document.getElementById('config-quiz-translation').value) || 0
            },
            listening: {
                count: parseInt(document.getElementById('config-quiz-listening').value) || 0
            },
            video: {
                count: parseInt(document.getElementById('config-quiz-video').value) || 0
            }
        };

        const tts_engine = document.getElementById('config-tts-engine').value;
        const tts_voice_a = document.getElementById('config-tts-voice-a').value;
        const tts_voice_b = document.getElementById('config-tts-voice-b').value;
        const target_language = document.getElementById('config-target-language').value;
        const enable_expert_audit = document.getElementById('config-enable-expert-audit').checked;
        const judge_model = document.getElementById('config-judge-model').value.trim();
        const export_dir = (document.getElementById('config-export-dir')?.value || '').trim();

        fetch('/api/config/update', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                api_type, model, api_url, api_key,
                compile_defaults, quiz_defaults,
                tts_engine, tts_voice_a, tts_voice_b, target_language,
                enable_expert_audit, judge_model, export_dir
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
                if (data.tts_voices) {
                    ttsVoiceCatalogs = data.tts_voices;
                }
                
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
                const parseQCount = (val, def) => {
                    if (val && typeof val === 'object' && val.count !== undefined) return val.count;
                    if (typeof val === 'number') return val;
                    return def;
                };

                const qVocab = parseQCount(qDefaults.vocabulary, 10);
                const qCloze = parseQCount(qDefaults.cloze, 10);
                const qReading = parseQCount(qDefaults.reading, 5);
                const qTranslation = parseQCount(qDefaults.translation, 5);
                const qListening = parseQCount(qDefaults.listening, 5);
                const qVideo = parseQCount(qDefaults.video, 5);

                document.getElementById('config-quiz-vocabulary').value = qVocab;
                document.getElementById('config-quiz-vocabulary-val').innerText = qVocab;
                document.getElementById('config-quiz-cloze').value = qCloze;
                document.getElementById('config-quiz-cloze-val').innerText = qCloze;
                document.getElementById('config-quiz-reading').value = qReading;
                document.getElementById('config-quiz-reading-val').innerText = qReading;
                document.getElementById('config-quiz-translation').value = qTranslation;
                document.getElementById('config-quiz-translation-val').innerText = qTranslation;
                document.getElementById('config-quiz-listening').value = qListening;
                document.getElementById('config-quiz-listening-val').innerText = qListening;
                document.getElementById('config-quiz-video').value = qVideo;
                document.getElementById('config-quiz-video-val').innerText = qVideo;

                document.getElementById('config-tts-engine').value = appConfig.tts_engine || 'edge';
                document.getElementById('config-target-language').value = appConfig.target_language || 'Chinese';

                // Level 2 Expert Audit options
                document.getElementById('config-enable-expert-audit').checked = !!appConfig.enable_expert_audit;
                document.getElementById('config-judge-model').value = appConfig.judge_model || '';

                if (document.getElementById('config-export-dir')) {
                    document.getElementById('config-export-dir').value = appConfig.export_dir || '';
                }

                // Refresh the "Total questions" badge to match the loaded config
                refreshQuizTotal();


                // Load API type first (sets up model field visibility)
                onApiTypeChange(false);

                // Populate Female & Male voice dropdowns extracted from Python TTS controls
                onTTSEngineChange(appConfig.tts_voice_a, appConfig.tts_voice_b);

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

    let lastLlmHealth = null;
    function updateLlmHealthUI(health) {
        if (!health) return;
        lastLlmHealth = health;
        const dotEl = document.getElementById('headbar-status-dot');
        const askDotEl = document.getElementById('lexis-ask-status-dot');
        const nameEl = document.getElementById('headbar-model-name');
        const pillEl = document.getElementById('headbar-status-pill');

        const activeModel = health.active_model || (appConfig && appConfig.model) || '';
        let color = '#10b981'; // Green
        let label = activeModel || 'Ready';
        let tooltip = `LLM Online & Ready (${activeModel} @ ${health.api_url || 'local'})`;

        if (!health.online) {
            color = '#ef4444'; // Red
            label = activeModel ? `LLM Offline (${activeModel})` : 'LLM Offline';
            tooltip = health.error || `LLM server unreachable at ${health.api_url || 'configured endpoint'}`;
        } else if (!health.model_ready) {
            color = '#f59e0b'; // Amber
            label = activeModel ? `${activeModel} (Missing)` : 'No Model Configured';
            tooltip = health.error || `Model '${activeModel}' is not loaded on ${health.api_url}`;
        }

        if (dotEl) {
            dotEl.style.backgroundColor = color;
            dotEl.style.boxShadow = `0 0 8px ${color}`;
        }
        if (askDotEl) {
            askDotEl.style.backgroundColor = color;
            askDotEl.style.boxShadow = `0 0 8px ${color}`;
        }
        if (nameEl) {
            nameEl.innerText = label;
            nameEl.style.color = !health.online ? '#ef4444' : (!health.model_ready ? '#f59e0b' : '');
        }
        if (pillEl) {
            pillEl.title = tooltip;
        }
    }

    function fetchModels(userTriggered = false) {
        // Send the engine/endpoint SELECTED in the dropdown so the backend lists models
        // for THAT engine (not just the saved one). This is what lets the list update live
        // when the user switches API Engine or modifies URL/Token before saving.
        const apiType = document.getElementById('config-api-type').value;
        const apiUrl = ((document.getElementById('config-api-url') || {}).value || '').trim().replace(/\/+$/, '');
        const apiKey = ((document.getElementById('config-api-key') || {}).value || '').trim();
        const params = new URLSearchParams();
        params.set('api_type', apiType);
        if (apiUrl) params.set('api_url', apiUrl);
        if (apiKey) params.set('api_key', apiKey);

        const modelSelect = document.getElementById('config-active-model');
        const refreshIcon = document.getElementById('model-refresh-icon');
        if (refreshIcon) refreshIcon.classList.add('spin-icon');

        // Check if the current form endpoint matches the saved server configuration
        const savedApiType = (appConfig && appConfig.api_type) || '';
        const savedApiUrl = ((appConfig && appConfig.api_url) || '').trim().replace(/\/+$/, '');
        const savedApiKey = ((appConfig && appConfig.api_key) || '').trim();
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
                    if (!endpointChanged) {
                        updateLlmHealthUI({
                            online: false,
                            model_ready: false,
                            active_model: activeModel,
                            api_url: apiUrl || savedApiUrl,
                            error: data.error || 'Endpoint unreachable'
                        });
                    }
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
                    if (!endpointChanged) {
                        updateLlmHealthUI({
                            online: true,
                            model_ready: false,
                            active_model: activeModel,
                            api_url: apiUrl || savedApiUrl,
                            error: 'No models found on endpoint'
                        });
                    }
                    return;
                }

                // If saved active model exists on this endpoint, preserve it.
                // Support exact match first, then fuzzy match (case-insensitive, tag-ignoring, or basename matching)
                let targetSelected = null;
                if (activeModel) {
                    if (modelsList.includes(activeModel)) {
                        targetSelected = activeModel;
                    } else {
                        // Fuzzy search: e.g. "gemma-4-12b" matching "google/gemma-4-12b-qat"
                        const norm = (s) => s.toLowerCase().replace(/[:/._-]/g, '');
                        const actNorm = norm(activeModel);
                        const match = modelsList.find(m => {
                            const mNorm = norm(m);
                            return mNorm.includes(actNorm) || actNorm.includes(mNorm);
                        });
                        if (match) {
                            targetSelected = match;
                        }
                    }
                }

                // If active model still not matched but exists in saved config, add it as a preserved choice
                // so user's configured model is never silently lost or wiped
                let effectiveActiveModel = targetSelected || modelsList[0];

                const judgeSelect = document.getElementById('config-judge-model');
                const savedJudgeModel = (appConfig && appConfig.judge_model) || '';
                if (judgeSelect) {
                    judgeSelect.innerHTML = '<option value="">(Same as Active Generation Model)</option>';
                }

                // If activeModel is configured on THIS SAME endpoint but temporarily unloaded, preserve it as an option.
                // If the user changed endpoint/engine (endpointChanged === true), do NOT drag the old engine's model over.
                if (!endpointChanged && activeModel && !modelsList.includes(activeModel) && !targetSelected) {
                    const savedOpt = document.createElement('option');
                    savedOpt.value = activeModel;
                    savedOpt.innerText = `${activeModel} (Configured, Not Loaded)`;
                    savedOpt.selected = true;
                    modelSelect.appendChild(savedOpt);
                    effectiveActiveModel = activeModel;
                }

                modelsList.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m;
                    opt.innerText = m;
                    if (m === effectiveActiveModel) {
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

                if (!endpointChanged) {
                    const modelReady = !!(targetSelected || (!activeModel && modelsList.length > 0));
                    updateLlmHealthUI({
                        online: true,
                        model_ready: modelReady,
                        active_model: effectiveActiveModel,
                        api_url: apiUrl || savedApiUrl,
                        error: modelReady ? null : `Configured model '${activeModel}' is not loaded on ${apiUrl || savedApiUrl}`
                    });
                }
            })
            .catch(() => {
                if (refreshIcon) refreshIcon.classList.remove('spin-icon');
                updateLlmHealthUI({
                    online: false,
                    model_ready: false,
                    active_model: (appConfig && appConfig.model) || '',
                    api_url: apiUrl || savedApiUrl,
                    error: 'Endpoint unreachable'
                });
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

                // Update lightweight Series & Book Filter Ribbon
                renderSeriesPillsBar();

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

                // 2.5 Series & Book Hierarchy Filter
                if (selectedSeries === 'standalone') {
                    filesToRender = filesToRender.filter(f => !f.series);
                } else if (selectedSeries !== 'all') {
                    filesToRender = filesToRender.filter(f => f.series === selectedSeries);
                    if (selectedBook !== 'all') {
                        filesToRender = filesToRender.filter(f => String(f.book) === String(selectedBook));
                    }
                }

                // Real-time search query filtering
                if (librarySearchQuery) {
                    const q = librarySearchQuery;
                    filesToRender = filesToRender.filter(f => {
                        const name = (f.name || '').toLowerCase();
                        const stem = (f.stem || '').toLowerCase();
                        const readableStem = stem.replace(/_/g, ' ');
                        const title = (f.title || '').toLowerCase();
                        const cefr = (f.cefr_level || '').toLowerCase();
                        return name.includes(q) || stem.includes(q) || readableStem.includes(q) || title.includes(q) || cefr === q || (cefr && cefr.includes(q));
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

                let addCardTitle = 'Add New Source';
                let addCardSubtitle = '';
                if (selectedSeries !== 'all' && selectedSeries !== 'standalone') {
                    if (selectedBook !== 'all') {
                        addCardTitle = `Add to Book ${selectedBook}`;
                        addCardSubtitle = '';
                    } else {
                        addCardTitle = `Add to ${getSeriesDisplayName(selectedSeries)}`;
                        addCardSubtitle = '';
                    }
                }

                createNewCol.innerHTML = `
                    <div class="create-new-card h-100 d-flex flex-column align-items-center justify-content-center" onclick="triggerFileInput()" title="${selectedBook !== 'all' ? 'Upload will auto-bind to Series: ' + selectedSeries + ', Book: ' + selectedBook : 'Add New Source'}">
                        <i class="bi bi-plus-lg fs-3"></i>
                        <span class="fw-semibold">${esc(addCardTitle)}</span>
                        ${addCardSubtitle}
                    </div>
                `;
                grid.appendChild(createNewCol);

                if (filesToRender.length === 0) {
                    const emptyCol = document.createElement('div');
                    emptyCol.className = libraryViewMode === 'grid' ? 'col-sm-6 col-md-8' : 'col-12';
                    if (librarySearchQuery) {
                        emptyCol.innerHTML = `
                            <div class="text-center py-5 border rounded-3" style="background: var(--card-bg); border-color: var(--card-border) !important;">
                                <i class="bi bi-search text-muted fs-2 d-block mb-2"></i>
                                <div class="text-muted fw-semibold">No matching units found for "${esc(librarySearchQuery)}"</div>
                                <button class="btn btn-sm btn-outline-secondary rounded-pill mt-3 px-3" onclick="clearLibrarySearch()"><i class="bi bi-x-circle me-1"></i> Clear Search</button>
                            </div>
                        `;
                    } else {
                        emptyCol.innerHTML = `
                            <div class="text-center py-5 border rounded-3" style="background: var(--card-bg); border-color: var(--card-border) !important;">
                                <i class="bi bi-folder2-open text-muted fs-2 d-block mb-2"></i>
                                <div class="text-muted">No units found in this category.</div>
                            </div>
                        `;
                    }
                    grid.appendChild(emptyCol);
                }

                // Render matching notebooks
                filesToRender.forEach(f => {
                    const baseName = f.stem || f.name.replace(/\.[^/.]+$/, "");
                    const displayName = baseName;
                    
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

                    const isCompiled = Boolean(f.compiled || hasVocab || hasGrammar || hasSummary);
                    const hasQuizzes = f.has_quizzes !== undefined ? Boolean(f.has_quizzes) : (compiledUnits.quizzes && compiledUnits.quizzes.some(q => q.toLowerCase().includes(baseName.toLowerCase()) && !q.toLowerCase().endsWith('_mindmap.html') && !q.toLowerCase().endsWith('_mindmap.json')));

                    let exportDisabled = false;
                    let exportTooltip = 'Download complete handouts package (.zip)';

                    if (!isCompiled) {
                        exportDisabled = true;
                        exportTooltip = 'Unit not compiled yet. Compile to generate extractions.';
                    } else if (!hasQuizzes) {
                        exportDisabled = true;
                        exportTooltip = 'No quizzes generated yet. Generate quizzes first.';
                    }

                    const cefrBadgeHtml = f.cefr_level ? `<span class="badge bg-primary-subtle text-primary border border-primary-subtle rounded-pill py-0.5 px-2 ms-1" style="font-size: 0.62rem; font-weight: 600;">${esc(f.cefr_level)}</span>` : '';

                    let seriesBadgeHtml = '';
                    if (f.series) {
                        let sLabel = f.series_display || f.series;
                        if (!f.series_display) {
                            if (sLabel.includes('New Horizon College English')) {
                                sLabel = 'New Horizon · 4e';
                            } else if (sLabel.includes('New Practical English')) {
                                sLabel = 'Practical English';
                            } else if (sLabel.length > 14) {
                                sLabel = sLabel.substring(0, 14) + '…';
                            }
                        }
                        seriesBadgeHtml = `<span class="badge bg-info-subtle text-info border border-info-subtle rounded-pill py-0.5 px-2" style="font-size: 0.62rem; font-weight: 500;" title="${esc(f.series)}${f.book_name ? ' · ' + esc(f.book_name) : ''}">${esc(sLabel)}</span>`;
                    } else {
                        seriesBadgeHtml = `<span class="badge bg-secondary-subtle text-secondary border border-secondary-subtle rounded-pill py-0.5 px-2" style="font-size: 0.62rem; font-weight: 500;">Standalone</span>`;
                    }

                    const card = document.createElement('div');
                    card.className = libraryViewMode === 'grid' ? 'col-sm-6 col-md-4' : 'col-12';
                    const dateStr = "Jun 29, 2026"; // Consistent date metadata

                    if (libraryViewMode === 'grid') {
                        card.innerHTML = `
                            <div class="source-card d-flex flex-column justify-content-between h-100 position-relative ${f.is_transcribing ? 'transcribing-card' : ''}" onclick="if (!event.target.closest('.notebook-action-dropdown')) openUnitWorkspace('${esc(f.name)}')">
                                <div>
                                    <!-- Cover Thumbnail and Action menu -->
                                    <div class="d-flex align-items-start justify-content-between mb-3">
                                        <div class="notebook-thumbnail d-flex align-items-center justify-content-center ${f.is_transcribing ? 'transcribing-spinner' : ''}">
                                            <div class="notebook-spine"></div>
                                            <i class="bi ${f.is_transcribing ? 'bi-soundwave text-info' : 'bi-book-fill'}" style="font-size: 1.4rem; color: #ffffff; opacity: 0.95;"></i>
                                        </div>
                                        
                                        <!-- Three-dot Action Menu -->
                                        <div class="dropdown notebook-action-dropdown">
                                            <button class="btn btn-link text-muted p-0 border-0" type="button" data-bs-toggle="dropdown" aria-expanded="false" style="font-size: 1.1rem; line-height: 1;">
                                                <i class="bi bi-three-dots-vertical"></i>
                                            </button>
                                            <ul class="dropdown-menu dropdown-menu-end shadow-sm">
                                                ${f.is_transcribing ? '' : `<li><a class="dropdown-item py-1.5 small text-on-surface" href="#" onclick="closeAllDropdowns(); triggerUnitCompileAll('${esc(f.name)}'); return false;"><i class="bi bi-cpu me-2 text-warning"></i>Compile Extractions</a></li>`}
                                                ${f.is_transcribing ? '' : `<li><a class="dropdown-item py-1.5 small text-on-surface" href="#" onclick="closeAllDropdowns(); triggerUnitGenerateAllQuizzes('${esc(f.name)}'); return false;"><i class="bi bi-patch-question-fill me-2 text-danger"></i>Generate Quizzes</a></li>`}
                                                ${f.is_transcribing ? '' : `<li><a class="dropdown-item py-1.5 small text-on-surface ${(!f.compiled && !f.has_quizzes) ? 'disabled text-muted' : ''}" href="#" onclick="${(!f.compiled && !f.has_quizzes) ? 'return false;' : `closeAllDropdowns(); triggerUnitReAudit('${esc(baseName)}'); return false;`}" title="Re-audit generated extractions and quizzes"><i class="bi bi-patch-check-fill me-2 text-success"></i>Re-Audit Unit</a></li>`}
                                                ${f.is_transcribing ? '' : `<li><hr class="dropdown-divider my-1"></li>`}
                                                ${f.is_transcribing ? '' : `<li><a class="dropdown-item py-1.5 small text-on-surface ${exportDisabled ? 'disabled text-muted' : ''}" href="#" onclick="${exportDisabled ? 'return false;' : `closeAllDropdowns(); exportUnit('${esc(baseName)}'); return false;`}" title="${esc(exportTooltip)}"><i class="bi bi-download me-2 text-info"></i>Export Unit Handouts (.zip)</a></li>`}
                                            </ul>
                                        </div>
                                    </div>
                                    
                                    ${f.is_transcribing ? `
                                    <div class="mb-1.5 d-flex align-items-center gap-1 text-info fw-semibold font-monospace" style="font-size: 0.62rem; letter-spacing: 0.05em;">
                                        <span class="spinner-grow spinner-grow-sm text-info me-1" style="width: 0.5rem; height: 0.5rem;"></span>
                                        TRANSCRIBING...
                                    </div>
                                    ` : ''}

                                    <h6 class="fw-bold mb-1 text-truncate" title="${f.title ? esc(f.title) : esc(displayName)}">${esc(displayName)}</h6>
                                    <!-- Date & Source Metadata -->
                                    <div class="d-flex align-items-center gap-2 mb-3" style="font-size: 0.7rem; color: var(--on-surface-variant);">
                                        <span>${dateStr}</span>
                                        <span class="text-muted">•</span>
                                        <span>${f.is_transcribing ? 'Whisper Active' : `${(f.size / 1024).toFixed(1)} KB`}</span>
                                    </div>
                                </div>
                                
                                <div class="d-flex align-items-center justify-content-between pt-2 border-top" style="border-color: var(--card-border) !important;">
                                    <div class="d-flex align-items-center gap-2">
                                        ${f.is_transcribing ? `<span class="text-muted small" style="font-size: 0.65rem;">Processing audio...</span>` : ''}
                                        ${seriesBadgeHtml}
                                        ${cefrBadgeHtml}
                                    </div>
                                    <div class="d-flex align-items-center gap-2">
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
                            </div>
                        `;
                    } else {
                        // List mode
                        card.innerHTML = `
                            <div class="source-card d-flex align-items-center justify-content-between py-2.5 px-3 ${f.is_transcribing ? 'transcribing-card' : ''}" onclick="if (!event.target.closest('.notebook-action-dropdown')) openUnitWorkspace('${esc(f.name)}')">
                                <div class="d-flex align-items-center gap-3 text-truncate" style="flex: 1;">
                                    <div class="notebook-thumbnail d-flex align-items-center justify-content-center ${f.is_transcribing ? 'transcribing-spinner' : ''}" style="width: 32px; height: 40px; border-radius: 4px;">
                                        <div class="notebook-spine" style="width: 4px;"></div>
                                        <i class="bi ${f.is_transcribing ? 'bi-soundwave text-info' : 'bi-book-fill'}" style="font-size: 0.95rem; color: #ffffff; opacity: 0.95;"></i>
                                    </div>
                                    <div class="text-truncate">
                                        <h6 class="fw-bold mb-0 text-truncate" style="font-size: 0.9rem;" title="${f.title ? esc(f.title) : esc(displayName)}">${esc(displayName)}</h6>
                                        <div class="d-flex align-items-center gap-2" style="font-size: 0.68rem; color: var(--on-surface-variant);">
                                            <span>${dateStr}</span>
                                            <span class="text-muted">•</span>
                                            <span>${f.is_transcribing ? 'Transcribing local video...' : `${(f.size / 1024).toFixed(1)} KB`}</span>
                                        </div>
                                    </div>
                                </div>
                                <div class="d-flex align-items-center gap-2">
                                    ${seriesBadgeHtml}
                                    ${cefrBadgeHtml}
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
                                    <div class="dropdown notebook-action-dropdown">
                                        <button class="btn btn-link text-muted p-0 border-0" type="button" data-bs-toggle="dropdown" aria-expanded="false" style="font-size: 1.1rem; line-height: 1;">
                                            <i class="bi bi-three-dots-vertical"></i>
                                        </button>
                                        <ul class="dropdown-menu dropdown-menu-end shadow-sm">
                                            ${f.is_transcribing ? '' : `<li><a class="dropdown-item py-1.5 small text-on-surface" href="#" onclick="closeAllDropdowns(); triggerUnitCompileAll('${esc(f.name)}'); return false;"><i class="bi bi-cpu me-2 text-warning"></i>Compile Extractions</a></li>`}
                                            ${f.is_transcribing ? '' : `<li><a class="dropdown-item py-1.5 small text-on-surface" href="#" onclick="closeAllDropdowns(); triggerUnitGenerateAllQuizzes('${esc(f.name)}'); return false;"><i class="bi bi-patch-question-fill me-2 text-danger"></i>Generate Quizzes</a></li>`}
                                            ${f.is_transcribing ? '' : `<li><a class="dropdown-item py-1.5 small text-on-surface ${(!f.compiled && !f.has_quizzes) ? 'disabled text-muted' : ''}" href="#" onclick="${(!f.compiled && !f.has_quizzes) ? 'return false;' : `closeAllDropdowns(); triggerUnitReAudit('${esc(baseName)}'); return false;`}" title="Re-audit generated extractions and quizzes"><i class="bi bi-patch-check-fill me-2 text-success"></i>Re-Audit Unit</a></li>`}
                                            ${f.is_transcribing ? '' : `<li><hr class="dropdown-divider my-1"></li>`}
                                            ${f.is_transcribing ? '' : `<li><a class="dropdown-item py-1.5 small text-on-surface ${exportDisabled ? 'disabled text-muted' : ''}" href="#" onclick="${exportDisabled ? 'return false;' : `closeAllDropdowns(); exportUnit('${esc(baseName)}'); return false;`}" title="${esc(exportTooltip)}"><i class="bi bi-download me-2 text-info"></i>Export Unit Handouts (.zip)</a></li>`}
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        `;
                    }
                    grid.appendChild(card);
                });

                // Initialize Bootstrap tooltips on newly rendered cards
                if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
                    grid.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => {
                        new bootstrap.Tooltip(el);
                    });
                }

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
                        return file.replace(/(_vocabulary|_grammar|_expressions|_summary|_mindmap|_listening_quiz|_reading_quiz|_cloze_quiz|_video_learning|_translation_quiz)?\.(md|html|txt)$/i, '');
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
                                                 const targetUnitName = unit.rawFile ? esc(unit.rawFile.name) : esc(unit.id);
                                                 const clickAction = isMd
                                                     ? `(currentUnit === '${targetUnitName}' ? null : openUnitWorkspace('${targetUnitName}')); loadWorkspaceSourceDoc('/wiki/${esc(mediaPath)}', '${esc(mName)}'); switchWorkspaceSubTab('raw')`
                                                     : `(currentUnit === '${targetUnitName}' ? null : openUnitWorkspace('${targetUnitName}')); loadHandoutPreview('/wiki/${esc(mediaPath)}')`;
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
                if (data && data.llm_health) {
                    updateLlmHealthUI(data.llm_health);
                }
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

    function pollJobStatus(jobId, service, attempt = 0) {
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
                if (service === 'compiler' || service === 'quiz' || job.type === 'audit-batch') {
                    fetchRecentJobs();
                }
                if (job.type === 'audit-batch') {
                    updateReAuditProgress(job);
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
                        if (service === 'quiz') {
                            const qp = document.getElementById('quiz-progress-section');
                            if (qp) qp.style.display = 'none';
                        }
                        fetch('/api/llm-health')
                            .then(r => r.json())
                            .then(h => updateLlmHealthUI(h))
                            .catch(() => {});
                        const lastLog = (job.logs && job.logs.length > 0) ? job.logs[job.logs.length - 1] : "Job encountered an unexpected error.";
                        let errorMsg = `<div class="mb-2 text-danger fw-semibold">${esc(lastLog)}</div>` +
                            `<div class="small text-muted">A failure record has been written to <code>logs/</code> and flagged in the <strong>LLM Hero Board &amp; QA Audit</strong> panel. Please check that your LLM server (Ollama / API) is online and the configured model is loaded.</div>`;
                        if (lastLog.includes("faster-whisper is not installed")) {
                            errorMsg = "<strong>faster-whisper is not installed.</strong><br><br>To transcribe local video/audio files, please run in your terminal:<br><pre class='p-2 bg-dark rounded border mt-2'>pip install faster-whisper</pre>";
                        }
                        showInfoModal("Task Execution Failed", errorMsg);
                    }

                    if (service === 'quiz' && job.status === 'completed' && job.saved_files.length > 0) {
                        loadHandoutPreview(job.saved_files[0]);
                        document.getElementById('quiz-progress-section').style.display = 'none';
                        switchWorkspaceSubTab('handout');
                    } else if (service === 'all-quizzes' && job.status === 'completed') {
                        const count = (job.saved_files && job.saved_files.length) ? job.saved_files.length : 'all';
                        showToast(`Successfully generated ${count} quizzes for ${job.name || 'unit'}!`, 'success');
                    }
                } else {
                    const delay = (service === 'quiz' && attempt < 6) ? 250 : 1500;
                    activePollTimers[jobId] = setTimeout(() => pollJobStatus(jobId, service, attempt + 1), delay);
                }
            }
        });
    }

    const LISTENING_ARCHETYPE_SCENARIOS = {
        "arch_01_arranging_and_rescheduling": [
            { id: "arch_01_a2_inf_sc01", label: "[Informal] Weekend outing or study session planning" },
            { id: "arch_01_a2_inf_sc02", label: "[Informal] Coordinating meetup time and location" },
            { id: "arch_01_b1p_form_sc01", label: "[Formal] Office or academic review rescheduling" }
        ],
        "arch_02_enquiring_comparing_and_deciding": [
            { id: "arch_02_a2_form_sc01", label: "[Everyday] Item or service enquiry and comparison" },
            { id: "arch_02_a2_form_sc02", label: "[Campus] Campus facility or route enquiry" },
            { id: "arch_02_b1p_inf_sc01", label: "[Peer] Tool, workshop or local facility discussion" }
        ],
        "arch_03_requesting_and_negotiating_help": [
            { id: "arch_03_a2_inf_sc01", label: "[Informal] Borrowing items or errand assistance" },
            { id: "arch_03_b1p_form_sc01", label: "[Workplace] Urgent shift cover or feedback assistance" }
        ],
        "arch_04_problem_sharing_empathy_and_advice": [
            { id: "arch_04_a2_inf_sc01", label: "[Informal] Stressful task or everyday worry sharing" },
            { id: "arch_04_b1p_form_sc01", label: "[Professional] Project setback or career dilemma" }
        ],
        "arch_05_opinion_exchange_and_consensus": [
            { id: "arch_05_a2_inf_sc01", label: "[Informal] Campus policy, app, or lifestyle debate" },
            { id: "arch_05_b1p_form_sc01", label: "[Seminar] Policy, technology, or trend evaluation" }
        ],
        "arch_06_collaborative_proposals_and_tradeoffs": [
            { id: "arch_06_a2_inf_sc01", label: "[Classroom] Presentation or club activity planning" },
            { id: "arch_06_b1p_form_sc01", label: "[Workplace] Project proposal or workflow planning" }
        ],
        "arch_07_complaints_apologies_and_remedies": [
            { id: "arch_07_a2_form_sc01", label: "[Service] Reporting faulty item, order, or facility" },
            { id: "arch_07_b1p_form_sc01", label: "[Professional] Quality gap or unmet service review" }
        ],
        "arch_08_social_catching_up_and_closing": [
            { id: "arch_08_a2_inf_sc01", label: "[Informal] Running into an acquaintance or friend" },
            { id: "arch_08_b1p_form_sc01", label: "[Networking] Seminar break or orientation event" }
        ],
        "arch_09_academic_seminar_and_causal_probing": [
            { id: "arch_09_a2_inf_sc01", label: "[Classroom] Why an everyday social trend occurs" },
            { id: "arch_09_b1p_form_sc01", label: "[Academic] Analyzing causal mechanisms & boundary conditions" }
        ],
        "arch_10_interviews_and_credential_pitching": [
            { id: "arch_10_a2_inf_sc01", label: "[Campus] Club recruitment or assistant interview" },
            { id: "arch_10_b1p_form_sc01", label: "[Professional] Internship, graduate, or fellowship role" }
        ],
        "arch_11_instruction_walkthrough_and_onboarding": [
            { id: "arch_11_a2_inf_sc01", label: "[Campus] Studio, library, or activity center walkthrough" },
            { id: "arch_11_b1p_form_sc01", label: "[Professional] Laboratory or project operations briefing" }
        ],
        "arch_12_negotiating_terms_and_compromise": [
            { id: "arch_12_a2_inf_sc01", label: "[Everyday] Second-hand sale, flatshare, or room booking" },
            { id: "arch_12_b1p_form_sc01", label: "[Business] Partnership or cross-team resource negotiation" }
        ],
        "arch_13_medical_consultation_and_health_triage": [
            { id: "arch_13_a2_inf_sc01", label: "[Everyday] Health center, pharmacy, or wellness room" },
            { id: "arch_13_b1p_form_sc01", label: "[Clinical] Medical clinic or specialist consultation" }
        ],
        "arch_14_resolving_misunderstandings_and_mediation": [
            { id: "arch_14_a2_inf_sc01", label: "[Interpersonal] Shared flat or friend misunderstanding" },
            { id: "arch_14_b1p_form_sc01", label: "[Workplace] Post-handover review or team mediation" }
        ]
    };

    function onListeningArchetypeChange() {
        const archSelect = document.getElementById('listening-archetype-select');
        const scenarioSelect = document.getElementById('listening-scenario-select');
        const scenarioBadge = document.getElementById('listening-scenario-badge');
        if (!archSelect || !scenarioSelect) return;

        const archVal = archSelect.value;
        scenarioSelect.innerHTML = '<option value="">Auto (Setting-Adaptive)</option>';

        if (!archVal || !LISTENING_ARCHETYPE_SCENARIOS[archVal]) {
            scenarioSelect.disabled = true;
            if (scenarioBadge) {
                scenarioBadge.className = 'badge bg-secondary-subtle text-secondary border border-secondary-subtle px-1.5 py-0.5 rounded-pill';
                scenarioBadge.innerText = 'Auto-Matched';
            }
        } else {
            scenarioSelect.disabled = false;
            const scenarios = LISTENING_ARCHETYPE_SCENARIOS[archVal];
            scenarios.forEach(sc => {
                const opt = document.createElement('option');
                opt.value = sc.id;
                opt.textContent = sc.label;
                scenarioSelect.appendChild(opt);
            });
            if (scenarioBadge) {
                scenarioBadge.className = 'badge bg-success-subtle text-success border border-success-subtle px-1.5 py-0.5 rounded-pill';
                scenarioBadge.innerText = `${scenarios.length} Scenarios`;
            }
        }
    }

    function generateQuiz() {
        const template = document.getElementById('quiz-template-select').value;
        const count = document.getElementById('quiz-count-input').value;
        const archSelect = document.getElementById('listening-archetype-select');
        const scenarioSelect = document.getElementById('listening-scenario-select');
        const listeningArchetype = (template === 'listening' && archSelect) ? archSelect.value : '';
        const listeningScenario = (template === 'listening' && scenarioSelect) ? scenarioSelect.value : '';

        if (!currentUnit) {
            showInfoModal("No Workspace Selected", "Please select a unit workspace first.");
            return;
        }

        // Prerequisite Extraction Gate: Quizzes strictly consume compiled extraction nodes
        if (template !== 'video') {
            const baseName = currentUnit.replace(/\.[^/.]+$/, "");
            const vocabName = `${baseName}_vocabulary.md`;
            const hasVocab = compiledUnits.vocabulary && compiledUnits.vocabulary.some(v => v.endsWith(vocabName) || v === vocabName);
            if (!hasVocab) {
                showInfoModal(
                    "Extraction Required",
                    `The unit <strong>${esc(baseName)}</strong> has not been extracted yet.<br><br>Please compile and extract knowledge nodes (Vocabulary &amp; Grammar) before generating quizzes.`
                );
                return;
            }
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
                    proceedWithQuizGeneration(currentUnit, count, template, listeningArchetype, listeningScenario);
                } else {
                    showInfoModal("Video Handout Unavailable", "Cannot generate video handout: This raw unit source is not mapped to an active media file or transcript.");
                }
            });
        } else {
            proceedWithQuizGeneration(currentUnit, count, template, listeningArchetype, listeningScenario);
        }
    }

    function proceedWithQuizGeneration(filename, count, template, listeningArchetype = '', listeningScenario = '') {
        const payload = { filename, count, template };
        if (template === 'listening') {
            if (listeningArchetype) payload.listening_archetype = listeningArchetype;
            if (listeningScenario) payload.listening_scenario = listeningScenario;
        }
        fetch('/api/generate-quiz', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
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
        const urlInput = document.getElementById('config-api-url');
        if (urlInput && triggerFetch) {
            const curVal = urlInput.value.trim();
            const savedVal = (appConfig && appConfig.api_url) ? appConfig.api_url.trim() : '';
            const savedType = (appConfig && appConfig.api_type) || 'ollama';
            if (val === savedType && savedVal) {
                urlInput.value = savedVal;
            } else if (val === 'ollama') {
                if (!curVal || curVal === 'http://localhost:1234' || curVal === 'http://127.0.0.1:1234' || curVal.includes('api.openai.com')) {
                    urlInput.value = savedVal || 'http://localhost:11434';
                }
            } else if (val === 'openai') {
                if (!curVal || curVal === 'http://localhost:11434' || curVal === 'http://127.0.0.1:11434') {
                    urlInput.value = 'http://localhost:1234';
                }
            }
        }
        if (triggerFetch) {
            fetchModels(true);
        }
    }

    let ttsVoiceCatalogs = {
        edge: {
            female: [
                { value: "en-US-AriaNeural", label: "US Female: Aria" },
                { value: "en-US-AvaMultilingualNeural", label: "US Female: Ava (Multilingual)" },
                { value: "en-US-EmmaMultilingualNeural", label: "US Female: Emma (Multilingual)" },
                { value: "en-US-JennyNeural", label: "US Female: Jenny" },
                { value: "en-GB-SoniaNeural", label: "UK Female: Sonia" },
                { value: "en-GB-LibbyNeural", label: "UK Female: Libby" }
            ],
            male: [
                { value: "en-GB-RyanNeural", label: "UK Male: Ryan" },
                { value: "en-US-AndrewMultilingualNeural", label: "US Male: Andrew (Multilingual)" },
                { value: "en-US-BrianMultilingualNeural", label: "US Male: Brian (Multilingual)" },
                { value: "en-US-GuyNeural", label: "US Male: Guy" },
                { value: "en-US-ChristopherNeural", label: "US Male: Christopher" },
                { value: "en-GB-ThomasNeural", label: "UK Male: Thomas" }
            ],
            default_female: "en-US-AriaNeural",
            default_male: "en-GB-RyanNeural"
        },
        kokoro: {
            female: [
                { value: "af_sarah", label: "US Female: Sarah" },
                { value: "af_heart", label: "US Female: Heart" },
                { value: "af_bella", label: "US Female: Bella" },
                { value: "af_nova", label: "US Female: Nova" },
                { value: "af_nicole", label: "US Female: Nicole" },
                { value: "bf_emma", label: "UK Female: Emma" },
                { value: "bf_isabella", label: "UK Female: Isabella" }
            ],
            male: [
                { value: "am_michael", label: "US Male: Michael" },
                { value: "am_fenrir", label: "US Male: Fenrir" },
                { value: "am_puck", label: "US Male: Puck" },
                { value: "am_adam", label: "US Male: Adam" },
                { value: "bm_george", label: "UK Male: George" },
                { value: "bm_lewis", label: "UK Male: Lewis" }
            ],
            default_female: "af_sarah",
            default_male: "am_michael"
        }
    };

    function _populateGenderVoiceSelect(selectEl, voiceList, selectedVal, defaultVal) {
        if (!selectEl) return;
        selectEl.innerHTML = '';
        (voiceList || []).forEach(v => {
            const opt = document.createElement('option');
            opt.value = v.value;
            opt.innerText = v.label;
            selectEl.appendChild(opt);
        });
        if (selectedVal && (voiceList || []).some(v => v.value === selectedVal)) {
            selectEl.value = selectedVal;
        } else if (defaultVal && (voiceList || []).some(v => v.value === defaultVal)) {
            selectEl.value = defaultVal;
        } else if (voiceList && voiceList.length > 0) {
            selectEl.value = voiceList[0].value;
        }
    }

    function onTTSEngineChange(selectedVoiceA = null, selectedVoiceB = null) {
        stopTTSVoicePreview();
        const engine = (document.getElementById('config-tts-engine').value || 'edge').toLowerCase();
        const voiceASelect = document.getElementById('config-tts-voice-a');
        const voiceBSelect = document.getElementById('config-tts-voice-b');
        if (!voiceASelect || !voiceBSelect) return;

        const applyCatalog = (cat) => {
            if (!cat) return;
            _populateGenderVoiceSelect(voiceASelect, cat.female || [], selectedVoiceA, cat.default_female);
            _populateGenderVoiceSelect(voiceBSelect, cat.male || [], selectedVoiceB, cat.default_male);
        };

        const cachedCat = ttsVoiceCatalogs[engine];
        if (cachedCat) {
            applyCatalog(cachedCat);
        }

        fetch(`/api/tts/voices?engine=${encodeURIComponent(engine)}`)
            .then(res => res.json())
            .then(data => {
                if (data && data.voices) {
                    ttsVoiceCatalogs[engine] = data.voices;
                    const curA = selectedVoiceA || voiceASelect.value;
                    const curB = selectedVoiceB || voiceBSelect.value;
                    _populateGenderVoiceSelect(voiceASelect, data.voices.female || [], curA, data.voices.default_female);
                    _populateGenderVoiceSelect(voiceBSelect, data.voices.male || [], curB, data.voices.default_male);
                }
            })
            .catch(() => {});
    }

    let _ttsPreviewAudio = null;
    let _ttsPreviewReqId = 0;

    function stopTTSVoicePreview() {
        _ttsPreviewReqId += 1;
        if (_ttsPreviewAudio) {
            try {
                _ttsPreviewAudio.pause();
                _ttsPreviewAudio.currentTime = 0;
            } catch (e) {}
            _ttsPreviewAudio = null;
        }
        const iconA = document.getElementById('tts-preview-icon-a');
        const iconB = document.getElementById('tts-preview-icon-b');
        if (iconA) iconA.className = 'bi bi-volume-up-fill';
        if (iconB) iconB.className = 'bi bi-volume-up-fill';
    }

    document.addEventListener('DOMContentLoaded', function () {
        const settingsModalEl = document.getElementById('settings-modal');
        if (settingsModalEl) {
            settingsModalEl.addEventListener('hide.bs.modal', stopTTSVoicePreview);
        }
    });

    function previewTTSVoice(slot) {
        const engineEl = document.getElementById('config-tts-engine');
        const selectEl = document.getElementById(slot === 'b' ? 'config-tts-voice-b' : 'config-tts-voice-a');
        const iconEl = document.getElementById(slot === 'b' ? 'tts-preview-icon-b' : 'tts-preview-icon-a');
        if (!engineEl || !selectEl) return;

        const engine = engineEl.value || 'edge';
        const voice = selectEl.value;
        if (!voice) return;

        stopTTSVoicePreview();
        const reqId = ++_ttsPreviewReqId;

        if (iconEl) iconEl.className = 'bi bi-hourglass-split';

        fetch('/api/tts/preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ engine, voice })
        })
        .then(res => res.json())
        .then(data => {
            if (reqId !== _ttsPreviewReqId) return;
            if (iconEl) iconEl.className = 'bi bi-volume-up-fill';
            if (data && data.success && data.audio_url) {
                _ttsPreviewAudio = new Audio(data.audio_url);
                _ttsPreviewAudio.play().catch(err => {
                    showToast('Browser blocked audio playback: ' + err.message, 'error');
                });
            } else {
                showToast((data && data.error) || 'Failed to preview voice', 'error');
            }
        })
        .catch(err => {
            if (reqId !== _ttsPreviewReqId) return;
            if (iconEl) iconEl.className = 'bi bi-volume-up-fill';
            showToast('TTS preview failed: ' + err.message, 'error');
        });
    }

    function selectQuizArch(val) {
        document.getElementById('quiz-template-select').value = val;
        ['vocabulary', 'cloze', 'reading', 'translation', 'listening', 'video'].forEach(arch => {
            const btn = document.getElementById('btn-arch-' + arch);
            if (btn) {
                if (arch === val) btn.classList.add('active');
                else btn.classList.remove('active');
            }
        });

        const archContainer = document.getElementById('listening-archetype-container');
        if (archContainer) {
            if (val === 'listening') {
                archContainer.classList.remove('d-none');
                onListeningArchetypeChange();
            } else {
                archContainer.classList.add('d-none');
            }
        }
        
        let defaultCount = 10;
        if (appConfig.quiz_defaults && appConfig.quiz_defaults[val] !== undefined) {
            const rawVal = appConfig.quiz_defaults[val];
            defaultCount = (rawVal && typeof rawVal === 'object' && rawVal.count !== undefined) ? rawVal.count : rawVal;
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
        if (saveBtn) saveBtn.disabled = true;

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
            if (saveBtn) saveBtn.disabled = false;
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
            if (saveBtn) saveBtn.disabled = false;
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
        if (selectedSeries && selectedSeries !== 'all' && selectedSeries !== 'standalone') {
            xhr.setRequestHeader('X-Series', encodeURIComponent(selectedSeries));
        }
        if (selectedBook && selectedBook !== 'all') {
            xhr.setRequestHeader('X-Book', encodeURIComponent(selectedBook));
        }
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

    function exportUnit(unitName) {
        closeAllDropdowns();
        if (!unitName) {
            showToast('No unit selected for export.', 'error');
            return;
        }
        const cleanUnit = unitName.replace(/\.[^/.]+$/, "");

        showToast(`Preparing handouts package for ${cleanUnit}...`, 'info');

        fetch(`/api/export-zip?unit_name=${encodeURIComponent(cleanUnit)}`)
            .then(async response => {
                if (!response.ok) {
                    let errMsg = `Server returned ${response.status}`;
                    try {
                        const errJson = await response.json();
                        if (errJson && errJson.error) errMsg = errJson.error;
                    } catch (_) {
                        try {
                            const errText = await response.text();
                            if (errText) errMsg = errText;
                        } catch (_) {}
                    }
                    throw new Error(errMsg);
                }
                const disposition = response.headers.get('Content-Disposition');
                let filename = `${cleanUnit}_handouts.zip`;
                if (disposition && disposition.indexOf('filename=') !== -1) {
                    const matches = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/.exec(disposition);
                    if (matches != null && matches[1]) {
                        filename = matches[1].replace(/['"]/g, '');
                    }
                }
                return response.blob().then(blob => ({ blob, filename }));
            })
            .then(({ blob, filename }) => {
                const blobUrl = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = blobUrl;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                a.remove();
                setTimeout(() => window.URL.revokeObjectURL(blobUrl), 30000);
            })
            .catch(err => {
                showToast(`Export failed: ${err.message}`, 'error');
            });
    }

    function exportCurrentUnitHandouts() {
        if (!currentUnit) { showToast('No active unit selected.', 'error'); return; }
        exportUnit(currentUnit);
    }

    document.addEventListener('click', (e) => {
        if (isAskOpen) {
            const container = document.getElementById('lexis-ask-container');
            if (container && !container.contains(e.target)) {
                toggleLexisAsk(false);
            }
        }
    });

    // Bootstrap 5 tooltips are NOT auto-initialized by the bundle — create instances here.
    if (window.bootstrap && bootstrap.Tooltip) {
        document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(function (el) {
            bootstrap.Tooltip.getOrCreateInstance(el);
        });
    }

    // --- Global Window Exposures for HTML inline event handlers & ES Module backward compatibility ---
    window._populateGenderVoiceSelect = _populateGenderVoiceSelect;
    window.buildStandaloneExtractionHtml = buildStandaloneExtractionHtml;
    window.bundleStandaloneExe = bundleStandaloneExe;
    window.clearLibrarySearch = clearLibrarySearch;
    window.closeSettingsModal = closeSettingsModal;
    window.closeUnitWorkspace = closeUnitWorkspace;
    window.copyActiveNode = copyActiveNode;
    window.copyActiveWorkspaceTab = copyActiveWorkspaceTab;
    window.copyMindmap = copyMindmap;
    window.downloadActiveNode = downloadActiveNode;
    window.downloadActiveWorkspaceTab = downloadActiveWorkspaceTab;
    window.downloadMindmap = downloadMindmap;
    window.downloadRawSource = downloadRawSource;
    window.exportActiveNodeHtml = exportActiveNodeHtml;
    window.exportActiveWorkspaceTabHtml = exportActiveWorkspaceTabHtml;
    window.exportCurrentUnitHandouts = exportCurrentUnitHandouts;
    window.exportMarkdownPathAsHtml = exportMarkdownPathAsHtml;
    window.exportRawSourceHtml = exportRawSourceHtml;
    window.exportUnit = exportUnit;
    window.fetchCompiledUnits = fetchCompiledUnits;
    window.fetchConfig = fetchConfig;
    window.fetchMediaFiles = fetchMediaFiles;
    window.fetchModels = fetchModels;
    window.fetchRawFiles = fetchRawFiles;
    window.fetchRecentJobs = fetchRecentJobs;
    window.formatInlineLexisMarkdown = formatInlineLexisMarkdown;
    window.generateQuiz = generateQuiz;
    window.getActiveSubTabCategory = getActiveSubTabCategory;
    window.getSeriesDisplayName = getSeriesDisplayName;
    window.handleFileSelect = handleFileSelect;
    window.handleLocalVideoAction = handleLocalVideoAction;
    window.importVideoLink = importVideoLink;
    window.loadHandoutPreview = loadHandoutPreview;
    window.loadWorkspaceSourceDoc = loadWorkspaceSourceDoc;
    window.onApiTypeChange = onApiTypeChange;
    window.onEndpointInput = onEndpointInput;
    window.onLibrarySearchInput = onLibrarySearchInput;
    window.onLibrarySearchKeydown = onLibrarySearchKeydown;
    window.onLibrarySort = onLibrarySort;
    window.onListeningArchetypeChange = onListeningArchetypeChange;
    window.onTTSEngineChange = onTTSEngineChange;
    window.openSettingsModal = openSettingsModal;
    window.openSourceEditorForCurrentUnit = openSourceEditorForCurrentUnit;
    window.openSourceEditorForNew = openSourceEditorForNew;
    window.openSourceEditorForUnit = openSourceEditorForUnit;
    window.openUnitWorkspace = openUnitWorkspace;
    window.parseSourceContentForEditor = parseSourceContentForEditor;
    window.pollJobStatus = pollJobStatus;
    window.previewTTSVoice = previewTTSVoice;
    window.proceedWithMediaUpload = proceedWithMediaUpload;
    window.proceedWithQuizGeneration = proceedWithQuizGeneration;
    window.proceedWithRawFileUpload = proceedWithRawFileUpload;
    window.refreshQuizTotal = refreshQuizTotal;
    window.renderMarkdownNode = renderMarkdownNode;
    window.renderSeriesPillsBar = renderSeriesPillsBar;
    window.saveGlobalConfigurationAndClose = saveGlobalConfigurationAndClose;
    window.saveSourceAndSyllabus = saveSourceAndSyllabus;
    window.selectActiveMedia = selectActiveMedia;
    window.selectCompileCategory = selectCompileCategory;
    window.selectQuizArch = selectQuizArch;
    window.serializeMindmap = serializeMindmap;
    window.setBookFilter = setBookFilter;
    window.setLibraryFilter = setLibraryFilter;
    window.setSeriesFilter = setSeriesFilter;
    window.setViewMode = setViewMode;
    window.stopTTSVoicePreview = stopTTSVoicePreview;
    window.submitLexisAsk = submitLexisAsk;
    window.switchMediaTab = switchMediaTab;
    window.switchSettingsTab = switchSettingsTab;
    window.switchWorkspaceSubTab = switchWorkspaceSubTab;
    window.syncHeaderHeight = syncHeaderHeight;
    window.toggleLexisAsk = toggleLexisAsk;
    window.toggleSidebarUnitGroup = toggleSidebarUnitGroup;
    window.toggleTheme = toggleTheme;
    window.transcribeLocalVideo = transcribeLocalVideo;
    window.triggerFileInput = triggerFileInput;
    window.triggerHtmlBlobDownload = triggerHtmlBlobDownload;
    window.triggerUnitCompileAll = triggerUnitCompileAll;
    window.triggerUnitGenerateAllQuizzes = triggerUnitGenerateAllQuizzes;
    window.triggerWorkspaceCompile = triggerWorkspaceCompile;
    window.triggerWorkspaceCompileDirect = triggerWorkspaceCompileDirect;
    window.updateCompileButtonUI = updateCompileButtonUI;
    window.updateEditorCharCount = updateEditorCharCount;
    window.updateLlmHealthUI = updateLlmHealthUI;
    window.updateLocalVideoBtnIcon = updateLocalVideoBtnIcon;
    window.updateQuizCount = updateQuizCount;
    window.updateSyllabusGrammarCount = updateSyllabusGrammarCount;
    window.updateSyllabusVocabCount = updateSyllabusVocabCount;
    window.updateThemeUI = updateThemeUI;
    window.updateWorkspaceViewport = updateWorkspaceViewport;
    window.uploadFile = uploadFile;
    window.uploadLocalMediaAndTranscribe = uploadLocalMediaAndTranscribe;
