/**
 * Lexis Wiki Dashboard - Reactive State Module
 * Encapsulates global state properties and event dispatching.
 */

export const state = {
    appConfig: {},
    rawFiles: [],
    compiledUnits: {},
    currentUnit: null,
    activeJobId: null,
    activePollTimers: {},
    activePreviewPath: null,
    activeSubTab: 'raw',
    libraryFilter: 'all',
    librarySortOption: 'recent',
    libraryViewMode: 'grid',
    librarySearchQuery: '',
    selectedSeries: 'all',
    selectedBook: 'all',
    activeGlossarySeries: '',
    activeGlossaryBook: '',
    glossaryEntries: [],
    glossaryFilter: 'all',
    glossarySearch: '',

    // Graph state
    canvas: null,
    ctx: null,
    nodes: [],
    links: [],
    scale: 1,
    transformX: 0,
    transformY: 0,
    dragNode: null,
    hoveredNode: null,
    isDraggingCanvas: false,
    dragStart: { x: 0, y: 0 },
    hasAutoZoomed: false,
    activeGraphCategory: 'all'
};

const listeners = new Map();

export function subscribe(key, callback) {
    if (!listeners.has(key)) {
        listeners.set(key, new Set());
    }
    listeners.get(key).add(callback);
    return () => listeners.get(key).delete(callback);
}

export function setState(key, value) {
    const prev = state[key];
    state[key] = value;
    if (listeners.has(key)) {
        listeners.get(key).forEach(cb => cb(value, prev));
    }
}
