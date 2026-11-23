/**
 * Lexis Wiki Dashboard - Main ES Module Entry Point
 * Orchestrates modules, manages state, and guarantees backward compatibility for window.*
 */

import * as api from './api.js';
import * as state from './state.js';
import * as utils from './utils.js';
import * as glossary from './glossary.js';

export { api, state, utils, glossary };

// Expose modules namespace on window for developer console and debugging
if (typeof window !== 'undefined') {
    window.LexisModules = {
        api,
        state,
        utils,
        glossary
    };
}
