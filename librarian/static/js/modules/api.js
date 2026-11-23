/**
 * Lexis Wiki Dashboard - API Client Module
 * Unified Fetch wrapper for all backend REST endpoints.
 */

export async function apiGet(endpoint, params = {}) {
    const query = new URLSearchParams(params).toString();
    const url = query ? `${endpoint}?${query}` : endpoint;
    const response = await fetch(url, {
        headers: { 'Accept': 'application/json' },
        cache: 'no-store'
    });
    if (!response.ok) {
        throw new Error(`HTTP error ${response.status} from ${endpoint}`);
    }
    return response.json();
}

export async function apiPost(endpoint, data = {}) {
    const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        body: JSON.stringify(data),
        cache: 'no-store'
    });
    if (!response.ok) {
        throw new Error(`HTTP error ${response.status} from ${endpoint}`);
    }
    return response.json();
}

export async function apiPostBinary(endpoint, body, headers = {}) {
    const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/octet-stream',
            ...headers
        },
        body: body,
        cache: 'no-store'
    });
    if (!response.ok) {
        throw new Error(`HTTP error ${response.status} from ${endpoint}`);
    }
    return response.json();
}
