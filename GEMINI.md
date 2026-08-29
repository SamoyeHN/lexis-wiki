# Lexis Wiki Project Instructions

> **Purpose**: This document serves as the single source of truth for the Lexis Wiki project -- an AI-powered wiki generator that produces Obsidian-style educational content from source materials using a schema-first prompt architecture.

---
## 1. Prompt Architecture

Prompts drive pedagogical quality and assessment design; JSON schemas enforce output structure.

- **Clean Separation of Responsibilities**:
  - **Schema = Structure**: `schemas.py` defines output format, required keys, JSON data types, and array constraints via native API structured outputs.
  - **Prompt = Pedagogy & Quality**: `.md` prompts focus 100% on educational standards (CEFR/TOEFL), item-writing rules, distractor engineering, and `MANDATE:` rules—completely free of mechanical JSON formatting instructions.
- **Dual-Mode Structured Output Architecture**:
  - **Prompt-Guided JSON Mode (`format: "json"`, Default)**: When calling local engines (Ollama), `format: "json"` is active (`"enforce_gbnf": false` in `wiki_config.json`). The system automatically injects the JSON Schema derived programmatically from `schemas.py` into the system role. This eliminates GBNF grammar parser stalls, tokenizer conflicts (e.g. 131k tokenizers in Nemotron/Gemma), and CPU-bound token-masking timeouts while maintaining 100% schema fidelity.
  - **Strict GBNF / Native Schema Mode**: Supported via `"enforce_gbnf": true` in `wiki_config.json` for engines with hardware-accelerated grammar transducers (e.g., OpenAI `json_schema` strict mode).
  - **Automatic Empty-Output Fallback**: If strict GBNF mode fails or returns empty tokens, `llm.py` automatically catches the failure, logs a warning, and retries seamlessly in `format: "json"`.
- **Human-Readable Logging**:
  - `librarian/logger.py` automatically indents and formats `--- RAW RESPONSE ---` with 2-space pretty-printed JSON in all task logs under `logs/`.

---

## 2. Self-Contained Unit Structure

Each unit is a self-contained folder under `wiki/`. All lookups use `normalize_name()` (case-insensitive, ignores spaces/special chars).

### 2.1 Directory Layout

```
wiki/<UnitName>/sources/<UnitName>.md | media/ 
              /extractions/<UnitName>_{vocab,grammar,summary,mindmap}.md
              /handouts/<UnitName>_[type]_quiz.html
```

### 2.2 Stateless Rules

- **Deprecation of Global `raw/` and `raw/media/` Directories**.
- **Zero Background Scans / No Self-Healing**: There are no automatic, implicit, or startup scans running in the background to self-heal or move files. Content is managed and organized strictly through explicit compilation commands (`lexis compile`) or explicit dashboard upload/compile buttons.
- **No metadata files (Removal of `unit.json`).** The architecture is 100% stateless. `unit.json` has been completely eliminated from the system. Unit title = formatted folder name (e.g., `Book_4_Unit_1` → `Book 4 Unit 1`).
- **Active media** = newest file in `sources/media/` by mtime; dashboard click updates mtime.
- **Video Handout Validation**: The `/api/check-video-source` endpoint robustly validates video units by checking both (1) companion transcript markdown files in the unit's `sources/media/` directory, and (2) primary standalone transcript files directly inside the `sources/` directory (matching `sources/<UnitName>.md` which contains the video's YAML frontmatter).
- **Media & Supplementary Layout**:
  - **Standalone Video Unit**: Video file is saved under `sources/media/<UnitName>.<ext>`, and transcript `.md` is saved as the primary source at `sources/<UnitName>.md`.
  - **Supplementary Video (Existing Unit)**: Both the video file (e.g., `Three_Gorges_Dam.mkv`) and its transcript `.md` (e.g., `Three_Gorges_Dam.md`) are saved under `sources/media/` to protect the primary text source (`sources/<UnitName>.md` or `.txt`) from overwrite.
- **Library Sorting (Recency-First)**: The Document Library automatically displays files sorted by modification time (`mtime`) descending when the default `'recent'` sorting option is active. This places newly uploaded, created, or edited units first, directly next to the "Add Source" card.
- **Instant Workspace Transition on Completion**: The poller instantly opens the workspace and auto-switches to the **Raw Source** tab the moment background transcription completes.
- **Commands**: `lexis rename <old> <new>` renames unit + all internals.

---

## 3. Naming Conventions

Use `snake_case` for all JSON keys and variable names.

### Must-use (per schemas.py)
- **`concepts`** (not `items`) -- semantic concept topics
- **`part_of_speech`** (not `pos`) -- lexical category
- **`explanation`** -- pedagogical reasoning across all quiz types
- **`target_language`** (not `language`) -- translation target lang

---
## 4. Dashboard Input Configuration

### Autofill Prevention
Prevent browsers from misclassifying text inputs as password/URL fields:
- Passwords: `autocomplete="new-password"` (mandatory)
- URL/text fields: distinct names (`video-source-url`), `autocomplete="url"`, `autocorrect="off"`, `inputmode="url"`

### TTS API Endpoints
- **Kokoro**: `http://localhost:8880/v1/audio/speech`
- **Edge-TTS**: `http://localhost:5050/v1/audio/speech`

### TTS Voice Validation
Invalid voices in `wiki_config.json` self-heal to defaults: Kokoro (`af_sarah`/`am_michael`), Edge-TTS (`en-US-AriaNeural`/`en-GB-RyanNeural`). Configure via Dashboard dropdowns.

---

## 5. To-Do List

> ✅ Complete | 🔄 In Progress | ⬜ Pending

### Completed
- [x] Embed raw audio data (Base64) into HTML handouts
- [x] Unit Dependency Graph visualization
- [x] Video Quiz Extension (Bilibili/MP4 via Whisper)
- [x] pyproject.toml dependency management
- [x] QA: Evaluation schema (faithfulness/completeness/pedagogical/schema adherence) + retry loop at <80% threshold

### Pending
**UI/UX**: HTML HUB, wiki file list with categories, breadcrumb nav, Bootstrap 5 CDN only
**Visualization**: Knowledge graph legend/zoom/filter/export, clickable nodes with side preview

---

## 6. Automatic Interlinking (Wikilinks)

The system generates an Obsidian-style wiki. The following interlinking rules apply:

- `librarian/processor.py` automatically wraps extracted vocabulary `word` and grammar `name` in double brackets `[[ ]]` during Markdown formatting.
- Concept `connections` are formatted as `[[Linked Concept]]`.

---

## 7. Frontmatter Traceability

All generated Markdown files must include properly structured YAML frontmatter:

- **Source tracking**: `source: "[[filename.md]]"` to trace back to the original material.
- **Semantic categorization**: Use meaningful categories (e.g., `category: ["vocabulary", "extraction"]`).
- **Consistent titling**: Vocabulary and grammar titles follow a uniform format (e.g, "Book 3 Unit 4").

---

## 8. Validation Over Cleaning

Enforce structure through the schema API rather than post-processing. If the returned JSON is malformed, investigate the schema or prompt alignment before adding cleanup logic.

---

## 9. MCP Tool Integration and Design Guidelines

To maintain visual excellence, security, and accuracy:
- **Bootstrap Reference**: Always use the `context7` MCP server to query and fetch the latest official specifications and best practices for Bootstrap 5 elements (e.g., spinners, classes, grid layout).
- **Web Verification**: Use the `playwright` MCP server to load, interact with, and view generated web pages or local HTML dashboards to verify that interfaces display correctly.
- **Web Design with Stitch**: Utilize the `stitch` MCP server for prototyping and executing advanced design systems and mockups to ensure a premium, modern, and highly polished visual aesthetic.