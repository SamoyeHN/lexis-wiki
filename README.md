# Lexis Wiki

AI-powered Obsidian-style wiki generator for EFL educators. Transform source materials into structured learning content with automatic interlinking, vocabulary, grammar and mindmap extraction, and interactive quiz generation.

---

## Features:
- 
- 

---

## Quick Start: Generate Your First Quiz

1. **Windows — one-click install** 
   - Copy librarian folder, pyproject.toml, install.cmd and start_dashboard.vbs to a folder.
   - double-click `install.cmd` to install.

2. Double click `start_dashboard.vbs` or cli `lexis dashboard` to start Dashboard.

2. Input ollama (local) or OpenAI compatible API url in Dashboard Configuration.

3. Import a markdown file. Compile --> Generate quiz.

---

## Project Workflow & folder Structure

```
Dashboard Upload  →  wiki/<Unit>/sources/  →  Auto-compiles  →  wiki/<Unit>/extractions/ 
                                           →  Generate quiz  →  wiki/<Unit>/handouts/ 

wiki/<UnitName>/
├── sources/              # Raw texts, subtitles, media files
├── extractions/          # AI-generated wiki nodes (.md)
└── handouts/             # Interactive quizzes (.html)
```

---

## Configuration

Edit `wiki_config.json` for persistent settings:

| Setting | Purpose | Example |
|---------|---------|---------|
| `api_type` | LLM engine | `"ollama"` or `"openai"` |
| `api_url` | Endpoint URL | `"http://localhost:11434"` |
| `api_key` | Access token (if remote) | `"sk-..."` |
| `active_model` | Active model name | `"qwen3.5:9b"` |
| `max_parallel` | Worker threads | `3` (match your VRAM) |
| `tts_engine` | Audio synthesis | `"kokoro"` or `"edge"` |
| `tts_voice_a/b` | Speaker voices | `"af_sarah"` / `"am_michael"` |

### TTS API Endpoints

| Engine | URL | Defaults (Spk1/Spk2) |
|--------|-----|----------------------|
| Kokoro | `http://localhost:8880/v1/audio/speech` | `af_sarah` / `am_michael` |
| Edge-TTS | `http://localhost:5050/v1/audio/speech` | `en-US-AriaNeural` / `en-GB-RyanNeural` |

Invalid voice names auto-heal to defaults. Configure via Dashboard dropdowns or manual edit.

---

## Customization

Redesign outputs without editing Python code. Modify files in `librarian/prompts/` folder:

| Task | Instruction File | Schema File |
|------|-----------------|-------------|
| Vocabulary | `extract_vocabulary.md` | `extract_vocabulary.json` |
| Expression | `extract_expression.md` | `extract_expression.json` |
| Grammar | `extract_grammar.md` | `extract_grammar.json` |
| Summary | `extract_summary.md` | `extract_summary.json` |
| Mindmap | `extract_mindmap.md` | `extract_mindmap.json` |
| Quiz (any type) | `[type]_quiz.md` | `[type]_quiz.json` |
| Expert Audit (any type) | `[type]_quiz.md` | `[type]_quiz.json` |

**Adding fields**: Add a key to the `.json` schema and it automatically appears in wiki output.

---

## Manual Install

```bash
pip install -e .            # lexis + core dependencies
pip install pyinstaller     # EXE export      pytest  # test suite
lexis init .                # initialize the project (safe to re-run)
```

# Optional: Install Video & Local Audio Transcription dependencies (faster-whisper for video quiz)
`pip install -e ".[video]"` or `pip install faster-whisper`

---

## CLI Reference (by Task)

### Compile Content

Transform source texts (from any location) into structured wiki nodes (vocabulary, grammar, concepts):

```bash
lexis compile "D:\My Lessons\Book_4_Unit_1.md" # Process file in-place from explicit path without copying it
lexis compile Book_4_Unit_1                    # Re-compile existing unit from wiki/Book_4_Unit_1/sources/
```

Output: `wiki/<Unit>/extractions/` — interlinked Markdown files with automatic `[[wikilinks]]`.

### Generate Quizzes

Create interactive HTML handouts from compiled units:

```bash
lexis quiz Book_4_Unit_1 --template reading --count 10   # Reading comprehension
lexis quiz Book_4_Unit_1 --template vocabulary --count 20 # Vocabulary quiz
lexis quiz Book_4_Unit_1 --template translation --count 15 # Translation quiz
lexis quiz Book_4_Unit_1 --template listening --count 10 # Listening (requires TTS)
lexis quiz Book_4_Unit_1 --template video --count 5      # Video-based quiz
```

### Import Videos

Generate quizzes from video content:

```bash
# From YouTube/Bilibili URL (auto-downloads subtitles)
lexis video-import "https://www.youtube.com/watch?v=..."

# From local file (auto-transcribes via Whisper)
lexis video-import "video.mp4"

# From manual .srt/.vtt subtitle file
lexis video-import "https://example.com/video" --subtitle "subtitles.srt"
```

### Maintenance & Organization

```bash
# Check for broken wikilinks
lexis lint

# Fix tag casing to kebab-case
lexis lint --fix-tags

# Remove orphaned extractions/quizzes (source files deleted)
lexis lint --prune

# Rename a unit directory + all its files
lexis rename Book_4_Unit_1 Book_4_Unit_A
```

### Quality Audit & LLM Hero Board

Evaluate LLM extraction performance, schema adherence, verbatim source faithfulness, and slot-filling pedagogical compliance across execution logs:

```bash
# Run manual quality audit and display the leaderboard
lexis audit

# Output structured evaluation results as JSON
lexis audit --json
```

### Re-audit Existing Handouts

Re-run the de-biased **Level-2 expert judge** over quizzes you have already generated. This is **read-only** — it never regenerates content; it only re-scores each handout (and re-computes the blind-solve accuracy). Handouts that already carry an embedded audit are reported side-by-side with their previous verdict, so you can spot PASS → FAIL / FAIL → PASS flips.

```bash
# Re-audit every quiz handout in the project
lexis re-audit

# Re-audit a single unit
lexis re-audit Book_4_Unit_1

# Only re-audit reading quizzes, with 4 parallel judge workers
lexis re-audit --template reading --workers 4

# Machine-readable summary
lexis re-audit --json
```

Output: passed / failed / error counts, per-unit blind-solve accuracy and overall score, plus the before/after verdict change for each handout.

**Dashboard:** the same feature is in the web UI — click the clipboard-check icon in the header to open the *Level-2 Expert Re-Audit* modal, then press **RE-AUDIT ALL**. It runs as a background job (visible in *Recent Jobs* as an "Expert Re-audit" job) and opens the before/after comparison when it finishes, saving a timestamped report to `logs/re_audit_report_*.json`.

### Dashboard & Configuration

```bash
# Start local web server (interactive preview & Hero Board modal)
lexis dashboard

# Run quality audit on log files and display the LLM Hero Board
lexis audit

# Output quality audit results as JSON
lexis audit --json

# List/manage models
lexis config --list-models
lexis config --set-model gemma4:e4b
```

---

## Recommended LLM Models

For local use via Ollama (recommended engines):

| Tier | Model | Best For |
|------|-------|----------|
| 🥇 Highest quality | `mistral-small3.2:24b` | Vocabulary/grammar extraction accuracy |
| ⚡ Fastest | `gemma4:e4b-it-qat` | Speed-optimized, good quality |
| 💪 Powerful | `qwen3.5:9b`, `granite4.1:30b` | Complex reasoning tasks |

---

## Browser Autofill Prevention

When building custom dashboard forms, prevent password manager interference:

- Password fields: `autocomplete="new-password"`
- URL fields: distinct names (`video-source-url`), `autocorrect="off"`, `inputmode="url"`

See [GEMINI.md §4](GEMINI.md) for full configuration details.