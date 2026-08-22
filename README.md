# Lexis Wiki

AI-powered Obsidian-style wiki generator for educators. Transform source materials into structured learning content with automatic interlinking, vocabulary extraction, and interactive quiz generation.

---

## Quick Start: Your First Unit in 60 Seconds

```bash
# 1. Install
pip install -e .

# 2. Initialize project structure
lexis init .

# 3. Compile an explicit raw text file from any folder
lexis compile "D:\My Lessons\Book_4_Unit_1.md"

# 4. Generate a reading quiz
lexis quiz Book_4_Unit_1 --template reading --count 10
```

---

## Project Workflow

```
Source Materials (any folder/URL)  →  lexis compile  →  wiki/<Unit>/extractions/ (wiki nodes in-place)
Webpage Upload (Dashboard)         →  wiki/<Unit>/sources/   →  Auto-compiles to extractions/
                                                     →  lexis quiz  →  wiki/<Unit>/handouts/ (interactive HTML quizzes)
                                                     →  lexis dashboard  →  web preview
```

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

## Project Structure

Self-contained unit folders under `wiki/`:

```
wiki/<UnitName>/
├── sources/              # Raw texts, subtitles, media files
├── extractions/          # AI-generated wiki nodes (.md)
└── handouts/             # Interactive quizzes (.html)
```

- **Case-insensitive lookups**: `normalize_name()` handles all queries
- **No metadata files needed (100% Stateless)**: `unit.json` has been completely removed. Unit title is derived dynamically from the folder name, and active media is determined by the newest file (mtime) in `sources/media/`.
- **Video Handout Validation**: The `/api/check-video-source` API checks for video transcripts and URLs within the `sources/media/` directory files (e.g. `YouTube_Video_Mun_KJYXsco.md`) rather than the main unit text files, eliminating false negatives.
- **Atomic operations**: `lexis rename` handles structure changes

---

## Customization

Redesign outputs without editing Python code. Modify files in your `prompts/` folder:

| Task | Instruction File | Schema File |
|------|-----------------|-------------|
| Vocabulary | `extract_vocabulary.md` | `extract_vocabulary.json` |
| Grammar | `extract_grammar.md` | `extract_grammar.json` |
| Quiz (any type) | `[type]_quiz.md` | `[type]_quiz.json` |

**Adding fields**: Add a key to the `.json` schema and it automatically appears in wiki output.

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