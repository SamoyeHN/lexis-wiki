# Lexis Wiki

> **AI-Powered Educational Knowledge Base & Interactive Assessment Generator**  
> Transform text, lecture notes, audio, and video into Obsidian-style interlinked wiki units, structured pedagogical extractions, and psychometrically audited interactive quizzes.

---

## 🌟 Key Features

- 📚 **Obsidian-Style Self-Contained Wiki Units**: Automatically parses and structures lessons into vocabulary, grammar rules, summaries, and interactive visual mind maps with bidirectional `[[wikilinks]]`.
- 🎯 **Five Modality Quiz Generators**: Generates comprehensive, assessment-ready quizzes with instantaneous interactive grading:
  - **Reading Comprehension** (Text-grounded inference & detail extraction)
  - **Vocabulary in Context** (Collocation, POS traps, single-blank fill-in)
  - **Comparative Translation** (Idiomatic appraisal & contrastive defect analysis)
  - **Listening Comprehension** (Multi-turn spoken dialogue with integrated Kokoro / Edge-TTS audio)
  - **Video Comprehension** (Timestamp-anchored video questions with Whisper transcription)
- 🛡️ **Dual-Tier QA Quality Audit (LLM-as-a-Judge)**:
  - **Level 1 (Deterministic Code Gate)**: Instant invariant checks (verbatim quote verification, single-key indexing, zero duplicate options).
  - **Level 2 (Expert Semantic Audit)**: Blind solving, cognitive distractor trap analysis, single-fit validity proofs, and minimal-invasive surgical repair.
- ⚡ **Two-Turn Prose-to-JSON Architecture**: Solves local LLM grammar token-stalls. Turn 1 focuses on deep pedagogical reasoning; Turn 2 packs outputs deterministically into strict JSON schemas.
- 🎨 **Modern Web Dashboard**: Zero-restart live config hot-reloading, 2D interactive knowledge graph, syllabus editor, and standalone HTML handout previews.

---

## 🚀 Quick Start Guide

### Option A: Windows One-Click Setup (Recommended)

1. **Clone or Download** this repository into a folder.
2. **Double-click `install.cmd`**:
   - Automatically installs `lexis` along with core dependencies in editable mode.
   - Prompts optionally to install local video/audio transcription (`faster-whisper`).
3. **Launch the Web Dashboard**:
   - Double-click `start_dashboard.vbs` (runs quietly in background) or run `lexis dashboard` in your terminal.
4. **Configure LLM Connection**:
   - In the Dashboard top-right bar, click the **Settings** icon.
   - Enter your Ollama (`http://localhost:11434`) or OpenAI-compatible endpoint URL and select your active model.
5. **Create Your First Unit**:
   - Click **Add Source** or drop a Markdown/Text/Video file.
   - Click **Compile Unit** to extract vocabulary, grammar, and mind maps.
   - Choose a quiz type (e.g. Reading, Vocabulary) and click **Generate Quiz**!

---

### Option B: Manual Installation (Cross-Platform)

```bash
# 1. Clone repository and install core package
git clone https://github.com/SamoyeHN/lexis-wiki.git
cd lexis-wiki
pip install -e .

# 2. (Optional) Install local Whisper transcription support for video quizzes
pip install -e ".[video]"
# Or directly: pip install faster-whisper

# 3. Initialize current workspace
lexis init .

# 4. Start the interactive dashboard
lexis dashboard
```

---

## 📂 Project Architecture & Directory Layout

The workspace operates under a **100% stateless, self-contained** directory model without fragile global databases:

```
wiki/<UnitName>/
├── sources/              # Primary lesson text (<UnitName>.md) and source media
│   └── media/            # Video/audio files and companion transcripts
├── extractions/          # AI-generated educational nodes (Markdown + [[wikilinks]])
│   ├── <UnitName>_vocab.md
│   ├── <UnitName>_grammar.md
│   ├── <UnitName>_summary.md
│   └── <UnitName>_mindmap.html
└── handouts/             # Exported interactive HTML quizzes with embedded assets
    ├── <UnitName>_reading_quiz.html
    ├── <UnitName>_vocab_quiz.html
    └── ...
```

- **Stable Physical Primary Keys**: Directory and filenames serve as physical anchors; human-facing lesson titles are dynamically decoupled via YAML frontmatter (`title: "..."`).
- **Zero Renaming Side-Effects**: Renaming display titles in markdown never breaks filesystem links or media bindings.

---

## ⚙️ Configuration (`wiki_config.json`)

Settings can be edited directly in the Dashboard UI or persistently saved in `wiki_config.json`. Changes are **hot-reloaded instantly** without restarting the server:

| Setting | Type | Description | Default / Example |
|---------|------|-------------|-------------------|
| `api_type` | `str` | LLM backend engine (`ollama` or `openai`) | `"ollama"` |
| `api_url` | `str` | API endpoint address | `"http://localhost:11434"` |
| `api_key` | `str` | Authentication token (for remote OpenAI-compatible APIs) | `""` or `"sk-..."` |
| `model` | `str` | Primary generation model | `"qwen2.5:14b"` |
| `enable_expert_audit`| `bool` | Master toggle for Level 2 LLM-as-a-Judge semantic auditing | `false` |
| `judge_model` | `str` | Dedicated Level 2 evaluation judge model | `"mistral-small3.2:24b"` |
| `enable_prose_pipeline`| `bool`| Multi-turn Prose-to-JSON pipeline toggle | `true` |
| `tts_engine` | `str` | Listening quiz TTS engine (`kokoro` or `edge`) | `"edge"` |
| `tts_voice_a` | `str` | Dialogue Speaker 1 Voice | `"en-US-AriaNeural"` / `"af_sarah"` |
| `tts_voice_b` | `str` | Dialogue Speaker 2 Voice | `"en-GB-RyanNeural"` / `"am_michael"` |

### Local TTS Service Endpoints

| Engine | Default Endpoint | Speaker 1 / Speaker 2 Defaults |
|--------|------------------|--------------------------------|
| **Kokoro** | `http://localhost:8880/v1/audio/speech` | `af_sarah` / `am_michael` |
| **Edge-TTS** | `http://localhost:5050/v1/audio/speech` | `en-US-AriaNeural` / `en-GB-RyanNeural` |

---

## 🛠️ Command-Line Interface (CLI) Manual

### 1. Compiling Source Material into Wiki Extractions

```bash
# Compile directly from an external source file without manual copying
lexis compile "D:\Courseware\Book_4_Unit_1.md"

# Re-compile an existing unit already placed inside wiki/
lexis compile Book_4_Unit_1
```

### 2. Generating Interactive Quizzes

```bash
# Reading Comprehension Quiz (10 questions)
lexis quiz Book_4_Unit_1 --template reading --count 10

# In-context Vocabulary Quiz (20 questions)
lexis quiz Book_4_Unit_1 --template vocabulary --count 20

# Comparative Translation Quiz (15 questions)
lexis quiz Book_4_Unit_1 --template translation --count 15

# Listening Dialogue Quiz with auto-synthesized audio (10 questions)
lexis quiz Book_4_Unit_1 --template listening --count 10

# Timestamped Video Comprehension Quiz (5 questions)
lexis quiz Book_4_Unit_1 --template video --count 5
```

### 3. Importing Video & Audio Sources

```bash
# Import from YouTube or Bilibili URL (auto-fetches subtitles)
lexis video-import "https://www.youtube.com/watch?v=..."

# Import local video file (auto-transcribes via faster-whisper)
lexis video-import "lecture.mp4"

# Import video with explicit external subtitle track
lexis video-import "lecture.mp4" --subtitle "subtitles.srt"
```

### 4. Running Quality Audits & Inspection

```bash
# Run quality audit across generation execution logs and display LLM Hero Board
lexis audit

# Run batch re-audit over existing handouts using the Level-2 Expert Judge
lexis re-audit

# Re-audit reading quizzes for a specific unit with 4 parallel worker threads
lexis re-audit Book_4_Unit_1 --template reading --workers 4
```

### 5. Maintenance & Refactoring

```bash
# Inspect wiki integrity and check for broken [[wikilinks]]
lexis lint

# Normalize tags to standard kebab-case format
lexis lint --fix-tags

# Safely prune orphaned extractions/handouts where sources were removed
lexis lint --prune

# Safely rename a unit folder and cascade updates across all internal files
lexis rename Book_4_Unit_1 Book_4_Unit_A
```

---

## 🧩 Customization & Prompt Engineering

Lexis Wiki enforces a clean **Schema-First Prompt Architecture**:
- **Schemas (`librarian/prompts/*.json`)**: Enforce data types, JSON structures, and array boundaries.
- **Prompts (`librarian/prompts/*.md`)**: Focus purely on pedagogical standards, distractor traps, and CEFR item writing guidelines.

| Modality / Task | Pedagogical Prompt | JSON Schema Specification |
|-----------------|--------------------|---------------------------|
| **Vocabulary Extraction** | `extract_vocabulary.md` | `extract_vocabulary.json` |
| **Grammar Pattern Extraction** | `extract_grammar.md` | `extract_grammar.json` |
| **Unit Summary & Key Points** | `extract_summary.md` | `extract_summary.json` |
| **Visual Mind Map** | `extract_mindmap.md` | `extract_mindmap.json` |
| **Reading Quiz** | `reading_quiz.md` | `reading_quiz.json` |
| **Vocabulary Quiz** | `vocabulary_quiz.md` | `vocabulary_quiz.json` |
| **Translation Quiz** | `translation_quiz.md` | `translation_quiz.json` |
| **Listening Quiz** | `listening_quiz.md` | `listening_quiz.json` |
| **Video Quiz** | `video_quiz.md` | `video_quiz.json` |

*To modify questions or extractions, simply customize the Markdown instructions or JSON fields in `librarian/prompts/` without touching Python code.*

---

## 🧠 Recommended LLM Models

For best results when running locally with **Ollama**:

| Tier | Model | Recommended Use Case |
|------|-------|----------------------|
| 🥇 **Pedagogical Master** | `mistral-small3.2:24b` / `qwen2.5:14b` | Outstanding distractor traps, academic vocabulary, and Level 2 semantic audit |
| ⚡ **Fast & Lightweight** | `gemma4:e4b-it-qat` / `qwen2.5:7b` | High-speed drafting and vocabulary extraction on low-VRAM laptops |
| 🛡️ **Judge & Audit Specialist**| `phi4:14b` / `gemma4:12b` | Blind-solving verification and surgical distractor repair |

---

## 📄 License

MIT License. Designed with excellence for EFL/ESL educators, instructional designers, and autonomous learners.