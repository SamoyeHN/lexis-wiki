# Lexis Wiki

> **Neuro-Symbolic Educational Knowledge Base & Interactive Assessment Generator**  
> Transform textbook units, lecture notes, audio, and video into Obsidian-style interlinked wiki nodes, deterministically grounded lexical/grammar extractions, and psychometrically audited interactive HTML quizzes.

---

## 🌟 Key Features

- 🧠 **Neuro-Symbolic Extraction Engine (`LinguisticEngine` + `PedLex`)**:
  - **Deterministic Vocabulary & Expression Mining**: Combines `spaCy` syntactic dependency trees, `cefrpy` lexical profiling, WordNet (`oewn:2025+`), and the offline Pedagogical Lexicon (`pedlex.db`, 106k+ phrase index) to extract single-word headwords and multi-word expressions (`phrasal verb`, `collocation`, `set phrase`, `idiom`) with zero hallucination.
  - **True Neuro-Symbolic WSD Reranker**: Locks monosemous words (~71%) instantaneously in 0ms via symbolic syntactic gates, and resolves polysemous words (~29%) via a Top-$K$ ($K \le 4$) symbolic candidate pool + 1-token Local LLM arbitration (~0.3s latency).
  - **Deterministic Grammar Pattern Extraction**: Automatically mines COBUILD syntactic formulas and structural patterns across 4 macro functional domains (*Rhetoric & Emphasis*, *Cohesion & Framing*, *Information Packaging*, *Logic & Stance*).
- 🎯 **Sense-First Five-Modality Quiz Generators**:
  - **Vocabulary in Context**: Pre-computes single-fit collocational anchors, atomic Pattern–Example frames, parallel inflected options, and zero-collision distractors directly from the locked pedagogical sense.
  - **Reading Comprehension**: Text-grounded inference, detail recall, and main-idea questions calibrated to passage CEFR ceilings.
  - **Comparative Translation**: Source-grounded bilingual appraisal and contrastive syntactic flaw analysis.
  - **Listening Comprehension**: Multi-turn spoken dialogues with integrated **Kokoro** / **Edge-TTS** dual-speaker audio synthesis.
  - **Video Comprehension**: Timestamp-anchored comprehension questions powered by **faster-whisper** transcription.
- 📏 **Deterministic CEFR Calibration (`cefrpy`)**:
  - Automatically computes passage and word-level CEFR ratings (`A1`–`C2`) at zero token cost and enforces Zipf/CEFR lexical ceilings across question stems and distractors.
- 🛡️ **Two-Level QA Quality Audit**:
  - **Level 1 (Deterministic Code Gate)**: Sub-5ms physical invariant verification (single-blank enforcement, post-inflection double-key collision screening, CEFR option bounds, Fisher-Yates option shuffling, and automatic category remapping).
  - **Level 2 (Expert LLM-as-a-Judge Audit)**: Optional blind-solver verification, distractor validity proofs, and surgical repair (`PASS` / `REPAIR` / `REWRITE`).
- ⚡ **Unified One-Shot Structured JSON Pipeline**:
  - Prompt-guided schema enforcement (`schemas.py`) delivers 100% structural completion in a single turn (~2.2× faster than legacy multi-turn pipelines) with zero secondary-turn drift.
- 🎨 **Modern Web Dashboard & Standalone Handouts**:
  - Live configuration hot-reloading, interactive 2D knowledge graph, syllabus editor, dark/light theme support, and self-contained HTML/EXE quiz exports.

---

## 🚀 Quick Start Guide

### Option A: Windows One-Click Setup (Recommended)

1. **Clone or Download** this repository.
2. **Double-click `install.cmd`**:
   - Installs `lexis` and all core NLP/lexical dependencies in editable mode.
   - Optionally installs local video/audio transcription (`faster-whisper`).
3. **Launch the Web Dashboard**:
   - Double-click `start_dashboard.vbs` (background launch) or run `lexis dashboard` in a terminal.
4. **Configure LLM Connection**:
   - Click the **Settings** icon in the Dashboard top-right bar.
   - Configure your Ollama (`http://localhost:11434`) or OpenAI-compatible API endpoint and select your model.
5. **Compile & Generate**:
   - Drop a source Markdown/Text/Video file into a unit, click **Compile Unit** to generate vocabulary, grammar, summary, and mind map nodes, then click **Generate Quiz**!

---

### Option B: Linux / WSL / macOS One-Click Setup

```bash
git clone https://github.com/SamoyeHN/lexis-wiki.git
cd lexis-wiki
chmod +x install.sh
./install.sh
```

- Automatically creates/reuses `.venv` (avoiding PEP 668 `externally-managed-environment` errors on Ubuntu/WSL), installs dependencies and `en_core_web_sm`, and runs `lexis init .` to download and verify `pedlex.db` from mirrors.

---

### Option C: Manual Installation (Cross-Platform)

```bash
# 1. Clone repository and install core package
git clone https://github.com/SamoyeHN/lexis-wiki.git
cd lexis-wiki
pip install -e .

# 2. (Optional) Install local Whisper transcription support for video quizzes
pip install -e ".[video]"

# 3. Initialize workspace and launch the interactive dashboard
lexis init .
lexis dashboard
```

---

## 📂 Stateless Project Architecture & Directory Layout

Lexis Wiki uses a **100% stateless, self-contained unit directory layout** without fragile external metadata files:

```text
wiki/<UnitName>/
├── sources/                              # Primary lesson text and source media
│   ├── <UnitName>.md                     # Source passage + optional syllabus wordlist
│   └── media/                            # Video/audio files and companion transcripts
├── extractions/                          # Deterministically grounded educational nodes
│   ├── <UnitName>_vocabulary.md          # Headwords, multi-word expressions, POS, definitions & examples
│   ├── <UnitName>_grammar.md             # Syntactic formulas, functional domains & grammar warnings
│   ├── <UnitName>_summary.md             # Structured passage summary & key points
│   └── <UnitName>_mindmap.json           # Interactive hierarchical concept map data
└── handouts/                             # Self-contained interactive HTML assessment handouts
    ├── <UnitName>_vocabulary_quiz.html
    ├── <UnitName>_reading_quiz.html
    ├── <UnitName>_translation_quiz.html
    ├── <UnitName>_listening_quiz.html
    └── <UnitName>_video_quiz.html
```

- **Immutable Physical Primary Keys**: Folder slug `<UnitName>` serves as the stable filesystem key, while human-readable lesson titles are decoupled via YAML frontmatter (`title: "..."`).
- **Source-Grounded Assessments**: Quiz generators always extract fresh semantic blueprints directly from `sources/<UnitName>.md`, ensuring user edits or translations in `extractions/*.md` never corrupt assessment integrity.
- **Dual-Distribution Ready**: Runs out-of-the-box in open-source mode (`spaCy` + `cefrpy` + Open English WordNet `oewn:2025+`) and automatically upgrades to full corpus-powered mode when `pedlex.db` is present.

---

## ⚙️ Configuration (`wiki_config.json`)

Settings can be updated live in the Web Dashboard or in `wiki_config.json` (changes hot-reload automatically without restarting the server):

| Setting | Type | Description | Default / Example |
|---------|------|-------------|-------------------|
| `api_type` | `str` | LLM backend engine (`ollama` or `openai`) | `"ollama"` |
| `api_url` | `str` | API endpoint URL | `"http://localhost:11434"` |
| `api_key` | `str` | API token (for OpenAI-compatible endpoints) | `""` or `"sk-..."` |
| `model` | `str` | Primary generation & 1-token WSD arbitration model | `"gemma4:12b"` / `"llama3.1:8b"` |
| `enable_expert_audit` | `bool` | Master switch for Level 2 LLM-as-a-Judge semantic audit | `false` |
| `judge_model` | `str` | Independent Level 2 evaluation model | `"gemma4:27b"` / `"mistral-small3.2:24b"` |
| `enable_authentic_cloze` | `bool` | Enable authentic corpus cloze items in vocabulary quizzes | `true` |
| `num_ctx` | `int` | LLM context window size (tokens) | `16384` |
| `max_tokens` | `int` | Maximum generation token budget | `8192` |
| `compile_defaults.max_parallel` | `int` | Maximum concurrent compilation tasks | `1` |
| `tts_engine` | `str` | Listening quiz TTS backend (`kokoro` or `edge`) | `"edge"` |

### 💡 Hardware & VRAM Concurrency Recommendations

- **$\le 12\text{ GB}$ VRAM (e.g., RTX 3060 / 4070)**:
  - Use 7B–12B models (`llama3.1:8b`, `ministral-3:8b`, `gemma4:12b`).
  - Set `max_parallel: 1` in `wiki_config.json`, `OLLAMA_NUM_PARALLEL=1` in Ollama, or `Max Concurrent Predictions = 1` in LM Studio.
- **$\ge 24\text{ GB}$ VRAM (e.g., RTX 3090 / 4090)**:
  - Safely scale to `max_parallel: 2` and run 24B–31B judge models locally.

### Local TTS Service Endpoints

| Engine | Default Endpoint | Default Speaker Voices |
|--------|------------------|------------------------|
| **Kokoro** | `http://localhost:8880/v1/audio/speech` | `af_sarah` / `am_michael` |
| **Edge-TTS** | `http://localhost:5050/v1/audio/speech` | `en-US-AriaNeural` / `en-GB-RyanNeural` |

---

## 🛠️ Quick CLI Reference

```bash
# 1. Start the interactive Web Dashboard
lexis dashboard

# 2. Compile a unit (or specific categories: vocabulary,grammar,summary,mindmap)
lexis compile Book_2_Unit_1_Section_A
lexis compile Book_2_Unit_1_Section_A -c vocabulary,grammar

# 3. Generate interactive HTML quizzes (vocabulary | reading | translation | listening | video)
lexis quiz Book_2_Unit_1_Section_A --template vocabulary --count 15
lexis quiz Book_2_Unit_1_Section_A --template reading --count 10

# 4. Re-run Level 2 expert semantic audit on existing quiz handouts
lexis re-audit Book_2_Unit_1_Section_A --template vocabulary

# 5. Import video/audio media with automatic Whisper transcription
lexis video-import "lecture.mp4" --name Unit_Video_1

# 6. Bundle an HTML quiz and media into a standalone single-file EXE
lexis export-exe wiki/Unit_Video_1/handouts/Unit_Video_1_video_quiz.html

# 7. Lint [[wikilinks]] integrity and clean up orphan files
lexis lint
```

---

## 📚 Technical Documentation (`docs/`)

For architectural deep dives and algorithmic specifications, see:
- **[PedLex Database & WSD Architecture](docs/pedlex-database-architecture.md)**: SQLite schema, 106k phrase inverted index, Definition-Locked WSD, and phrase distractor synthesis.
- **[Vocabulary Selection & Distractor Methodology](docs/vocabulary-selection-and-distractor-methodology.md)**: POS valency cascades, single-fit collocational anchors, and zero-collision distractor filtering.
- **[QA Audit & Scoring Guide](docs/audit-scores-guide.md)**: Level 1 deterministic code gate invariants and Level 2 LLM-as-a-Judge rubrics.
- **[Complete CLI Manual](docs/cli.md)**: Full command-line reference and batch automation options.

---

## 🧠 Recommended Local LLM Models

| Role | Recommended Models | Highlights |
|------|--------------------|------------|
| ⚡ **Fast Extraction & WSD Arbitration** | `llama3.1:8b` / `ministral-3:8b` / `qwen2.5:7b` | ~0.28s–0.40s 1-token sense arbitration; 100% WSD benchmark accuracy |
| 🥇 **Primary Quiz Generation** | `gemma4:12b` / `qwen3.5:9b` / `gpt-oss:20b` | Natural academic context crafting and high-discrimination distractors |
| 🛡️ **Level 2 Judge & Audit** | `gemma4:27b` / `mistral-small3.2:24b` | Rigorous blind-solving verification and minimal-invasive surgical repair |

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for full details.

```text
Copyright (c) 2026 Lexis Wiki Contributors
```