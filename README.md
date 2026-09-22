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
| `compile_defaults.max_parallel` | `int` | Maximum parallel extraction tasks | `1` |
| `tts_engine` | `str` | Listening quiz TTS engine (`kokoro` or `edge`) | `"edge"` |
| `tts_voice_a` | `str` | Dialogue Speaker 1 Voice | `"en-US-AriaNeural"` / `"af_sarah"` |
| `tts_voice_b` | `str` | Dialogue Speaker 2 Voice | `"en-GB-RyanNeural"` / `"am_michael"` |

### 💡 Recommendation Setting for different VRAM

- **For 12 GB VRAM**:
  - Run model below 12b.
  - Set `max_parallel: 1` in wiki_config.json.
  - Align your local LLM server slots as well:
    - **Ollama**: Set the environment variable `OLLAMA_NUM_PARALLEL=1`.
    - **LM Studio**: Set **`Max Concurrent Predictions`** to `1` in the model loading panel.
    
- **For 24 GB VRAM and above**:
  - Set `max_parallel: 2` in wiki_config.json.
  - Adjust server slots accordingly.

### Local TTS Service Endpoints

| Engine | Default Endpoint | Speaker 1 / Speaker 2 Defaults |
|--------|------------------|--------------------------------|
| **Kokoro** | `http://localhost:8880/v1/audio/speech` | `af_sarah` / `am_michael` |
| **Edge-TTS** | `http://localhost:5050/v1/audio/speech` | `en-US-AriaNeural` / `en-GB-RyanNeural` |

---

## 🛠️ Quick CLI Reference

Lexis provides an intuitive command-line interface for terminal users. Below are the most frequently used commands:

```bash
# 1. Launch the Web Dashboard & Workspace
lexis dashboard

# 2. Compile source text into vocabulary, grammar, and mind maps
lexis compile Book_4_Unit_1

# 3. Generate interactive quizzes (e.g., Reading Comprehension)
lexis quiz Book_4_Unit_1 --template reading --count 10

# 4. Import video/audio media (with Whisper transcription)
lexis video-import "lecture.mp4"

# 5. Lint [[wikilinks]] integrity & prune orphans
lexis lint
```

📖 **Need full parameters, batch auditing, or unit refactoring commands?**  
👉 See the complete **[Command-Line Interface (CLI) Manual](docs/cli.md)**.

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
| 🥇 **Pedagogical Master** | `gemma4:12b` / `qwen3.5:9b` / `gpt-oss:20b` | Outstanding distractor traps, academic vocabulary, and Level 2 semantic audit |
| ⚡ **Fast & Lightweight** | `gemma4:e4b` / `granite4.2:8b` | High-speed drafting and vocabulary extraction on low-VRAM laptops |
| 🛡️ **Judge & Audit Specialist**| `gemma4:31b` / `qwen3.8:27b` | Blind-solving verification and surgical distractor repair |

---

## 💖 Acknowledgments & Built With

Lexis Wiki is powered by outstanding open-source frameworks, foundational research, and design systems:

- **LLM & Inference Infrastructure**: [Ollama](https://ollama.com/) & [OpenAI API Compatible Specification](https://platform.openai.com/)
- **Core Pedagogical & Judge Models**: [Google Gemma](https://ai.google.dev/gemma), Mistral AI, and Qwen
- **Speech & Multimodal Audio**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [Kokoro TTS](https://github.com/hexgrad/kokoro), and [Edge-TTS](https://github.com/rany2/edge-tts)
- **Design System & Typography**: [Bootstrap 5](https://getbootstrap.com/), [Google Fonts & Material Symbols](https://fonts.google.com/), and [Bootstrap Icons](https://icons.getbootstrap.com/)
- **Special Thanks**: Developed with the assistance of **Google DeepMind's Antigravity** agentic pair programming platform.

---

## 📄 License & Copyright

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for full details.

```text
Copyright (c) 2026 Lexis Wiki Contributors
```

Designed with pedagogical excellence for language educators, instructional designers, and researchers in AI-assisted education.