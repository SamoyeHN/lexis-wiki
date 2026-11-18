# 📖 Lexis Wiki

<p align="center">
  <b>Automated English Lesson Deep-Reading & Self-Grading Interactive Quiz Generator</b><br>
  <i>Transform textbook units, articles, audio, and video into Obsidian-style study notes and self-contained interactive HTML quizzes.</i>
</p>

<p align="center">
  <a href="README.md"><b>English</b></a> •
  <a href="README.zh-CN.md">简体中文</a>
</p>

---

## 💡 What is Lexis Wiki?

Whether you have an English textbook unit, an academic reading, a news article, or an untranscribed lecture video/audio:

👉 **Just drop in the text or media, and in 30 seconds Lexis Wiki automatically creates:**
1. **Encyclopedic Study Notes**: Key vocabulary, idiomatic expressions, phrasal verbs, grammar patterns, and interactive Mind Maps with Obsidian-style `[[wikilinks]]`.
2. **5 Self-Grading Interactive HTML Quizzes**: Vocabulary cloze, reading comprehension, comparative translation, dual-speaker listening dialogues, and video comprehension.
3. **Single-File Standalone Delivery**: Each quiz is exported as an **independent `.html` file**. Double-click to open in any web browser on your phone, tablet, or PC—featuring a countdown timer, instant grading, answer explanations, and native AI audio narration with zero installation or server requirements!

---

## 🥊 Why Lexis Wiki vs. Prompting Raw ChatGPT?

When using generic AI chatbots to generate quizzes, teachers and learners frequently suffer from three major headaches: **arbitrary word choices, hallucinated unnatural examples, and distractor flaws (multiple valid answers or obvious grammatical giveaways)**.

Lexis Wiki solves this with a **Neuro-Symbolic Architecture** that pairs authoritative offline dictionaries with local LLM arbitration:

| Dimension | Raw LLM Prompting (e.g. ChatGPT) | Lexis Wiki (Neuro-Symbolic Engine) |
| :--- | :--- | :--- |
| **Vocabulary & Expressions** | Random words; frequently misses multi-word collocations | Analyzes syntactic dependency trees to capture whole collocations and idioms |
| **Example Authenticity** | Hallucinates unverified, often unnatural sentences | **100% locked to authentic dictionary corpus examples**; zero hallucination |
| **Word Sense Disambiguation (WSD)** | Often picks irrelevant dictionary definitions | Symbolic grammar-slot constraints + smart candidate arbitration locks the true contextual sense |
| **Quiz Distractors** | Asymmetric forms, obvious syntax clues, or ambiguous double answers | Auto-inflected parallel forms and synonym collision screening ensure **exactly one valid answer** |
| **Difficulty Calibration** | The LLM "guesses" difficulty; highly inconsistent | Enforces deterministic CEFR lexical ceilings (`A1`–`C2`) via `cefrpy` |
| **Output Delivery** | Plain text or Markdown; requires tedious manual formatting | **Exports standalone interactive HTML/EXE files** with auto-grading, audio, and mobile responsiveness |

---

## ✨ Key Features

### 1. 📚 Encyclopedic Lesson Notes
- **Headwords & Idiomatic Expressions**: Automatically identifies target words and multi-word expressions (`phrasal verb`, `collocation`, `set phrase`, `idiom`), tags CEFR levels, and binds them directly to authentic corpus sentences.
- **Grammar Formulas & Syntactic Structures**: Outlines sentence formulas (e.g., negative inversion, cleft sentences, passive frames) based on the COBUILD framework, highlighting common student error traps.
- **Passage Summaries & Concept Mind Maps**: Extracts hierarchical rhetorical outlines into an interactive, draggable concept mind map.

### 2. 📝 5 Self-Contained Interactive Quizzes
- 🔤 **Vocabulary & Collocation Cloze**: Strict sense-locked context stems with symmetric, zero-collision distractors.
- 📖 **Reading Comprehension**: Text-grounded main idea, detail recall, and logical inference questions aligned with IELTS / TOEFL / CET standards.
- ✍️ **Comparative Translation**: Highlighting natural target language phrasing vs. common Chinglish translation flaws.
- 🎧 **Dual-Speaker Listening Comprehension**: Realistic multi-turn dialogues with built-in **Kokoro** or **Edge-TTS** dual-speaker voice synthesis.
- 🎬 **Video Comprehension**: Uses **faster-whisper** to auto-transcribe local video/audio, anchoring questions to exact timestamps.

### 3. 📦 Instant Single-File Distribution
- Every generated quiz is a **standalone `.html` file** with bundled styling, scripts, and media.
- Share quizzes directly via chat or email—learners open them in any browser without signing up or setting up a server.
- Optional one-click bundling into a single standalone `.exe` application.

---

## ⚡ 3-Minute Quick Start

### Option A: Windows (Recommended 1-Click Setup)

1. **Clone or Download** this repository;
2. **Double-click `install.cmd`**: Sets up the Python environment and downloads core linguistic assets automatically;
3. **Double-click `start_dashboard.vbs`** (or run `lexis dashboard` in your terminal) to open the Web Dashboard;
4. **Configure your LLM**:
   - Click the **Settings** gear icon in the top-right corner;
   - Connect to local **Ollama** (`http://localhost:11434`) or any **OpenAI-compatible API** (DeepSeek, Qwen, Moonshot, OpenAI, etc.);
5. **Start Creating**:
   - Add your text or media into a unit folder, click **"Compile Extractions"** to generate notes, then click **"Generate Quizzes"**!

---

### Option B: Linux / macOS / WSL

```bash
# 1. Clone the repository
git clone https://github.com/SamoyeHN/lexis-wiki.git
cd lexis-wiki

# 2. Run the automated installer
chmod +x install.sh
./install.sh

# 3. Launch the Web Dashboard
lexis dashboard
```

The script sets up a dedicated `.venv`, installs dependencies, downloads `pedlex.db`, and initializes your workspace.

---

### Option C: Manual Python Installation

```bash
# 1. Install core package
git clone https://github.com/SamoyeHN/lexis-wiki.git
cd lexis-wiki
pip install -e .

# 2. (Optional) Install Whisper transcription support for video quizzes
pip install -e ".[video]"

# 3. Initialize workspace and launch the dashboard
lexis init .
lexis dashboard
```

---

## 🖥️ Quick CLI Reference

Prefer the command line? Run any task directly via the `lexis` CLI:

```bash
# Launch the interactive Web Dashboard
lexis dashboard

# Compile study notes (vocabulary, grammar, summary, mindmap) for a unit
lexis compile Book_2_Unit_1_Section_A

# Generate interactive HTML quizzes (vocabulary | reading | translation | listening | video)
lexis quiz Book_2_Unit_1_Section_A --template vocabulary --count 15
lexis quiz Book_2_Unit_1_Section_A --template reading --count 10

# Import video/audio media with automatic Whisper transcription
lexis video-import "lecture.mp4" --name Unit_Video_1

# Bundle an HTML quiz into a standalone single-file EXE
lexis export-exe wiki/Unit_Video_1/handouts/Unit_Video_1_video_quiz.html

# Check [[wikilinks]] consistency and clean up orphans
lexis lint
```

---

## 📂 Project Directory Layout

Lexis Wiki uses a clean, **self-contained unit structure** with zero external database dependencies:

```text
wiki/<UnitName>/
├── sources/                              # Primary input materials
│   ├── <UnitName>.md                     # Source lesson text + optional syllabus wordlist
│   └── media/                            # Local audio/video and companion transcripts
├── extractions/                          # Deterministically generated study notes
│   ├── <UnitName>_vocabulary.md          # Target words, collocations, definitions & examples
│   ├── <UnitName>_grammar.md             # Syntactic formulas, grammar domains & error traps
│   ├── <UnitName>_summary.md             # Passage summary & core takeaways
│   └── <UnitName>_mindmap.json           # Interactive hierarchical Mind Map data
└── handouts/                             # Standalone interactive HTML assessment handouts
    ├── <UnitName>_vocabulary_quiz.html   # Vocabulary in context cloze quiz
    ├── <UnitName>_reading_quiz.html      # Deep reading comprehension quiz
    ├── <UnitName>_translation_quiz.html  # Contrastive translation quiz
    ├── <UnitName>_listening_quiz.html    # Spoken dialogue listening quiz with audio
    └── <UnitName>_video_quiz.html        # Video timestamp-anchored quiz
```

---

## ⚙️ Configuration & Model Recommendations

Settings can be changed on the fly in the Web Dashboard or inside `wiki_config.json` (auto-reloads without restarting):

| Setting | Type | Description | Default / Example |
| :--- | :--- | :--- | :--- |
| `api_type` | `str` | LLM API backend (`ollama` or `openai`) | `"ollama"` |
| `api_url` | `str` | LLM endpoint URL | `"http://localhost:11434"` |
| `api_key` | `str` | API key (for cloud OpenAI-compatible endpoints) | `""` or `"sk-..."` |
| `model` | `str` | Primary generation & WSD arbitration model | `"gemma4:12b"` / `"qwen2.5:7b"` |
| `enable_expert_audit` | `bool` | Enable Level 2 LLM-as-a-Judge quality verification | `false` |
| `judge_model` | `str` | Independent auditor model for Level 2 verification | `"gemma4:27b"` / `"mistral-small3.2:24b"` |
| `tts_engine` | `str` | TTS backend for listening quizzes (`kokoro` or `edge`) | `"edge"` |

### 💡 Hardware & VRAM Guidelines
- **Standard Laptop / 8GB–12GB VRAM (e.g. RTX 3060 / 4060 / 4070)**:
  - Run 7B–12B models (`qwen2.5:7b`, `llama3.1:8b`, `gemma4:12b`) with `max_parallel: 1`. Fast and reliable.
- **High-End Desktop / 24GB+ VRAM (e.g. RTX 3090 / 4090)**:
  - Run 27B–32B models or enable Level 2 expert judge audits locally.
- **No Dedicated GPU**:
  - Simply enter any cloud API key (OpenAI, DeepSeek, Qwen, etc.) in the dashboard settings to run seamlessly in the cloud.

---

## 🔬 Under the Hood (For Developers & Linguists)

<details>
<summary><b>Click to expand: Theoretical foundations & neuro-symbolic mechanisms</b></summary>

Unlike simple prompt wrappers, Lexis Wiki implements computational Second Language Acquisition (SLA) principles and neuro-symbolic constraints:

1. **Deterministic Mining & Sense Locking (WSD)**:
   - Combines `spaCy` dependency parsing with an offline pedagogical lexicon (`pedlex.db` containing a 106,000+ phrase inverted index);
   - Monosemous entries (~71%) are locked in 0ms at 0 token cost via syntactic subcategorization frames;
   - Polysemous entries (~29%) are evaluated using a Top-$K$ ($K \le 4$) symbolic candidate pool followed by 1-token LLM arbitration, preventing semantic drift.
2. **Five-Source Authentic Corpus Pool**:
   - Authentic examples are harvested strictly from 5 compartments of the pedagogical corpus (Grammar Box, Sense Examples, Thesaurus, Language Activator, and Collocation dictionaries).
3. **Two-Level QA Audit Pipeline**:
   - **Level 1 Deterministic Code Gate (<5ms)**: Validates physical invariants (single-blank constraints, post-inflection double-key screening, CEFR lexical bounds, and Fisher-Yates shuffling);
   - **Level 2 Expert LLM-as-a-Judge Audit (Optional)**: Employs a blind solver model to evaluate and surgically repair question stems and distractors (`PASS` / `REPAIR` / `REWRITE`).
4. **In-Depth Documentation Index**:
   - [Vocabulary Selection & Distractor Methodology](docs/vocabulary-selection-and-distractor-methodology.md)
   - [Multi-Word Units Slicing & Distractor Architecture](docs/multi-word-units-slicing-and-distractor-architecture.md)
   - [PedLex Database Schema & Query Specifications](docs/pedlex-database-architecture.md)
   - [Listening Functional Dialogue Architecture](docs/listening-functional-dialogue-architecture.md)
   - [Reading Discourse & Psychometric Distractor Model](docs/reading-discourse-architecture.md)
   - [QA Audit Scores & Rubric Guide](docs/audit-scores-guide.md)
</details>

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

```text
Copyright (c) 2026 Lexis Wiki Contributors
```
