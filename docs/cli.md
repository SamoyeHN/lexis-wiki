# Command-Line Interface (CLI) Manual

`lexis` is the central command-line engine for managing curriculum units, extracting pedagogical concepts, synthesizing multimodal quizzes, and auditing assessment quality.

---

## 📋 Table of Contents

- [1. Initialization & Dashboard](#1-initialization--dashboard)
- [2. Compiling Source Material into Wiki Extractions](#2-compiling-source-material-into-wiki-extractions)
- [3. Generating Interactive Handouts & Quizzes](#3-generating-interactive-handouts--quizzes)
- [4. Importing Multimodal Media (Video & Audio)](#4-importing-multimodal-media-video--audio)
- [5. Quality Audits & Psychometric Inspection](#5-quality-audits--psychometric-inspection)
- [6. Maintenance, Refactoring & Linting](#6-maintenance-refactoring--linting)

---

## 1. Initialization & Dashboard

### Initialize Workspace
Sets up a standard Lexis Wiki structure (`wiki/`, `wiki_config.json`) in the specified directory:
```bash
# Initialize current directory as a Lexis Wiki workspace
lexis init .

# Initialize in a designated folder
lexis init /path/to/my-courseware
```

### Start Web Dashboard
Launches the zero-configuration graphical dashboard and opens your default browser:
```bash
lexis dashboard
```
> **Tip (Windows)**: You can also double-click `start_dashboard.vbs` in the project root to run the server quietly in the background without keeping a terminal open.

---

## 2. Compiling Source Material into Wiki Extractions

The compiler extracts core pedagogical targets—**Vocabulary, Grammar Patterns, Summaries, and Interactive Visual Mind Maps**—and structures them into the unit folder.

```bash
# Compile directly from an external markdown/text source file
# (Automatically creates the unit directory under wiki/ and organizes sources)
lexis compile "D:\Courseware\Book_4_Unit_1.md"

# Re-compile an existing unit already placed inside wiki/
lexis compile Book_4_Unit_1

# Targeted extraction: Compile only vocabulary and grammar
lexis compile Book_4_Unit_1 --categories vocabulary,grammar
```

---

## 3. Generating Interactive Handouts & Quizzes

Lexis supports 5 distinct assessment modalities, each outputting self-contained, offline-ready HTML handouts:

```bash
# 1. Reading Comprehension Quiz (10 questions)
# Text-grounded reasoning, detail validation, and 3-tier distractor traps
lexis quiz Book_4_Unit_1 --template reading --count 10

# 2. In-Context Vocabulary Quiz (20 questions)
# Target-collocation blanking and part-of-speech trap options
lexis quiz Book_4_Unit_1 --template vocabulary --count 20

# 3. Comparative Translation Quiz (15 questions)
# Idiomatic translation appraisal & contrastive defect analysis
lexis quiz Book_4_Unit_1 --template translation --count 15

# 4. Listening Dialogue Quiz (10 questions)
# Multi-turn spoken scripts with pre-synthesized audio via Kokoro / Edge-TTS
lexis quiz Book_4_Unit_1 --template listening --count 10

# 5. Timestamp-Anchored Video Quiz (5 questions)
# Grounded chronological comprehension with video transcript anchors
lexis quiz Book_4_Unit_1 --template video --count 5
```

---

## 4. Importing Multimodal Media (Video & Audio)

Ingest web video or local media directly into the unit's `sources/media/` folder with synchronized companion transcripts:

```bash
# Import from a YouTube URL (fetches transcript automatically)
lexis video-import "https://www.youtube.com/watch?v=N3EqEocLPJk"

# Import from a Bilibili video URL
lexis video-import "https://www.bilibili.com/video/BV1xx411c7mD"

# Import a local video/audio file (transcribes on-the-fly via faster-whisper)
lexis video-import "lecture.mp4"

# Import a video with an existing subtitle file (bypasses transcription)
lexis video-import "lecture.mp4" --subtitle "subtitles.srt"
```

---

## 5. Quality Audits & Psychometric Inspection

Lexis features an automated **Two-Level Quality Audit Framework** (Deterministic Code Gate + Expert Model Blind Solving):

```bash
# 1. View LLM Hero Board & Audit Statistics
# Analyzes all generation execution logs under logs/ and prints pedagogical benchmarks
lexis audit

# 2. Batch Re-Audit Handouts
# Run the Level-2 Expert Judge across existing generated HTML quizzes
lexis re-audit

# 3. Targeted Re-Audit with Multi-Threaded Workers
# Re-audit reading comprehension quizzes for Book 4 Unit 1 using 4 concurrent threads
lexis re-audit Book_4_Unit_1 --template reading --workers 4
```

---

## 6. Maintenance, Refactoring & Linting

Keep the Obsidian-style knowledge base clean, consistent, and free of broken linkages:

```bash
# 1. Inspect Wiki Integrity
# Verifies bidirectional [[wikilinks]], YAML frontmatter, and orphan references
lexis lint

# 2. Fix & Normalize Tags
# Recursively cleans and formats tags into standard kebab-case
lexis lint --fix-tags

# 3. Prune Orphaned Artifacts
# Safely removes derived extractions/handouts whose source files no longer exist
lexis lint --prune

# 4. Safe Cascade Unit Renaming
# Renames a unit directory and automatically updates all internal filenames,
# frontmatter links, and media paths without breaking connections
lexis rename Book_4_Unit_1 Book_4_Unit_A
```
