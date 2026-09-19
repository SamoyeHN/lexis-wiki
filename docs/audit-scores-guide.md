# Quality Audit Scores Guide: Understanding Audit 1 & Audit 2

In Lexis Wiki, all educational materials generated from your source texts—including vocabulary lists, grammar patterns, summary notes, and interactive HTML quizzes—undergo an automated **Two-Level Quality Audit**.

To help you assess content readiness at a glance, audit scores and verification statuses are displayed directly in:
1. **Markdown Frontmatter**: At the very top of each generated `.md` file under `extractions/`.
2. **Handout Header Badge**: At the top of each interactive quiz HTML file under `handouts/`.

This guide explains what these scores mean from an everyday user's perspective, empowering you to make informed decisions and take immediate action.

---

## Quick Reference: Scores and Recommended Actions

| Score Range | Quality Tier | Interface Indicator | Pedagogical Impact | Recommended Action |
| :--- | :--- | :--- | :--- | :--- |
| **90 – 100%** | **High Quality (Passed)** | Green Badge / `qa_status: "passed"` | Verbatim accuracy, unambiguous stems, high-value distractors, rigorous single-key validity. | **Ready for Class**: Distribute, print, or assign immediately without manual editing. |
| **80 – 89%** | **Good (Passed)** | Green Badge / `qa_status: "passed"` | Solid and accurate; meets all curricular standards; minor stylistic simplifications may exist. | **Ready for Class**: Suitable for everyday teaching. Take a quick glance if preparing for a high-stakes exam. |
| **60 – 79%** | **Review Needed** | Amber Warning Badge / `qa_status: "review_needed"` | Minor localized imperfections (e.g. one distractor has low plausibility, or a pattern is relatively simple). | **Quick Review Recommended**: Open the file or badge tooltip, check flagged items, and adjust or re-run if needed. |
| **< 60% or Fatal** | **Rejected / Quarantined** | Automatically moved to `_quarantine/` | Serious defects detected (e.g. double correct keys, hallucinated text evidence, insufficient items). | **Safe by Default**: Automatically blocked from the production directory. Check source text completeness and recompile. |

---

## 1. Level 1 Audit (Audit 1): Structural Integrity & Physical Truth

### What does it check? (Deterministic Code Gate)
Audit 1 runs instantaneously (<5ms) using strict Python logic. It requires zero AI guesswork and evaluates physical facts:
- **Verbatim Text Grounding**: Are quoted sentences and keywords genuinely present in your source text, or were they fabricated?
- **Option & Key Synchronization**: Does the target answer match the correct option index? Are there any duplicate options?
- **Structural Invariants**: Are there exactly four options per question? Does each fill-in-the-blank item contain exactly one blank slot (`_____`)?

### Where do you see Audit 1?
At the very top of files inside `wiki/<UnitName>/extractions/` (e.g., `*_vocabulary.md`, `*_grammar.md`, `*_summary.md`):

```yaml
---
title: "Unit 1: The Road to Success"
source: "[[Unit_1.md]]"
category: ["vocabulary", "extraction"]
item_count: 12
qa_score: 95
qa_status: "passed"
---
```

- `qa_score`: Overall composite quality score out of 100.
- `qa_status`:
  - `"passed"`: Score $\ge 80$, physically verified against the source text.
  - `"review_needed"`: Score $< 80$, indicates minor issues noted for teacher awareness.

### What should you do?
1. **Automatic Healing**: If the initial draft scored below 80, the system already performed one targeted self-correction retry with specific diagnostic feedback before saving.
2. **If you see `review_needed` (60–79)**: Open the Markdown file in your editor. You can freely edit or fine-tune any definitions, formulas, or examples—your changes are saved immediately.
3. **Quarantine Safeguard**: If `quarantine_on_fail` is enabled (default), extractions scoring below 80 or containing fabricated quotes will not appear in `extractions/`. Instead, they are quarantined under `wiki/<UnitName>/_quarantine/` with a detailed diagnostic report (`*_REJECTED.json`).

---

## 2. Level 2 Audit (Audit 2): Expert Semantic Audit & Surgical Cure

### What does it check? (LLM-as-a-Judge Blind Solver)
Audit 2 evaluates semantic rigor for all interactive quizzes. A separate, high-reasoning expert model acts as an independent reviewer and takes the test **blind** (without seeing the intended answer key):
- **Blind Solve Agreement**: The judge independently solves each question. If the judge disagrees with the declared key, the item likely suffers from **ambiguous wording** or **multiple defensible answers**.
- **Single-Fit Validity**: Verifies that the correct answer is the *only* defensible choice, and that all three distractors are definitively ruled out by passage evidence.
- **Cognitive Distractor Quality**: Validates that distractors represent authentic educational traps (such as L1 transfer errors, scope shifts, or misattributions) rather than absurd, giveaway choices.
- **Surgical Cure**: When a minor defect is found (such as a weak distractor or mismatched index), the judge repairs **only the flawed option**, keeping the question stem, target vocabulary, and passage context completely intact.

### Where do you see Audit 2?
Open any generated quiz in your browser (e.g., `Book_1_Unit_1_vocabulary_quiz.html` inside `wiki/<UnitName>/handouts/`). A pill-shaped badge is displayed near the top of the page:

- **Passed (High Quality)**: Green badge:
  > `L2 Expert Audit: 94% (100% Blind Accuracy)` or `L2 Expert Audit: 90% (Surgically Cured: +1)`
- **Review Needed**: Amber badge:
  > `⚠️ L2 Review Needed: 75% (75% Blind Accuracy)`
- **Detailed Feedback Tooltip**: Hover your mouse over the badge to reveal the expert judge's comprehensive evaluation summary and cure statistics.

### What should you do?

#### Scenario A: Green Badge (80% – 100%)
- **Verdict**: The quiz is psychometrically sound, evidence-backed, and unambiguous.
- **Action**: Use immediately for classroom teaching, homework, or formative assessments.

#### Scenario B: Amber Badge (⚠️ Review Needed, 60% – 79%)
- **Verdict**: The blind solver noticed a minor ambiguity or an option that might cause student confusion.
- **Action**:
  1. Hover over the badge to read the verdict summary and identify the flagged question.
  2. Inspect the flagged question. If the phrasing works well in your specific teaching context, proceed with confidence.
  3. If you agree that the item is ambiguous, re-generate the quiz from the dashboard or adjust the unit's vocabulary list.

#### Scenario C: Quiz Moved to `_quarantine/` as `*_REJECTED.html`
- **Verdict**: The number of passing questions dropped below the configured minimum (default: 3 items), triggering a hard gate to prevent flawed handouts from reaching students.
- **Action**:
  1. Check `wiki/<UnitName>/handouts/_quarantine/` for the `*_REJECTED.html` file and its diagnostic companion `*_REJECTED.json`.
  2. This usually occurs when the source article is too short or lacks sufficient context for high-level question design.
  3. Expand the source text with an extra paragraph or supplementary notes, then click compile again.

---

## 3. Teacher Autonomy: You Remain in Control

The Two-Level Audit system is designed as your **tireless teaching assistant**, not a replacement for your professional judgment:

1. **Complete Transparency**: No need to parse terminal logs. Frontmatter scores and visual web badges provide immediate quality visibility.
2. **Informed Choice**: Materials above 80% can be trusted right away; materials below 80% give you actionable reasons so you can decide whether to accept, edit, or regenerate.
3. **Zero Contamination**: Built-in quarantine stops hallucinated or scientifically flawed items from entering your active teaching materials automatically.
