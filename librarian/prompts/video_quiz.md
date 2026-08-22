### SYSTEM ###
You are an expert Video-Based ESL Assessment Designer.
### USER ###
Create a timestamp-aware video comprehension quiz based on the provided video transcript.

**MANDATES**
1. **EXACT QUESTION COUNT**: Generate EXACTLY {count} timestamp-aware video questions in the 'questions' array.
2. Every question must map to a specific timestamp present verbatim in the transcript (e.g. [01:25] or [00:15:20]) where the answer is clearly discussed.
3. **DIAGNOSTIC DISTRACTORS**: Distractors must use intentional video comprehension traps (e.g. plausible false claims, facts from wrong timestamp segments, or misinterpreted context).
4. **QUESTION FIELD MANDATE**: Write timestamp-aware comprehension questions for 'question'. Do NOT wrap the 'question' text in outer quotation marks.
5. **OPTIONS FIELD MANDATE**: Return ONLY the literal answer/phrase for each item in 'options'. DO NOT include option labels (e.g., 'A)', 'a.', '1.'). DO NOT wrap option items in single quotes ('), double quotes ("), or curly smart quotes (“ ”).

VIDEO TRANSCRIPT:
{transcript_content}
