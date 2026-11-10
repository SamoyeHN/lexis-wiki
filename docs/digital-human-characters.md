# Digital Human Educational Characters Architecture

> **Purpose**: Technical specification for dialogue cast archetypes, facial geometry standards, and multi-character rendering pipelines using Wan2.1 480P fp8 + InfiniteTalk on remote GPU nodes.

---

## 1. Dialogue Archetypes & Pedagogical Grounding

To power paired listening-speaking scenarios (e.g. Cambridge/IELTS Functional Dialogue, New Concept English dialogues, and SLA minimal-pair pragmatic exchanges), the character cast provides diverse ages, genders, and roles while adhering strictly to educational appropriateness.

### 1.1 Cast Archetypes

1. **Academic Instructor / Professional (Miss Green)**:
   - **Role**: Unit walkthrough, grammatical deconstruction, institutional representative.
   - **Visuals**: Tailored navy blazer, high-collared white blouse, calm intellectual expression.
2. **Young Collegiate Male (John)**:
   - **Role**: Inquirer, student peer, claimant (e.g., lost-and-found handbag dialogue).
   - **Visuals**: Clean collegiate sweater over light-blue oxford collar, crisp haircut, earnest demeanor.
3. **Young Collegiate Female (Alice)**:
   - **Role**: Cooperative interlocutor, study partner, library borrower.
   - **Visuals**: Modest beige knit cardigan over buttoned shirt, neat low ponytail, friendly and understated.
4. **Mature Gentleman / Officer (Mr. Blake)**:
   - **Role**: Cloakroom clerk (coat & umbrella), senior administrator, shopkeeper.
   - **Visuals**: Wireframe glasses, salt-and-pepper hair, tailored tweed waistcoat and blazer with tie.

---

## 2. Digital Human Face Invariants & Pre-Flight Checks

Before generating breathing video bases (`still_source.py`) and submitting audio-visual diffusion tasks to ComfyUI, every portrait must deterministically satisfy Level-1 geometric constraints via InsightFace (`face_check.py`):

$$\begin{aligned}
N_{\text{faces}} &= 1 \\
|\theta_{\text{eye\_line}}| &\le 2.0^\circ \\
|\text{yaw}| &\le 5.0^\circ \\
\text{face\_w} &\in [0.14, 0.25] \\
\text{mouth} &= \text{closed}
\end{aligned}$$

### Verification Audit Record

| Asset | Faces | Eye Line Tilt | Yaw Angle | Face Width Ratio | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Character_01_Miss_Green_896x512.png` | 1 | `-1.4°` | `-0.3°` | `0.146` | **PASS** |
| `Character_02_John_896x512.png` | 1 | `0.0°` | `+2.9°` | `0.212` | **PASS** |
| `Character_03_Alice_896x512.png` | 1 | `-1.8°` | `+0.9°` | `0.173` | **PASS** |
| `Character_04_Mr_Blake_896x512.png` | 1 | `-2.0°` | `-3.4°` | `0.173` | **PASS** |

---

## 3. Storage Layout & Asset Paths

```text
librarian/data/characters/
├── README.md                              # Directory manual & quick-start CLI
├── Characters_Dialogue_Cast_Grid.png      # 2x2 comparison contact sheet
├── Character_01_Miss_Green.png            # Master 1376x768 portrait
├── Character_01_Miss_Green_896x512.png    # Pre-cropped Wan2.1 480P ready
├── Character_02_John.png
├── Character_02_John_896x512.png
├── Character_03_Alice.png
├── Character_03_Alice_896x512.png
├── Character_04_Mr_Blake.png
└── Character_04_Mr_Blake_896x512.png
```

---

## 4. Multi-Turn Dialogue Video Generation Pipeline

In dual-speaker dialogues (e.g. Turn A: John, Turn B: Miss Green):
1. **Audio Segmentation**: Segment dialogue tracks into individual speaker turns with aligned timestamps.
2. **Breathing Source Generation**: Run `still_source.py` on the matching character image for the turn duration.
3. **InfiniteTalk Diffusion**: Submit audio + breathing video to Server 20 (`192.168.1.20:8188`) with `IT_START=2` and 0.16s English acoustic delay offset.
4. **Dialogue Concat / Shot Assembly**: Concatenate speaker video segments or render in split-screen/over-the-shoulder layouts into the unit's `handouts/<Unit>_listening_quiz.html` or `sources/media/`.

---

## 4. Multi-Character Same-Frame Interaction Pipeline (InfiniteTalk Multi)

When dialogic interactions require both speakers in the same frame (two-shot conversation without artificial splits or aspect ratio distortion):
1. **Base Two-Shot Composition**:
   - Synthesize a shared 16:9 canvas (e.g. 896x512) featuring Character A (Left) and Character B (Right) in an authentic educational environment (classroom, library, hallway) with closed-mouth resting state.
2. **Dual-Speaker Audio Stems**:
   - Master timeline $T = \sum (t_{\text{speech}} + t_{\text{pause}})$.
   - Stem 1 (Speaker A): active audio during Turn 1, padded with exact silence during Turn 2.
   - Stem 2 (Speaker B): exact silence during Turn 1, active audio during Turn 2.
3. **Per-Speaker Spatial Masks (`ref_target_masks`)**:
   - Batched tensor `[2, H, W]` where Channel 0 isolates Speaker A's head/torso and Channel 1 isolates Speaker B's head/torso.
   - Fed into `MultiTalkWav2VecEmbeds` (`multi_audio_type: "para"`).
4. **Diffusion Architecture**:
   - Model: `Wan2_1-InfiniteTalk-Multi_fp16.safetensors` loaded via `MultiTalkModelLoader`.
   - Sampler: `WanVideoSampler` (`dpm++_sde`, steps 4, start_step 2, shift 11.0, cfg 1.0).
   - Audio lip sync delay compensation: 0.16s acoustic offset muxed into final MP4.
5. **Auditing Standard**:
   - InsightFace (`buffalo_l`) cosine similarity $\ge 0.70$ tracked continuously across all turns for both characters.

