# Paired Listening–Speaking Workbench Architecture

> **Purpose**: Definitive technical specification and pedagogical architecture for the **Paired Listening–Speaking Workbench** — a two-act receptive-to-productive learning engine that transforms listening dialogues into interactive Conversation Analysis (CA) microscopes, AI-generated situational dossiers, and paired information-gap role cards.

---

## 1. Problem Statement & Reality-Grounded Design Philosophy

### 1.1 The Receptive-to-Productive Chasm in SLA
In Second Language Acquisition (SLA), listening is **low-impedance decoding (Recognition)**, while speaking is **high-impedance encoding (Retrieval, Formulation, and Articulation)**. Traditional digital English systems suffer from two fatal fallacies:
1. **Mechanical Shadowing Fallacy (跟读陷阱，效用极低)**:
   Having students repeat audio transcripts line-by-line builds acoustic muscle memory but zero communicative competence. The moment the screen is turned off, the learner cannot produce spontaneous speech in a real conversation.
2. **Unconstrained Free-Chat Fallacy (失控自由对话陷阱，脱钩教学大纲)**:
   Throwing learners into open-ended voice chats with an AI causes cognitive overload. Under pressure, learners reflexively retreat to their safest A1 vocabulary, completely avoiding the target vocabulary and pragmatic structures introduced in the unit.

### 1.2 The Hard Deployment Constraint: Rejection of Open-Mic Real-Time ASR
In actual classroom, lab, and web handout deployment:
- **Environmental Noise & Hardware Gaps**: Classrooms with 40+ students produce catastrophic acoustic cross-talk; browser microphone permissions and cross-platform WebRTC APIs introduce severe brittleness.
- **Latency Destroys Turn-Taking**: Round-trip cloud ASR + LLM evaluation takes 2–4 seconds, which shatters the natural cadence of conversational turn-taking (human turn-transitions average <250ms).
- **Affective Filter & Speaking Resistance**: Beginners and intermediate students often freeze when required to speak to a machine without structured scaffolding.

### 1.3 The Grounded Utilitarian Alternative
The Paired Listening–Speaking Workbench operates under a clean, deterministic, and highly practical paradigm:
**Display-Driven Pragmatic Scaffolding + Paired Human-to-Human (or Dual-Role) Role-Play**:
- **Act I (Listening Comprehension)**: Receptive input with psychometric signpost questions (`Bait → Real Key`).
- **Act II (Paired Speaking Workbench)**: 
  1. **Turn Microscope (`对话显微镜`)**: Visual deconstruction of Conversation Analysis (CA) turn structures.
  2. **AI Dynamic Scene Dossier (`情景档案`)**: Context, urgency, social register, and acoustic atmosphere.
  3. **Paired Jigsaw Role Cards (`双向信息差角色卡`)**: Asymmetric facts and goal-driven cue cards for Role A and Role B.
  4. **Role Reversal & Slot Swapping (`角色对调与变量置换`)**: Re-deploying the exact same 4 functional skeletons with altered variables to achieve automaticity.

---

## 2. Neuro-Symbolic Quad-Tier Workbench Pipeline

```mermaid
flowchart TD
    subgraph Tier 1: Symbolic Anchors [Tier 1: Symbolic Grounding (0 Tokens, 0ms)]
        UT["Unit Title Intent<br/>(e.g. Unit 3: How Can I Get There?)"] --> FA["Functional Archetype<br/>(arch_02: Route Options & Wayfinding)"]
        FA --> SK["4 Bi-directional Skeletons<br/>(e.g. be tight on, rule out, work around, watch out for)"]
        SK --> LV["Unit Target Vocabulary & Grammar"]
    end

    subgraph Tier 2: AI Co-generation [Tier 2: Co-generation Engine (One-Shot LLM Call)]
        Blueprint["LLM Micro-Task Blueprint"]
        Tier 1 --> Blueprint
        Blueprint --> SD["AI Scene Dossier (情景档案)<br/>Setting, Social Register, Urgency & Information Gap"]
        Blueprint --> DT["8-Turn Functional Dialogue<br/>Structured with Cushion + Skeleton + Handover"]
        Blueprint --> PQ["5 Psychometric Cambridge Listening Questions"]
    end

    subgraph Tier 3: Artifact Delivery [Tier 3: Multi-View Handout Presentation]
        SD & DT --> VM["View 1: Turn Microscope (对话显微镜)"]
        SD & DT --> RC["View 2: Paired Jigsaw Role Cards (双向角色卡)"]
        SD & DT --> RR["View 3: Role Reversal & Slot Swapper (对调与变体)"]
        PQ --> LQ["Act I: Audio Player & Interactive Quiz"]
    end
```

### 2.1 "One-Stone, Two-Birds" Co-generation Architecture (Zero Extra LLM Cost)
Rather than executing a separate LLM call for speaking after generating the listening dialogue, the LLM is prompted to **construct the scene dossier first** as cognitive grounding, then write the dialogue within that scene. 
- **Cognitive Benefit for LLM**: When the model explicitly defines the time pressure, physical props, and information asymmetry before drafting lines, the resulting dialogue is vastly more realistic and pragmatically anchored.
- **Zero Token Overhead**: The single generation pass produces both the listening test assets and the speaking workbench cards simultaneously.

---

## 3. Structural Schemas & JSON Specifications

### 3.1 AI Dynamic Scene Dossier Schema (`scenario_environment`)
```json
{
  "scenario_environment": {
    "setting_profile": {
      "location": "A bustling metro transfer hall customer service counter",
      "time_and_atmosphere": "Friday 17:45, pouring rain outside, crowds rushing for weekend trains",
      "ambient_soundscape_cue": "metro_station_rush_hour"
    },
    "social_distance_and_register": {
      "relationship": "Stranger to public service personnel",
      "power_and_social_distance": "Equal social distance; staff holds informational authority",
      "tone_and_register": "Polite, semi-formal, urgent yet cooperative"
    },
    "inciting_tension": {
      "urgency_and_stakes": "High: Speaker A has 35 minutes to reach the high-speed rail terminal before final boarding",
      "unforeseen_barrier": "Direct line (Line 2) is experiencing a severe signal failure with indefinite delays"
    },
    "information_gap": {
      "role_a_private_dossier": {
        "identity": "Out-of-town conference attendee with heavy luggage",
        "private_facts": [
          "Target destination: East Railway Station",
          "Hard deadline: Must arrive by 18:20 (35 mins left)",
          "Budget flexibility: Willing to pay extra for an express shuttle if it guarantees arrival"
        ],
        "primary_objective": "Find the fastest reliable transit route and verify the exact platform/transfer gate."
      },
      "role_b_private_dossier": {
        "identity": "Senior transit information officer",
        "private_facts": [
          "Line 2 (direct subway) is stalled for at least 45 minutes (unacceptable for Role A)",
          "Bypass route: Line 4 (3 stops to Terminal A) + Airport Express Shuttle takes only 22 minutes",
          "Crucial caveat: Shuttle requires exact cash or instant mobile transit code at Gate C"
        ],
        "primary_objective": "De-escalate passenger panic and guide them through the Line 4 bypass."
      }
    },
    "visual_scene_prompt": "Cinematic wide shot of a modern glass subway station information desk at dusk, blurred crowds with raincoats and umbrellas, glowing digital departure schedule boards, photorealistic."
  }
}
```

### 3.2 CA Three-Part Turn Micro-Structure Schema (`dialogue_turns`)
Each turn in the 8-turn dialogue is parsed into Conversation Analysis functional trios:
```json
{
  "turn_index": 3,
  "speaker": "Speaker A",
  "cushion": "To be completely candid with you,",
  "core_skeleton": "we're extraordinarily tight on time,",
  "propositional_body": "since my high-speed train boards in less than thirty-five minutes.",
  "handover_hook": "Is there any alternative way around the stalled line?",
  "full_turn": "To be completely candid with you, we're extraordinarily tight on time, since my high-speed train boards in less than thirty-five minutes. Is there any alternative way around the stalled line?"
}
```

### 3.3 Paired Speaking Cues & Milestone Stages (`paired_role_cards`)
```json
{
  "role_a_card": {
    "role_title": "Passenger / Traveler",
    "cues_by_phase": [
      {
        "phase": 1,
        "phase_name": "Inquiry & Stating Urgency (破冰与求助)",
        "goal": "Politely interrupt the staff, announce your destination, and state your tight time constraint.",
        "scaffolding_chunk": "Excuse me, do you happen to know...? / I'm extraordinarily tight on [time]..."
      },
      {
        "phase": 2,
        "phase_name": "Conflict & Seeking Trade-offs (方案协商)",
        "goal": "React to the delay, rule out waiting, and ask for an expedited bypass route.",
        "scaffolding_chunk": "We certainly can't rule out [plan]... / Could we work around this by [action]...?"
      },
      {
        "phase": 3,
        "phase_name": "Locking Agreement & Action (确认与收尾)",
        "goal": "Confirm the exact transfer gate and express gratitude.",
        "scaffolding_chunk": "Sounds like a solid plan. / I'll head straight for [Gate C]."
      }
    ]
  },
  "role_b_card": {
    "role_title": "Information Officer",
    "cues_by_phase": [
      {
        "phase": 1,
        "phase_name": "Receipt & Assessing Constraint (接待与核对)",
        "goal": "Acknowledge the traveler, verify the timeline, and deliver the signal-failure news.",
        "scaffolding_chunk": "Sure thing. / While [Line 2] is stalled, ..."
      },
      {
        "phase": 2,
        "phase_name": "Offering Alternative Bypass (提供备用捷径)",
        "goal": "Present the Line 4 + Express Shuttle combination with duration trade-offs.",
        "scaffolding_chunk": "Your best bet is to [action]... / Watch out for [prerequisite]..."
      },
      {
        "phase": 3,
        "phase_name": "Final Guidance & Signpost (指路送行)",
        "goal": "Give directions to the exit gate and send them off.",
        "scaffolding_chunk": "Take Exit 3 right behind you. / Safe travels!"
      }
    ]
  }
}
```

### 3.4 Variable Swapping for Automaticity Re-drill (`variable_swap_profile`)
```json
{
  "slot_swaps": [
    {
      "original_variable": "East Railway Station (35 minutes deadline)",
      "swapped_variable": "Terminal 2 Airport Check-in (50 minutes deadline)"
    },
    {
      "original_variable": "Signal failure on Line 2",
      "swapped_variable": "Track maintenance on Airport Line A"
    },
    {
      "original_variable": "Line 4 + Express Shuttle",
      "swapped_variable": "Maglev Express + Downtown Bus #1"
    }
  ],
  "pedagogical_instruction": "Swap roles! The previous Speaker A now plays Officer (Role B), and Speaker B plays Traveler (Role A). Deploy the same 4 functional skeletons with the updated variables above."
}
```

---

## 4. Multi-View Handout Presentation (HTML & Print)

### 4.1 View 1: Turn Microscope (`对话显微镜`)
Interactive CSS toggles allow students or teachers to switch between **Narrative View** and **Microscope View**:
- <span style="background-color: #fff3cd; padding: 2px 5px; border-radius: 3px; font-weight: bold;">Yellow Highlight</span>: **Cushion (听觉与思考缓冲垫)** — primes listener working memory for 0.8s.
- <span style="background-color: #cce5ff; padding: 2px 5px; border-radius: 3px; font-weight: bold;">Blue Highlight</span>: **Core Pragmatic Skeleton (核心语用骨架)** — carrier of the unit's key functional intent.
- <span style="background-color: #d4edda; padding: 2px 5px; border-radius: 3px; font-weight: bold;">Green Highlight</span>: **Handover Hook (轮元抓手)** — passes conversational initiative back to partner.

### 4.2 View 2: Paired Jigsaw Role Cards (`双向角色卡`)
A responsive grid renders two high-contrast cards side-by-side:
- **Card A (Role A)**: Colored in Slate Blue. Displays Private Facts, Target Goal, and Phase 1–3 speaking skeleton scaffolds.
- **Card B (Role B)**: Colored in Forest Green. Displays Private Facts, Caveats, and Phase 1–3 response skeleton scaffolds.
- **Strict Prohibition**: Neither card displays full verbatim sentences. Only semi-open skeletons with slot placeholders `[...]` are provided to prevent script memorization.

### 4.3 View 3: Role Reversal & Automaticity Drill (`对调与变体`)
A floating button **`[ ⇄ 对调角色并刷新变量 ]`** triggers:
1. Card A and Card B swap positions and color themes.
2. Slot variables dynamically refresh from `variable_swap_profile`.
3. Working memory is forced to recombine the same 4 syntactic frames with new lexical entities.

### 4.4 Classroom Print-Ready Mode (`@media print`)
When printed or exported to PDF:
- The page formats as an A4 sheet split down the middle with a dotted scissor line (`✂️ 沿虚线对折/撕开，两人对练`).
- Left half: Student 1 (Role A dossier & cues).
- Right half: Student 2 (Role B dossier & cues).
- Zero digital device or internet connection required in offline classrooms.

---

## 5. Implementation Roadmap & Verification Milestones

1. **Phase 1: Pipeline & Prompt Unification**
   - Extend `build_listening_micro_task_blueprints` in [librarian/linguistics_core/quiz_listening.py](file:///home/byteuser/projects/lexis-wiki/librarian/linguistics_core/quiz_listening.py) to incorporate `scenario_environment`, `turn_microstructure`, and `paired_role_cards` in the Prompt-Guided JSON schema.
2. **Phase 2: Code Gate Invariant Auditing**
   - Extend [librarian/processor_core/evaluator.py](file:///home/byteuser/projects/lexis-wiki/librarian/processor_core/evaluator.py) with deterministic code gates:
     - `assert len(role_a_cues) == len(role_b_cues) == 3` (Exactly 3 milestones).
     - `assert all(sk in dialogue_text for sk in selected_skeletons)` (Skeleton presence check).
     - Information asymmetry verification (Role A facts $\cap$ Role B facts has zero leaked solutions).
3. **Phase 3: HTML Handout Template Upgrade**
   - Update listening quiz HTML generator to append the `#paired-speaking-workbench` section beneath the listening comprehension test.
   - Inject Turn Microscope CSS styling, dual-card grid, and role-swap JavaScript interactivity.
4. **Phase 4: Book 1 Pilot Verification**
   - Run end-to-end compilation for `Book_1_Unit_1_Passage_A` through `Book_1_Unit_6_Passage_A`.
   - Audit Level 1 Code Gate pass rate (100% target).
