import unittest
from librarian.processor import WikiProcessor

class TestAssessmentGatesCEFR(unittest.TestCase):
    """
    Test suite for Level 1 Deterministic Code Gate CEFR adaptation
    across Translation, Listening, and Video assessment modalities.
    """

    def test_listening_a2_clean_options_pass(self):
        quiz = {
            "script": [
                {"speaker": "A", "text": "Hello, can I help you find a book?"},
                {"speaker": "B", "text": "Yes, I am looking for the history section."},
                {"speaker": "A", "text": "It is on the second floor near the stairs."},
                {"speaker": "B", "text": "Thank you very much for your help."}
            ],
            "questions": [
                {
                    "question": "Where is the history section located?",
                    "options": [
                        "On the second floor",
                        "In the basement",
                        "Near the entrance",
                        "Next to the cafe"
                    ],
                    "correct_answer_index": 0,
                    "category": "Detail"
                }
            ]
        }
        flagged, defects = WikiProcessor.audit_listening_integrity(quiz, cefr_level="A2")
        self.assertEqual(flagged, [])
        self.assertEqual(defects, [])

    def test_listening_a2_catches_c1_c2_word(self):
        quiz = {
            "script": [
                {"speaker": "A", "text": "Hello, can I help you find a book?"},
                {"speaker": "B", "text": "Yes, I am looking for the history section."},
                {"speaker": "A", "text": "It is on the second floor."},
                {"speaker": "B", "text": "Thank you very much."}
            ],
            "questions": [
                {
                    "question": "Where is the history section?",
                    "options": [
                        "On the second floor",
                        "In an enigmatic corner",  # 'enigmatic' is C1/C2
                        "Near the entrance",
                        "Next to the cafe"
                    ],
                    "correct_answer_index": 0,
                    "category": "Detail"
                }
            ]
        }
        flagged, defects = WikiProcessor.audit_listening_integrity(quiz, cefr_level="A2")
        self.assertIn(0, flagged)
        self.assertTrue(any("enigmatic" in d and "exceeding CEFR A2 ceiling" in d for d in defects))

    def test_listening_a2_catches_overly_long_option(self):
        quiz = {
            "questions": [
                {
                    "question": "Where is the section?",
                    "options": [
                        "On the second floor right next to the stairs where people walk every single afternoon", # 15 words (> 12 words)
                        "In the basement",
                        "Near the entrance",
                        "Next to cafe"
                    ],
                    "correct_answer_index": 0,
                    "category": "Detail"
                }
            ]
        }
        flagged, defects = WikiProcessor.audit_listening_integrity(quiz, cefr_level="A2")
        self.assertIn(0, flagged)
        self.assertTrue(any("overly long for CEFR A2" in d for d in defects))

    def test_video_a2_clean_options_pass(self):
        quiz = {
            "questions": [
                {
                    "question": "What is the speaker holding?",
                    "timestamp": "[01:15]",
                    "options": [
                        "A wooden brush",
                        "A modern phone",
                        "A glass cup",
                        "A small notebook"
                    ],
                    "correct_answer_index": 0
                }
            ]
        }
        flagged, defects = WikiProcessor.audit_video_integrity(quiz, cefr_level="A2")
        self.assertEqual(flagged, [])
        self.assertEqual(defects, [])

    def test_video_a2_catches_c1_c2_word(self):
        quiz = {
            "questions": [
                {
                    "question": "What is the speaker holding?",
                    "timestamp": "[01:15]",
                    "options": [
                        "A wooden brush",
                        "A ubiquitous device", # 'ubiquitous' is C1/C2
                        "A glass cup",
                        "A small notebook"
                    ],
                    "correct_answer_index": 0
                }
            ]
        }
        flagged, defects = WikiProcessor.audit_video_integrity(quiz, cefr_level="A2")
        self.assertIn(0, flagged)
        self.assertTrue(any("ubiquitous" in d and "exceeding CEFR A2 ceiling" in d for d in defects))

    def test_video_a2_catches_overly_long_option(self):
        quiz = {
            "questions": [
                {
                    "question": "What is shown in the video?",
                    "timestamp": "[01:15]",
                    "options": [
                        "A very old wooden brush used by traditional artists in the past during daily painting sessions", # 16 words (> 14 words)
                        "A smartphone",
                        "A cup",
                        "A notebook"
                    ],
                    "correct_answer_index": 0
                }
            ]
        }
        flagged, defects = WikiProcessor.audit_video_integrity(quiz, cefr_level="A2")
        self.assertIn(0, flagged)
        self.assertTrue(any("overly long for CEFR A2" in d for d in defects))

    def test_translation_a2_clean_options_pass(self):
        quiz = {
            "questions": [
                {
                    "translated_sentence": "他在图书馆认真阅读了一本书。",
                    "target_keyword": "library",
                    "options": [
                        "He read a book carefully in the library.",
                        "He read a book in the store."
                    ],
                    "correct_answer_index": 0,
                    "idiomatic_translation": "He read a book carefully in the library.",
                    "flawed_translation": "He read a book in the store."
                }
            ]
        }
        flagged, defects = WikiProcessor.audit_translation_integrity(quiz, cefr_level="A2")
        self.assertEqual(flagged, [])
        self.assertEqual(defects, [])

    def test_translation_a2_catches_c1_c2_word(self):
        quiz = {
            "questions": [
                {
                    "translated_sentence": "他在图书馆认真阅读了一本书。",
                    "target_keyword": "library",
                    "options": [
                        "He perused the volume meticulously in the library.", # 'peruse' / 'meticulously'
                        "He read a book in the store."
                    ],
                    "correct_answer_index": 0,
                    "idiomatic_translation": "He perused the volume meticulously in the library.",
                    "flawed_translation": "He read a book in the store."
                }
            ]
        }
        flagged, defects = WikiProcessor.audit_translation_integrity(quiz, cefr_level="A2")
        self.assertIn(0, flagged)
        self.assertTrue(any("exceeding CEFR A2 ceiling" in d for d in defects))

    def test_translation_a2_catches_overly_long_option(self):
        quiz = {
            "questions": [
                {
                    "translated_sentence": "他在图书馆认真阅读了一本书。",
                    "target_keyword": "library",
                    "options": [
                        "He read an interesting book very carefully in the quiet library with all his friends yesterday afternoon", # 18 words (> 16 words)
                        "He read a book in the store."
                    ],
                    "correct_answer_index": 0,
                    "idiomatic_translation": "He read an interesting book very carefully in the quiet library with all his friends yesterday afternoon",
                    "flawed_translation": "He read a book in the store."
                }
            ]
        }
        flagged, defects = WikiProcessor.audit_translation_integrity(quiz, cefr_level="A2")
        self.assertIn(0, flagged)
        self.assertTrue(any("overly long for CEFR A2" in d for d in defects))

    # =========================================================================
    # 5. SYMBOLIC LISTENING MICRO-TASK BLUEPRINT & PRAGMATIC STATE MACHINE TESTS
    # =========================================================================
    def test_listening_micro_task_blueprint_generation_a2_and_b2(self):
        from librarian.linguistics import LinguisticEngine

        sample_vocab_md = """
### [[compromise]]
- **Part of Speech**: verb
- **Definition**: to reach an agreement by giving up part of what you want
- **Quoted Sentence**: "Both sides had to compromise on the schedule."

### [[originate]]
- **Part of Speech**: verb
- **Definition**: to come from a particular place or start in a particular situation
- **Quoted Sentence**: "The tradition originated in small coastal towns."

### [[reluctant]]
- **Part of Speech**: adjective
- **Definition**: slow and unwilling to do something
- **Quoted Sentence**: "She was reluctant to accept the extra shift."

### [[postpone]]
- **Part of Speech**: verb
- **Definition**: to delay an event and plan for it to happen at a later time
- **Quoted Sentence**: "They decided to postpone the seminar until Friday."

### [[beneficial]]
- **Part of Speech**: adjective
- **Definition**: having a good or useful effect
- **Quoted Sentence**: "Regular feedback is beneficial to every student."
"""
        for cefr in ("A2", "B2"):
            bp = LinguisticEngine.build_listening_micro_task_blueprints(
                sample_vocab_md,
                thematic_topic="1. Project Schedule Coordination",
                target_count=5,
                cefr_level=cefr,
            )
            self.assertIsInstance(bp, dict)
            self.assertTrue(bp.get("archetype_id"))
            self.assertTrue(bp.get("trajectory_id"))
            self.assertEqual(len(bp["turn_specs"]), 8)
            self.assertEqual(len(bp["questions"]), 5)
            self.assertIn("FUNCTIONAL DIALOGUE ARCHETYPE & 8-TURN PRAGMATIC TRAJECTORY BLUEPRINT", bp["combined_prompt_block"])
            self.assertIn("Flexible Spoken Patterns", bp["combined_prompt_block"])
            # Ensure all 4 option indices (0, 1, 2, 3) are represented across 5 questions
            prescribed_indices = {q["correct_answer_index"] for q in bp["questions"]}
            self.assertEqual(prescribed_indices, {0, 1, 2, 3})
            # Check each question has immutable anchor_turn, target_skill, and 4 slot operators
            for q_spec in bp["questions"]:
                self.assertIn(q_spec["target_skill"], ("Detail", "Main Idea", "Inference"))
                self.assertTrue(q_spec.get("anchor_turn"))
                self.assertEqual(len(q_spec["slot_operators"]), 4)
                corr_op = q_spec["slot_operators"][q_spec["correct_answer_index"]]
                self.assertEqual(corr_op, "Paraphrase Operator (Correct Answer)")

    def test_hydrate_listening_blueprint_items_swaps_and_remaps_explanation(self):
        quiz_obj = {
            "title": "Seminar Schedule",
            "script": [
                {"speaker": "Speaker 1", "text": "We need to postpone the seminar until Friday."},
                {"speaker": "Speaker 2", "text": "I am reluctant to change the room booking."},
                {"speaker": "Speaker 1", "text": "Could we compromise on Thursday afternoon instead?"},
                {"speaker": "Speaker 2", "text": "That sounds beneficial for everyone involved."},
            ],
            "questions": [
                {
                    "design_audit": "AUDIT: [Speaker 1, Turn 1] -> Detail -> Speaker Swap",
                    "question": "What does Speaker 1 initially propose doing?",
                    "options": [
                        "Moving the seminar to Friday.",
                        "Keeping the current room booking.",
                        "Canceling the seminar completely.",
                        "Holding the meeting on Monday morning.",
                    ],
                    "correct_answer_index": 0,
                    "category": "Inference",  # Intentional mismatch to test category enforcement
                    "explanation": "Option A is the correct answer because Speaker 1 proposes moving the seminar to Friday. Option C overstates the plan.",
                }
            ],
        }
        blueprint_specs = [
            {
                "question_index": 1,
                "target_skill": "Detail",
                "correct_answer_index": 2,  # Prescribed index C (2)
            }
        ]
        hydrated = WikiProcessor.hydrate_listening_blueprint_items(quiz_obj, blueprint_specs)
        q0 = hydrated["questions"][0]
        self.assertEqual(q0["category"], "Detail")
        self.assertEqual(q0["correct_answer_index"], 2)
        self.assertEqual(q0["options"][2], "Moving the seminar to Friday.")
        self.assertEqual(q0["options"][0], "Canceling the seminar completely.")
        self.assertIn("Option C is the correct answer", q0["explanation"])
        self.assertIn("Option A overstates the plan", q0["explanation"])
        self.assertNotIn("target_word", q0)

    def test_listening_gate_catches_verbatim_turn_copy(self):
        quiz = {
            "script": [
                {"speaker": "Speaker 1", "text": "We should definitely postpone the big history seminar until next Friday."},
                {"speaker": "Speaker 2", "text": "I agree that extra time will help us prepare better."},
                {"speaker": "Speaker 1", "text": "Great, I will email the department secretary right away."},
                {"speaker": "Speaker 2", "text": "Thanks for taking care of the room reservation today."},
            ],
            "questions": [
                {
                    "question": "What does Speaker 1 suggest doing about the history seminar?",
                    "options": [
                        "postpone the big history seminar until next Friday",
                        "Move the history seminar to an online platform",
                        "Cancel the history seminar for the whole semester",
                        "Ask the department secretary to lead the seminar",
                    ],
                    "correct_answer_index": 0,
                    "category": "Detail",
                }
            ],
        }
        flagged, defects = WikiProcessor.audit_listening_integrity(quiz, cefr_level="B1")
        self.assertIn(0, flagged)
        self.assertTrue(any("verbatim copy of a dialogue turn" in d for d in defects))

    def test_listening_vocab_selector_prioritizes_pedlex_s1_mwe_and_awl_over_technical_nouns(self):
        from librarian.linguistics import LinguisticEngine
        mixed_items = [
            {"word": "reactor", "pos": "noun", "definition": "a large machine in which atoms are split", "quote": "Uranium is tapped to fuel nuclear reactors.", "example": "a nuclear reactor", "audit": ""},
            {"word": "hitherto", "pos": "adverb", "definition": "until this time", "quote": "Fossil fuels hitherto proved reliable.", "example": "a hitherto unknown species", "audit": ""},
            {"word": "fill up", "pos": "phrasal verb", "definition": "if a container fills up, it becomes full", "quote": "Restaurants fill up quickly.", "example": "The room soon filled up with people.", "audit": ""},
            {"word": "on a scale", "pos": "collocation", "definition": "used to measure the size or extent of something", "quote": "Making it work on a large scale is hard.", "example": "pollution on a global scale", "audit": ""},
            {"word": "derive from", "pos": "phrasal verb", "definition": "to come or develop from something", "quote": "Energy derived from the sun.", "example": "words derived from Latin", "audit": ""},
            {"word": "controversial", "pos": "adjective", "definition": "causing a lot of disagreement", "quote": "Uranium is controversial.", "example": "a highly controversial decision", "audit": ""},
            {"word": "reliable", "pos": "adjective", "definition": "able to be trusted or depended on", "quote": "Fossil fuels proved reliable.", "example": "a reliable source of information", "audit": ""},
        ]
        selected = LinguisticEngine._select_listening_vocab_by_pedlex_freq(
            mixed_items,
            cefr_level="C1",
            max_items=5,
        )
        selected_words = [it["word"] for it in selected]
        self.assertEqual(len(selected_words), 5)
        self.assertIn("fill up", selected_words)
        self.assertIn("on a scale", selected_words)
        self.assertNotIn("reactor", selected_words)
        self.assertNotIn("hitherto", selected_words)


if __name__ == "__main__":
    unittest.main()
