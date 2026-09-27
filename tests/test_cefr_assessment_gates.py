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

if __name__ == "__main__":
    unittest.main()
