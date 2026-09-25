import unittest
from librarian.processor import WikiProcessor
from librarian.prompts import Prompts

SAMPLE_PASSAGE_A2 = """
In the past, it was a small stamp that helped family members and friends to keep in touch with each other.
With the coming of the telephone, people could not only read words, but also hear each other's voices.
Later, the rise of the Internet joins people in different places with instant communication software which gives people even more ways to communicate.
They want to follow the time closely, but they also long for peace of mind.
The new ways of communication make this possible.
"""

class TestReadingGateCEFRAdaptation(unittest.TestCase):
    def test_a2_clean_options_pass(self):
        """A2 reading quiz with simple vocabulary and concise options must pass audit."""
        quiz = {
            "questions": [
                {
                    "question": "What did people use in the past to keep in touch?",
                    "options": [
                        "A small stamp",
                        "A modern telephone",
                        "The instant Internet",
                        "A smart mobile phone"
                    ],
                    "correct_answer_index": 0,
                    "category": "Detail/Recall",
                    "explanation": "Paragraph 1 mentions that a small stamp helped family members keep in touch."
                },
                {
                    "question": "What is the main topic of the passage?",
                    "options": [
                        "Changes in communication ways",
                        "The history of post offices",
                        "Why people dislike telephones",
                        "How to build internet software"
                    ],
                    "correct_answer_index": 0,
                    "category": "Main Idea",
                    "explanation": "The passage describes how communication tools developed from stamps to telephone and internet."
                }
            ]
        }
        flagged, defects = WikiProcessor.audit_reading_integrity(
            quiz,
            passage_text=SAMPLE_PASSAGE_A2,
            cefr_level="A2"
        )
        self.assertEqual(flagged, [])
        self.assertEqual(defects, [])

    def test_a2_catches_c1_c2_super_advanced_word(self):
        """A2 reading quiz must reject C1/C2 distractors like 'conceivability' or 'paramount'."""
        quiz = {
            "questions": [
                {
                    "question": "Why did people use a stamp in the past?",
                    "options": [
                        "To send letters to family",
                        "To prove the conceivability of technology",
                        "To hear voices over long distances",
                        "To surf on the Internet"
                    ],
                    "correct_answer_index": 0,
                    "category": "Detail/Recall",
                    "explanation": "Option B contains conceivability."
                }
            ]
        }
        flagged, defects = WikiProcessor.audit_reading_integrity(
            quiz,
            passage_text=SAMPLE_PASSAGE_A2,
            cefr_level="A2"
        )
        self.assertIn(0, flagged)
        self.assertTrue(any("conceivability" in d and "exceeding CEFR A2 ceiling" in d for d in defects))

    def test_a2_catches_overly_long_option(self):
        """A2 reading quiz must reject bloated options exceeding 16 words."""
        quiz = {
            "questions": [
                {
                    "question": "What did people use in the past?",
                    "options": [
                        "A small stamp that was used every single day by almost all family members to write letters", # 17 words
                        "A telephone",
                        "The internet",
                        "A computer"
                    ],
                    "correct_answer_index": 0,
                    "category": "Detail/Recall",
                    "explanation": "Option A is bloated."
                }
            ]
        }
        flagged, defects = WikiProcessor.audit_reading_integrity(
            quiz,
            passage_text=SAMPLE_PASSAGE_A2,
            cefr_level="A2"
        )
        self.assertIn(0, flagged)
        self.assertTrue(any("Option is overly long for CEFR A2" in d for d in defects))

    def test_meta_instructional_words_not_falsely_flagged(self):
        """Standard testing words (according, infer, paragraph, author) must not trigger false alarms."""
        quiz = {
            "questions": [
                {
                    "question": "According to the author in paragraph 1, what helped family members?",
                    "options": [
                        "A small stamp",
                        "A smart telephone",
                        "A computer device",
                        "A voice recorder"
                    ],
                    "correct_answer_index": 0,
                    "category": "Detail/Recall",
                    "explanation": "Valid stem."
                }
            ]
        }
        flagged, defects = WikiProcessor.audit_reading_integrity(
            quiz,
            passage_text=SAMPLE_PASSAGE_A2,
            cefr_level="A2"
        )
        self.assertEqual(flagged, [])
        self.assertEqual(defects, [])

    def test_reading_prompt_interpolation_cefr_adaptive(self):
        """Prompt template must interpolate without KeyError for both A2 and B2 contexts."""
        raw_prompt, _ = Prompts.get("reading_quiz")
        
        # Test A2 interpolation
        kwargs_a2 = {
            "count": 5,
            "passage_content": SAMPLE_PASSAGE_A2,
            "cefr_level": "A2",
            "cefr_descriptor": "Foundational English (CEFR A2)",
            "vocab_target_guidance": "5 to 8 key functional vocabulary items",
            "skill_distribution_guidance": "Detail/Recall and Main Idea",
            "question_stem_guidance": "Direct question stems",
            "option_complexity_guidance": "Concise options <= 12 words"
        }
        formatted_a2 = raw_prompt.format(**kwargs_a2)
        self.assertIn("Foundational English (CEFR A2)", formatted_a2)
        self.assertIn("Concise options <= 12 words", formatted_a2)

        # Test B2 interpolation
        kwargs_b2 = {
            "count": 5,
            "passage_content": SAMPLE_PASSAGE_A2,
            "cefr_level": "B2",
            "cefr_descriptor": "Advanced Academic English (CEFR B2)",
            "vocab_target_guidance": "5 to 8 academic vocabulary items",
            "skill_distribution_guidance": "Full skill spectrum",
            "question_stem_guidance": "Academic synthesis question stems",
            "option_complexity_guidance": "Mature options with lexical substitutions"
        }
        formatted_b2 = raw_prompt.format(**kwargs_b2)
        self.assertIn("Advanced Academic English (CEFR B2)", formatted_b2)
        self.assertIn("Academic synthesis question stems", formatted_b2)


if __name__ == "__main__":
    unittest.main()
