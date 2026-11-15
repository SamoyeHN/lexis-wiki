"""
Test suite for Dashboard API endpoints and background workers.
"""

from unittest.mock import MagicMock, patch
from pathlib import Path
from librarian.processor import WikiProcessor
from librarian.dashboard import DashboardState, background_all_quizzes_worker


def test_wiki_processor_normalize_name():
    assert WikiProcessor.normalize_name("Book 1 Unit 1 Passage A") == "book1unit1passagea"
    p = WikiProcessor()
    assert p.normalize_name("Book_1_Unit_1") == "book1unit1"


def test_background_all_quizzes_worker():
    job_id = DashboardState.create_job("quiz", "Test_Unit (All Quizzes)")

    with patch.object(WikiProcessor, "generate_quiz", return_value="/path/to/Book_1_Unit_1_Passage_A_quiz.html") as mock_gen:
        background_all_quizzes_worker(job_id, "Book_1_Unit_1_Passage_A")

    job = DashboardState.get_job(job_id)
    assert job is not None
    assert job["status"] == "completed"
    assert job["progress"] == 100
    assert len(job["saved_files"]) >= 4
    assert mock_gen.call_count >= 4
