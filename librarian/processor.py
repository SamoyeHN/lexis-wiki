"""
WikiProcessor facade for Lexis Wiki.
Coordinates deterministic extraction, multi-modal quiz generation, Level 1 & Level 2 QA auditing,
and Markdown/HTML artifact rendering.
"""

import random
import os
import json
import re
import dataclasses
import base64
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor
from .config import config
from .llm import llm
from .prompts import Prompts
from .linguistics import LinguisticEngine
from .schemas import (
    VocabularyExtraction, GrammarExtraction, SummaryExtraction,
    VocabularyItem, GrammarItem,
    VocabularyQuiz, ReadingQuiz, TranslationQuiz, ListeningQuiz,
    RoutingResult, MindMapExtraction,
    validate_and_map
)
from .processor_core.syllabus_and_markdown import SyllabusAndMarkdownMixin
from .processor_core.extraction_runner import ExtractionRunnerMixin
from .processor_core.quiz_generator import QuizGeneratorMixin
from .processor_core.quiz_integrity_gates import QuizIntegrityGatesMixin

logger = logging.getLogger("librarian.processor")


class WikiProcessor(
    SyllabusAndMarkdownMixin,
    ExtractionRunnerMixin,
    QuizGeneratorMixin,
    QuizIntegrityGatesMixin,
):
    """
    Unified WikiProcessor facade combining syllabus/Markdown I/O, extraction pipeline runner,
    quiz generation, and deterministic Level 1 integrity gates.
    """
    pass


processor = WikiProcessor()
