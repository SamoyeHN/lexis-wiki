"""TTS audio synthesis service for multi-modal quizzes (listening quiz dialogue).

Handles speaker gender inference, metadata resolution, dialogue script synthesis,
and MP3 base64 data URI encoding.
"""

from __future__ import annotations

import base64
import dataclasses
import logging
import re
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("librarian.processor")

_COMMON_FEMALE_NAMES = {
    "alice", "anna", "aria", "ava", "bella", "chloe", "claire", "clara", "diana",
    "elena", "ella", "emily", "emma", "eva", "grace", "hannah", "helen", "iris", "isabella",
    "jade", "jenny", "jessica", "julia", "karen", "kate", "laura", "leah", "libby", "lily",
    "linda", "lisa", "lucy", "maria", "maya", "mei", "mia", "michelle", "naomi", "natasha",
    "nicole", "nina", "nora", "olivia", "rachel", "rose", "ruby", "samantha",
    "sara", "sarah", "sofia", "sonia", "sophia", "stella", "tina", "victoria", "wendy", "zoe",
}

_COMMON_MALE_NAMES = {
    "adam", "alex", "andrew", "aris", "ben", "brian", "charles", "chris", "christopher",
    "dan", "daniel", "david", "edward", "eric", "ethan", "evan", "frank", "george", "guy",
    "harry", "henry", "ian", "isaac", "jack", "jacob", "james", "jason", "john", "jordan",
    "josh", "kevin", "leo", "lewis", "liam", "lucas", "luke", "marcus", "mark", "matt",
    "matthew", "max", "michael", "mike", "nathan", "nick", "noah", "oliver", "oscar",
    "owen", "paul", "peter", "ray", "richard", "robert", "roger", "ryan", "sam", "samuel",
    "scott", "sean", "simon", "steffan", "stephen", "steve", "steven", "thomas", "tim",
    "tom", "victor", "william",
}


def infer_name_gender(full_name: str) -> Optional[str]:
    """Infers speaker gender ('female' or 'male') from first token of name."""
    tokens = [
        re.sub(r"[^a-zA-Z]", "", t).lower()
        for t in str(full_name or "").split()
    ]
    tokens = [t for t in tokens if t and t not in ("dr", "prof", "mr", "ms", "mrs", "miss", "speaker")]
    if not tokens:
        return None
    first = tokens[0]
    if first in _COMMON_FEMALE_NAMES:
        return "female"
    if first in _COMMON_MALE_NAMES:
        return "male"
    return None


def resolve_speaker_metadata(
    s1_name: str,
    s2_name: str,
    raw_g1: Any,
    raw_g2: Any,
    raw_r1: Any,
    raw_r2: Any,
    default_g1: str = "female",
    default_g2: str = "male",
    tts_service: Any = None,
) -> Tuple[str, str, str, str, str, str]:
    """Resolves speaker genders, accents, and roles with fallback synchronization."""
    g1_norm = str(raw_g1 or default_g1 or "female").strip().lower()
    g2_norm = str(raw_g2 or default_g2 or "male").strip().lower()
    if g1_norm not in ("female", "male"):
        g1_norm = "female"
    if g2_norm not in ("female", "male"):
        g2_norm = "male" if g1_norm == "female" else "female"

    inferred_1 = infer_name_gender(s1_name)
    inferred_2 = infer_name_gender(s2_name)
    if inferred_1 and inferred_2 and inferred_1 != inferred_2:
        g1_norm, g2_norm = inferred_1, inferred_2
    elif inferred_1 and inferred_1 != g1_norm:
        g1_norm = inferred_1
        g2_norm = "male" if g1_norm == "female" else "female"
    elif inferred_2 and inferred_2 != g2_norm:
        g2_norm = inferred_2
        g1_norm = "male" if g2_norm == "female" else "female"
    elif g1_norm == g2_norm:
        g2_norm = "male" if g1_norm == "female" else "female"

    acc_female = "en-US"
    acc_male = "en-US"
    if tts_service is not None:
        try:
            acc_female = tts_service.get_accent(tts_service.voice_a)
            acc_male = tts_service.get_accent(tts_service.voice_b)
        except (AttributeError, TypeError):
            pass

    a1_resolved = acc_female if g1_norm == "female" else acc_male
    a2_resolved = acc_female if g2_norm == "female" else acc_male

    r1_clean = str(raw_r1 or "").strip()
    r2_clean = str(raw_r2 or "").strip()
    if not r1_clean:
        r1_clean = "Discussion Partner" if g1_norm == "female" else "Colleague"
    if not r2_clean or r2_clean == r1_clean:
        r2_clean = "Colleague" if r1_clean != "Colleague" else "Project Partner"

    return g1_norm, g2_norm, a1_resolved, a2_resolved, r1_clean, r2_clean


def synthesize_quiz_audio(
    quiz_obj: Any,
    default_s1: str = "",
    default_s2: str = "",
    default_g1: str = "female",
    default_g2: str = "male",
) -> Optional[str]:
    """Synthesizes dialogue audio using TTS service and returns a base64 MP3 data URI."""
    from ..schemas import ListeningQuiz
    from ..tts import tts_service

    is_dialogue = isinstance(quiz_obj, ListeningQuiz) or (isinstance(quiz_obj, dict) and "script" in quiz_obj)
    if not is_dialogue:
        return None

    try:
        tts_service._refresh()
    except (OSError, RuntimeError) as e:
        logger.warning(f"TTS service refresh failed: {e}")

    cfg_s1 = default_s1 or tts_service.get_speaker_name(tts_service.voice_a)
    cfg_s2 = default_s2 or tts_service.get_speaker_name(tts_service.voice_b)

    if isinstance(quiz_obj, dict):
        s1 = str(quiz_obj.get("speaker_1") or "").strip()
        s2 = str(quiz_obj.get("speaker_2") or "").strip()
        if not s1 or s1.lower() in ("speaker 1", "speaker1"):
            s1 = cfg_s1
        if not s2 or s2.lower() in ("speaker 2", "speaker2"):
            s2 = cfg_s2
        g1, g2, a1, a2, role1, role2 = resolve_speaker_metadata(
            s1,
            s2,
            quiz_obj.get("speaker_1_gender"),
            quiz_obj.get("speaker_2_gender"),
            quiz_obj.get("speaker_1_role"),
            quiz_obj.get("speaker_2_role"),
            default_g1=default_g1,
            default_g2=default_g2,
            tts_service=tts_service,
        )
        ordered_head: Dict[str, Any] = {
            "title": quiz_obj.get("title", ""),
            "topic": quiz_obj.get("topic", ""),
            "speaker_1": s1,
            "speaker_2": s2,
            "speaker_1_gender": g1,
            "speaker_2_gender": g2,
            "speaker_1_role": role1,
            "speaker_2_role": role2,
            "speaker_1_accent": a1,
            "speaker_2_accent": a2,
        }
        for k, v in list(quiz_obj.items()):
            if k not in ordered_head:
                ordered_head[k] = v
        quiz_obj.clear()
        quiz_obj.update(ordered_head)
        script_dicts = quiz_obj["script"]
    else:
        s1 = str(getattr(quiz_obj, "speaker_1", None) or "").strip()
        s2 = str(getattr(quiz_obj, "speaker_2", None) or "").strip()
        if not s1 or s1.lower() in ("speaker 1", "speaker1"):
            s1 = cfg_s1
        if not s2 or s2.lower() in ("speaker 2", "speaker2"):
            s2 = cfg_s2
        quiz_obj.speaker_1 = s1
        quiz_obj.speaker_2 = s2
        g1, g2, a1, a2, role1, role2 = resolve_speaker_metadata(
            s1,
            s2,
            getattr(quiz_obj, "speaker_1_gender", None),
            getattr(quiz_obj, "speaker_2_gender", None),
            getattr(quiz_obj, "speaker_1_role", None),
            getattr(quiz_obj, "speaker_2_role", None),
            default_g1=default_g1,
            default_g2=default_g2,
            tts_service=tts_service,
        )
        quiz_obj.speaker_1_gender = g1
        quiz_obj.speaker_2_gender = g2
        quiz_obj.speaker_1_role = role1
        quiz_obj.speaker_2_role = role2
        quiz_obj.speaker_1_accent = a1
        quiz_obj.speaker_2_accent = a2
        script_dicts = [dataclasses.asdict(t) for t in quiz_obj.script]

    try:
        audio_binary = tts_service.process_script(
            script_dicts,
            return_binary=True,
            speaker_1=s1,
            speaker_2=s2,
            speaker_1_gender=g1,
            speaker_2_gender=g2,
        )
        if audio_binary:
            b64_str = base64.b64encode(audio_binary).decode("utf-8")
            return f"data:audio/mp3;base64,{b64_str}"
    except (OSError, RuntimeError, ValueError) as err:
        logger.warning(f"Audio generation failed for listening quiz: {err}")

    return None
