"""Tests for Light-Verb Collocation Mutual Exclusion Web.

Verifies that for delexical/light-verb collocations ('cast doubt on', 'strike a balance',
'pay tribute to', 'pose a threat to', 'wreak havoc on', etc.):
1. High-discrimination peer verb distractors are generated from peer clusters.
2. Dictionary idiom validation (LDOCE) acts as the sole arbiter:
   - Verbs that form attested alternative collocations with the noun (e.g. 'throw doubt on',
     'keep a balance', 'present a threat', 'cause havoc') are strictly banned as double keys.
   - Verbs that are unattested with the noun (e.g. *put doubt on, *hit a balance, *bring havoc)
     are validated as single-fit safe foils.
3. Morphological inflection accurately mirrors the target verb's surface form (e.g. 'paid' -> 'gave',
   'poses' -> 'causes', 'wreaked' -> 'brought').
"""

import pytest
from librarian.linguistics import LinguisticEngine


class TestLightVerbCollocationArena:
    """Test suite for Light-Verb Collocation Mutual Exclusion Web."""

    def test_detect_light_verb_collocation_context(self):
        """Context detection accurately identifies light-verb collocations and rejects literal contexts."""
        # 1. Standard active light-verb collocations
        ctx1 = LinguisticEngine.detect_light_verb_collocation_context(
            quote="The discovery of new evidence will cast doubt on the official version.",
            target="cast",
        )
        assert ctx1 is not None
        assert ctx1["verb"] == "cast"
        assert ctx1["noun"] == "doubt"
        assert ctx1["prep"] == "on"

        ctx2 = LinguisticEngine.detect_light_verb_collocation_context(
            quote="We must strike a balance between economic growth and conservation.",
            target="strike",
        )
        assert ctx2 is not None
        assert ctx2["verb"] == "strike"
        assert ctx2["noun"] == "balance"

        ctx3 = LinguisticEngine.detect_light_verb_collocation_context(
            quote="The prime minister paid tribute to the victims of the disaster.",
            target="paid",
        )
        assert ctx3 is not None
        assert ctx3["verb"] == "pay"
        assert ctx3["noun"] == "tribute"
        assert ctx3["prep"] == "to"

        ctx4 = LinguisticEngine.detect_light_verb_collocation_context(
            quote="Deforestation poses a serious threat to biodiversity.",
            target="poses",
        )
        assert ctx4 is not None
        assert ctx4["verb"] == "pose"
        assert ctx4["noun"] == "threat"
        assert ctx4["prep"] == "to"

        ctx5 = LinguisticEngine.detect_light_verb_collocation_context(
            quote="The storm wreaked havoc across the region.",
            target="wreaked",
        )
        assert ctx5 is not None
        assert ctx5["verb"] == "wreak"
        assert ctx5["noun"] == "havoc"

        # 2. Literal non-collocational contexts must return None
        literal_ctx = LinguisticEngine.detect_light_verb_collocation_context(
            quote="The fisherman cast a net into the river.",
            target="cast",
        )
        assert literal_ctx is None

    def test_is_attested_light_verb_collocation(self):
        """Attested collocations are identified from LDOCE gold records; unattested verbs return False."""
        # 'cast/throw doubt on'
        assert LinguisticEngine.is_attested_light_verb_collocation("cast", "doubt", "on")[0] is True
        assert LinguisticEngine.is_attested_light_verb_collocation("throw", "doubt", "on")[0] is True
        assert LinguisticEngine.is_attested_light_verb_collocation("put", "doubt", "on")[0] is False
        assert LinguisticEngine.is_attested_light_verb_collocation("shed", "doubt", "on")[0] is False

        # 'strike a balance'
        assert LinguisticEngine.is_attested_light_verb_collocation("strike", "balance")[0] is True
        assert LinguisticEngine.is_attested_light_verb_collocation("keep", "balance")[0] is True
        assert LinguisticEngine.is_attested_light_verb_collocation("hit", "balance")[0] is False
        assert LinguisticEngine.is_attested_light_verb_collocation("beat", "balance")[0] is False

        # 'pose/present a threat'
        assert LinguisticEngine.is_attested_light_verb_collocation("pose", "threat", "to")[0] is True
        assert LinguisticEngine.is_attested_light_verb_collocation("present", "threat", "to")[0] is True
        assert LinguisticEngine.is_attested_light_verb_collocation("cause", "threat", "to")[0] is False

        # 'wreak havoc on'
        assert LinguisticEngine.is_attested_light_verb_collocation("wreak", "havoc", "on")[0] is True
        assert LinguisticEngine.is_attested_light_verb_collocation("cause", "havoc", "on")[0] is True
        assert LinguisticEngine.is_attested_light_verb_collocation("bring", "havoc", "on")[0] is False

    def test_double_key_guard_banning_attested_collocations(self):
        """double_key_collision bans attested alternative collocations while allowing unattested foils."""
        q_doubt = "The discovery will cast doubt on the official version."
        # 'throw' is an attested alternative in LDOCE -> double key!
        is_dk, reason = LinguisticEngine.double_key_collision("cast", "throw", quote=q_doubt)
        assert is_dk is True
        assert reason == "double_key_collocation"

        # 'put' and 'shed' are unattested with 'doubt on' -> single-fit safe foils!
        is_dk_put, reason_put = LinguisticEngine.double_key_collision("cast", "put", quote=q_doubt)
        assert is_dk_put is False
        assert reason_put == "none"

        is_dk_shed, reason_shed = LinguisticEngine.double_key_collision("cast", "shed", quote=q_doubt)
        assert is_dk_shed is False
        assert reason_shed == "none"

        q_balance = "We must strike a balance between cost and speed."
        # 'keep' is an attested collocation with 'balance' -> double key!
        is_dk_keep, _ = LinguisticEngine.double_key_collision("strike", "keep", quote=q_balance)
        assert is_dk_keep is True

        # 'hit' is unattested with 'balance' -> single-fit safe foil!
        is_dk_hit, _ = LinguisticEngine.double_key_collision("strike", "hit", quote=q_balance)
        assert is_dk_hit is False

        q_threat = "Invasive species pose a threat to local ecosystems."
        # 'present' is attested in LDOCE ('pose/present a threat') -> double key!
        is_dk_pres, _ = LinguisticEngine.double_key_collision("pose", "present", quote=q_threat)
        assert is_dk_pres is True

        # 'cause' is unattested with 'a threat to' -> safe foil!
        is_dk_cause, _ = LinguisticEngine.double_key_collision("pose", "cause", quote=q_threat)
        assert is_dk_cause is False

    def test_construction_contrast_distractor_generation(self):
        """get_construction_contrast_distractors returns peer foils and bans attested double keys."""
        # 1. 'cast doubt on'
        cands_cast, spec_cast = LinguisticEngine.get_construction_contrast_distractors(
            "cast",
            quote="The new study will cast doubt on the hypothesis.",
            context_anchor="doubt",
            anchor_type="lv_triad",
        )
        assert "throw" in spec_cast["ban_groups"]
        assert "throw" not in cands_cast
        assert any(f in cands_cast for f in ["shed", "put", "drop"])

        # 2. 'poses a threat to' (inflected 3rd-person singular)
        cands_pose, spec_pose = LinguisticEngine.get_construction_contrast_distractors(
            "poses",
            quote="Pollution poses a threat to coral reefs.",
            context_anchor="threat",
            anchor_type="lv_triad",
        )
        assert any(b in spec_pose["ban_groups"] for b in ["present", "presents"])
        assert "present" not in cands_pose and "presents" not in cands_pose
        assert any(c in cands_pose for c in ["causes", "creates", "raises"])

        # 3. 'paid tribute to' (inflected past tense)
        cands_pay, spec_pay = LinguisticEngine.get_construction_contrast_distractors(
            "paid",
            quote="The citizens paid tribute to the founder of the hospital.",
            context_anchor="tribute",
            anchor_type="lv_triad",
        )
        assert all(c.endswith("d") or c.endswith("t") or c in ("gave", "made", "brought") for c in cands_pay)

    def test_end_to_end_generate_vocab_distractors(self):
        """Full generate_vocab_distractors pipeline produces clean, high-discrimination, SLA-sound foils."""
        cases = [
            ("cast", "v", "The study will cast doubt on the conclusion.", "doubt", "lv_triad", ["throw"]),
            ("strike", "v", "We must strike a balance between options.", "balance", "lv_triad", ["keep"]),
            ("poses", "v", "This virus poses a threat to public health.", "threat", "lv_triad", ["presents", "present"]),
            ("wreaked", "v", "The storm wreaked havoc on coastal towns.", "havoc", "lv_triad", ["caused", "cause"]),
        ]
        for word, pos, quote, anchor, a_type, banned_words in cases:
            distractors = LinguisticEngine.generate_vocab_distractors(
                target_word=word,
                pos=pos,
                quote=quote,
                context_anchor=anchor,
                anchor_type=a_type,
                target_count=3,
            )
            assert len(distractors) == 3, f"Expected 3 distractors for {word}, got {distractors}"
            assert word not in distractors
            for banned in banned_words:
                assert banned not in distractors, f"Banned double key '{banned}' appeared in {distractors} for {word}"
