"""Tests for Pure Open Cloze Level-1 gate routing and Level-2 bypass in QuizGenerator."""

from pathlib import Path
from unittest.mock import PropertyMock, patch

from librarian.processor import WikiProcessor


def test_open_cloze_runs_l1_gate_and_bypasses_l2_vocab_auditor(tmp_path):
    """Pure Open Cloze (mode='open_cloze', options=[]) must run L1 audit_cloze_integrity
    and must NOT invoke ExpertAuditor.audit_quiz (which expects 4-option MCQ)."""
    unit_dir = tmp_path / "wiki" / "Sample_Unit"
    (unit_dir / "sources").mkdir(parents=True)
    (unit_dir / "extractions").mkdir(parents=True)
    (unit_dir / "handouts").mkdir(parents=True)

    passage = (
        "Although modern cities rely heavily on digital networks, many residents still struggle with "
        "everyday communication. Engineers have been working on new protocols for years, yet reliable "
        "access remains scarce in rural areas. Because infrastructure costs keep rising, local governments "
        "often turn to community partnerships to bridge the gap. In terms of long-term resilience, "
        "investment in local training makes a decisive difference. Ultimately, technology succeeds only "
        "when people themselves feel confident enough to use it."
    )
    (unit_dir / "sources" / "Sample_Unit.md").write_text(f"---\ntitle: Sample Unit\n---\n\n{passage}\n", encoding="utf-8")
    (unit_dir / "extractions" / "Sample_Unit_vocabulary.md").write_text("# Vocabulary\n", encoding="utf-8")
    (unit_dir / "extractions" / "Sample_Unit_grammar.md").write_text("# Grammar\n", encoding="utf-8")

    proc = WikiProcessor()
    orig_get = proc.config.get

    with patch.object(type(proc.config), "wiki_content_path", new_callable=PropertyMock, return_value=tmp_path / "wiki"), \
         patch.object(proc.config, "get", side_effect=lambda k, d=None: True if k == "enable_expert_audit" else orig_get(k, d)), \
         patch("librarian.expert_auditor.ExpertAuditor.audit_quiz") as mock_l2, \
         patch.object(proc, "audit_cloze_integrity", wraps=proc.audit_cloze_integrity) as mock_l1:
        out_path = proc.generate_quiz("Sample_Unit", count=5, template_name="cloze")

    assert Path(out_path).exists()
    assert mock_l1.call_count == 1
    assert mock_l2.call_count == 0

    import json, re
    html_text = Path(out_path).read_text(encoding="utf-8")
    m = re.search(r"const quizData = (.*?);\n", html_text)
    assert m is not None
    payload = json.loads(m.group(1))
    assert payload["mode"] == "open_cloze"
    assert len(payload["blanks"]) > 0
    for b in payload["blanks"]:
        marker = f"[{b['blank_index']}] ____"
        assert marker in payload["masked_passage"]
        assert any(marker in p for p in payload["paragraphs"])
