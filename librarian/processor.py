import random
import os
import json
import re
import dataclasses
import base64
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
from .config import config
from .llm import llm
from .prompts import Prompts
from .schemas import (
    VocabularyExtraction, GrammarExtraction, SummaryExtraction,
    VocabularyQuiz, ReadingQuiz, TranslationQuiz, ListeningQuiz,
    RoutingResult, MindMapExtraction
)

class WikiProcessor:
    def __init__(self):
        self.config = config

    def _shuffle_quiz_options(self, quiz_obj):
        """Randomizes the order of options for each question and updates the correct index."""
        if not hasattr(quiz_obj, "questions") and not isinstance(quiz_obj, dict):
            return quiz_obj
        
        # Access questions (handle both dict and dataclass)
        questions = quiz_obj["questions"] if isinstance(quiz_obj, dict) else quiz_obj.questions
        
        for q in questions:
            # Handle both dict and dataclass
            q_dict = q if isinstance(q, dict) else dataclasses.asdict(q)
            
            # Sanitize string fields (options, target_word, word, correct_english_answer)
            quote_strip_pattern = r'^[«»"\'\u201c\u201d\u2018\u2019\s]+|[«»"\'\u201c\u201d\u2018\u2019\s]+$'
            raw_options = q_dict["options"]
            options = [re.sub(quote_strip_pattern, '', str(opt or '')) for opt in raw_options]
            
            for str_field in ["target_word", "word", "correct_english_answer"]:
                if str_field in q_dict and q_dict[str_field]:
                    clean_val = re.sub(quote_strip_pattern, '', str(q_dict[str_field]))
                    if isinstance(q, dict):
                        q[str_field] = clean_val
                    else:
                        setattr(q, str_field, clean_val)

            # Resolve the ground-truth answer text when the schema exposes one, so we can
            # guarantee the declared correct index actually points at the right option.
            expected_answer = None
            for truth_field in ("correct_english_answer", "target_word", "word"):
                if truth_field in q_dict and q_dict[truth_field]:
                    expected_answer = q_dict[truth_field]
                    break

            declared_idx = q_dict.get("correct_answer_index")
            try:
                declared_idx = int(declared_idx)
            except (TypeError, ValueError):
                declared_idx = 0
            if options:
                declared_idx = max(0, min(declared_idx, len(options) - 1))

            # Enforce answer synchronization. LLMs occasionally desync the index from the
            # option it should point at; repair it here so a graded quiz is never wrong.
            if expected_answer is not None and options:
                if expected_answer in options:
                    correct_idx = options.index(expected_answer)
                else:
                    # Correct answer is missing from the bank entirely; slot it in at the
                    # declared position so the item stays 4-options and gradeable.
                    options[declared_idx] = expected_answer
                    correct_idx = declared_idx
            else:
                correct_idx = declared_idx
            
            # Shuffle
            combined = list(zip(options, range(len(options))))
            random.shuffle(combined)
            
            new_options = [opt for opt, old_idx in combined]
            new_correct_idx = next(i for i, (opt, old_idx) in enumerate(combined) if old_idx == correct_idx)
            
            # Update
            if isinstance(q, dict):
                q["options"] = new_options
                q["correct_answer_index"] = new_correct_idx
            else:
                q.options = new_options
                q.correct_answer_index = new_correct_idx
                
        return quiz_obj

    def run_pipeline(self, source_filename: str, categories: List[str] = None) -> Tuple[str, List[str]]:
        """
        Modular pipeline for processing a raw unit file.
        Runs Vocabulary, Grammar, and Concept extractions in parallel for efficiency.
        """
        from .config import normalize_name
        import shutil

        input_path = Path(source_filename)
        wiki_dir = self.config.wiki_content_path

        if input_path.exists() and input_path.is_file():
            # Explicit file path on disk (either absolute or relative)
            input_stem = input_path.stem
            normalized_input = normalize_name(input_stem)

            # Determine the canonical unit folder name
            unit_folder_name = None
            try:
                resolved_input = input_path.resolve()
                resolved_wiki = wiki_dir.resolve()
                if resolved_input.is_relative_to(resolved_wiki):
                    rel_parts = resolved_input.relative_to(resolved_wiki).parts
                    if len(rel_parts) >= 3 and rel_parts[1] == "sources":
                        unit_folder_name = rel_parts[0]
            except Exception:
                pass

            if not unit_folder_name:
                unit_folder_name = input_stem.replace(" ", "_")
                if wiki_dir.exists():
                    for child in wiki_dir.iterdir():
                        if child.is_dir() and normalize_name(child.name) == normalized_input:
                            unit_folder_name = child.name
                            break

            # Define paths
            unit_dir = wiki_dir / unit_folder_name
            sources_dir = unit_dir / "sources"
            extractions_dir = unit_dir / "extractions"
            handouts_dir = unit_dir / "handouts"

            sources_dir.mkdir(parents=True, exist_ok=True)
            extractions_dir.mkdir(parents=True, exist_ok=True)
            handouts_dir.mkdir(parents=True, exist_ok=True)

            # Compile directly from the original explicit file path without copying to sources_dir
            source_path = input_path
        else:
            # Treated as a unit name or filename under existing units
            input_stem = Path(source_filename).stem
            normalized_input = normalize_name(source_filename)

            unit_folder_name = None
            if wiki_dir.exists():
                for child in wiki_dir.iterdir():
                    if child.is_dir() and (normalize_name(child.name) == normalized_input or normalize_name(child.name) == normalize_name(input_stem)):
                        unit_folder_name = child.name
                        break

            if not unit_folder_name:
                return f"Error: Source file or unit folder '{source_filename}' not found.", []

            unit_dir = wiki_dir / unit_folder_name
            sources_dir = unit_dir / "sources"
            extractions_dir = unit_dir / "extractions"
            handouts_dir = unit_dir / "handouts"

            sources_dir.mkdir(parents=True, exist_ok=True)
            extractions_dir.mkdir(parents=True, exist_ok=True)
            handouts_dir.mkdir(parents=True, exist_ok=True)

            # Locate the source file in sources_dir
            text_sources = [f for f in sources_dir.iterdir() if f.is_file() and f.suffix in [".md", ".txt"]]
            if len(text_sources) == 1:
                source_path = text_sources[0]
            elif len(text_sources) > 1:
                matching = [f for f in text_sources if normalize_name(f.stem) == normalized_input]
                source_path = matching[0] if matching else text_sources[0]
            else:
                return f"Error: No source markdown or text file found under unit sources for {unit_folder_name}.", []

        with open(source_path, "r", encoding="utf-8") as f:
            content = f.read()

        self._last_source_content = content
        file_stem = unit_folder_name
        all_saved = []
        
        # Determine extraction counts from compile_defaults config
        compile_defaults = self.config.get("compile_defaults") or {}
        v_count = compile_defaults.get("vocabulary", 15)
        e_count = compile_defaults.get("expressions", 5)
        g_count = compile_defaults.get("grammar", 5)
        c_count = compile_defaults.get("concepts", 5)
        max_p = compile_defaults.get("max_parallel", 4)

        # 1. Prepare Extraction Tasks
        v_prompt_template, v_schema = Prompts.get("extract_vocabulary")
        e_prompt_template, e_schema = Prompts.get("extract_expressions")
        g_prompt_template, g_schema = Prompts.get("extract_grammar")
        s_prompt_template, s_schema = Prompts.get("extract_summary")
        m_prompt_template, m_schema = Prompts.get("extract_mindmap")

        v_kwargs = {"content": content, "count": v_count}
        e_kwargs = {"content": content, "count": e_count}
        g_kwargs = {"content": content, "count": g_count}
        s_kwargs = {"content": content, "count": c_count}
        m_kwargs = {"content": content}

        tasks = [
            ("vocabulary", v_prompt_template.format(**v_kwargs), self._interpolate_schema(v_schema, v_kwargs)),
            ("expressions", e_prompt_template.format(**e_kwargs), self._interpolate_schema(e_schema, e_kwargs)),
            ("grammar", g_prompt_template.format(**g_kwargs), self._interpolate_schema(g_schema, g_kwargs)),
            ("summary", s_prompt_template.format(**s_kwargs), self._interpolate_schema(s_schema, s_kwargs)),
            ("mindmap", m_prompt_template.format(**m_kwargs), self._interpolate_schema(m_schema, m_kwargs))
        ]

        if categories:
            normalized_cats = [c.lower() for c in categories]
            task_names_to_run = []
            for cat in normalized_cats:
                if cat == "vocabulary":
                    task_names_to_run.extend(["vocabulary", "expressions"])
                else:
                    task_names_to_run.append(cat)
            tasks = [t for t in tasks if t[0] in task_names_to_run]

        # 2. Run extractions in parallel
        results = []
        try:
            with ThreadPoolExecutor(max_workers=max_p) as executor:
                futures = {
                    executor.submit(llm.chat, [{"role": "user", "content": prompt}], schema=schema, task_name=f"extract_{name}_{file_stem}"): name 
                    for name, prompt, schema in tasks
                }
                for future in futures:
                    name = futures[future]
                    data = future.result()
                    if data:
                        results.append((name, data))

            # 3. Process, Merge, and Save Results
            vocab_data = None
            expressions_data = None
            other_results = []

            for category, data in results:
                if category == "vocabulary":
                    vocab_data = data
                elif category == "expressions":
                    expressions_data = data
                else:
                    other_results.append((category, data))

            # Merge expressions into vocabulary if both exist
            if vocab_data:
                merged_vocab = []
                if dataclasses.is_dataclass(vocab_data):
                    merged_vocab.extend(getattr(vocab_data, "vocabulary", []))
                elif isinstance(vocab_data, dict):
                    merged_vocab.extend(vocab_data.get("vocabulary", []))

                if expressions_data:
                    def _resolve_slotted_word(expr_obj):
                        w = expr_obj.get("word", "") if isinstance(expr_obj, dict) else getattr(expr_obj, "word", "")
                        aud = expr_obj.get("design_audit", "") if isinstance(expr_obj, dict) else getattr(expr_obj, "design_audit", "")
                        if ("[" not in w and "one's" not in w) and ("[" in aud or "one's" in aud):
                            parts = [p.strip() for p in aud.replace("->", "➔").split("➔")]
                            for p in parts:
                                if ("[" in p or "one's" in p):
                                    cand = p[5:].strip().lstrip(':').strip() if p.upper().startswith("DRAFT") else p
                                    candidate = cand.split(" -")[0].split(" (")[0].strip()
                                    cand_tokens = re.findall(r'[a-zA-Z]+', candidate.replace("[", "").replace("]", ""))
                                    w_tokens = re.findall(r'[a-zA-Z]+', w)
                                    if cand_tokens and w_tokens and cand_tokens[0].lower() == w_tokens[0].lower():
                                        return candidate
                        return w

                    expr_list = getattr(expressions_data, "expressions", []) if dataclasses.is_dataclass(expressions_data) else expressions_data.get("expressions", [])
                    for expr in expr_list:
                        resolved_word = _resolve_slotted_word(expr)
                        if isinstance(expr, dict):
                            mapped_item = {
                                "design_audit": expr.get("design_audit", ""),
                                "word": resolved_word,
                                "part_of_speech": expr.get("part_of_speech", "phrasal verb"),
                                "definition": expr.get("definition", ""),
                                "word_cefr_level": expr.get("word_cefr_level", "B2"),
                                "quoted_sentence": expr.get("quoted_sentence", ""),
                                "example_usage": expr.get("example_usage", ""),
                            }
                        else:
                            from .schemas import VocabularyItem
                            mapped_item = VocabularyItem(
                                design_audit=getattr(expr, "design_audit", ""),
                                word=resolved_word,
                                part_of_speech=getattr(expr, "part_of_speech", "phrasal verb"),
                                definition=getattr(expr, "definition", ""),
                                word_cefr_level=getattr(expr, "word_cefr_level", "B2"),
                                quoted_sentence=getattr(expr, "quoted_sentence", ""),
                                example_usage=getattr(expr, "example_usage", ""),
                            )
                        merged_vocab.append(mapped_item)

                if dataclasses.is_dataclass(vocab_data):
                    vocab_data.vocabulary = merged_vocab
                elif isinstance(vocab_data, dict):
                    vocab_data["vocabulary"] = merged_vocab

                all_saved.extend(self._save_extraction_results(vocab_data, source_path.name, category_override="vocabulary"))
            elif expressions_data:
                from .schemas import VocabularyExtraction, VocabularyItem
                vocab_list = []
                expr_list = getattr(expressions_data, "expressions", []) if dataclasses.is_dataclass(expressions_data) else expressions_data.get("expressions", [])
                for expr in expr_list:
                    resolved_word = _resolve_slotted_word(expr) if '_resolve_slotted_word' in locals() else getattr(expr, "word", "")
                    vocab_list.append(VocabularyItem(
                        design_audit=getattr(expr, "design_audit", ""),
                        word=resolved_word,
                        part_of_speech=getattr(expr, "part_of_speech", "phrasal verb"),
                        definition=getattr(expr, "definition", ""),
                        word_cefr_level=getattr(expr, "word_cefr_level", "B2"),
                        quoted_sentence=getattr(expr, "quoted_sentence", ""),
                        example_usage=getattr(expr, "example_usage", ""),
                    ))
                v_extracted = VocabularyExtraction(
                    title=f"{file_stem.replace('_', ' ')} Vocabulary",
                    overall_cefr_level="B2",
                    vocabulary=vocab_list
                )
                all_saved.extend(self._save_extraction_results(v_extracted, source_path.name, category_override="vocabulary"))

            # Save other results (grammar, summary)
            for category, data in other_results:
                all_saved.extend(self._save_extraction_results(data, source_path.name, category_override=category))

            return f"Pipeline completed for {source_filename}. Generated {len(all_saved)} files.", all_saved

        except Exception as e:
            if is_new_unit and unit_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(unit_dir)
                except Exception:
                    pass
            return f"Pipeline Error: {e}", all_saved

    def generate_quiz(self, unit_name, count=10, template_name="vocabulary"):
        """
        Generates a quiz handout based on extracted wiki data.
        """
        # 1. Normalize unit_name to extract core unit folder name robustly
        parts = Path(unit_name).parts
        core_name = None
        if "wiki" in parts:
            idx = parts.index("wiki")
            if idx + 1 < len(parts):
                core_name = parts[idx + 1]
        elif "raw" in parts:
            idx = parts.index("raw")
            if idx + 1 < len(parts):
                core_name = parts[idx + 1]
        else:
            core_name = parts[0] if parts else unit_name
            
        if core_name.endswith(".md"): core_name = core_name[:-3]
        elif core_name.endswith(".txt"): core_name = core_name[:-4]
            
        for suffix in ["_vocabulary", "_grammar", "_concepts"]:
            if core_name.endswith(suffix):
                core_name = core_name[:-len(suffix)]

        # 2. Resolve Data
        data = self._load_wiki_data(core_name, template_name)
        if not data:
            return f"Error: Could not find extracted {template_name} data for {core_name}. If this is a fresh setup, please verify that Ollama or your LLM API is running, and check if the extraction pipeline ran successfully."

        # 3. Build Prompt
        prompt_template, schema_cls = Prompts.get(f"{template_name}_quiz")
        
        kwargs = {"count": count}
        if template_name == "vocabulary":
            kwargs["vocabulary_content"] = data["content"]
            kwargs["cefr_level"] = data.get("cefr_level", "B2")
        elif template_name == "reading":
            kwargs["passage_content"] = data["passage"]
            kwargs["cefr_level"] = data.get("cefr_level", "B2")
        elif template_name == "translation":
            kwargs["vocabulary_content"] = data["vocab_list"]
            kwargs["grammar_content"] = data["grammar_list"]
            kwargs["target_language"] = self.config.get("target_language") or "Chinese"
            kwargs["cefr_level"] = data.get("cefr_level", "B2")
        elif template_name == "listening":
            kwargs["vocabulary_content"] = data["vocab_list"]
            kwargs["cefr_level"] = data.get("cefr_level", "B2")
            
            # Retrieve tts configurations to get genders & accents for Schema-First Live Injection
            from .tts import tts_service
            tts_service._refresh()
            
            def get_voice_gender(voice_name):
                v = str(voice_name or "").lower()
                if any(v.startswith(p) for p in ["af_", "bf_"]):
                    return "female"
                if any(v.startswith(p) for p in ["am_", "bm_"]):
                    return "male"
                if any(x in v for x in ["aria", "jenny", "sonia", "libby"]):
                    return "female"
                if any(x in v for x in ["guy", "christopher", "ryan", "thomas"]):
                    return "male"
                return "female"
                
            def get_voice_accent(voice_name):
                v_lower = str(voice_name or "").lower()
                if "gb" in v_lower or "bf_" in v_lower or "bm_" in v_lower or "sonia" in v_lower:
                    return "Accent: British"
                if "us" in v_lower or "af_" in v_lower or "am_" in v_lower or "aria" in v_lower or "michael" in v_lower or "guy" in v_lower:
                    return "Accent: American"
                if "cn" in v_lower or "zh" in v_lower or "xiaoxiao" in v_lower or "yunxi" in v_lower:
                    return "Accent: Chinese (Mandarin)"
                return "Accent: Standard"

            g1 = get_voice_gender(tts_service.voice_a)
            g2 = get_voice_gender(tts_service.voice_b)
            a1 = get_voice_accent(tts_service.voice_a)
            a2 = get_voice_accent(tts_service.voice_b)

            kwargs["speaker_1_gender"] = g1
            kwargs["speaker_2_gender"] = g2
            kwargs["speaker_1_accent"] = a1
            kwargs["speaker_2_accent"] = a2

        elif template_name == "video":
            kwargs["transcript_content"] = data["transcript"]
            kwargs["video_url"] = data["video_url"]
            kwargs["video_type"] = data["video_type"]
            kwargs["cefr_level"] = data.get("cefr_level", "B2")

        prompt = prompt_template.format(**kwargs)

        # 4. Call LLM
        try:
            quiz_obj = llm.chat(
                [{"role": "user", "content": prompt}],
                schema=self._interpolate_schema(schema_cls, kwargs),
                task_name=f"quiz_{template_name}_{core_name}"
            )

            if not quiz_obj:
                return "Error: LLM returned empty quiz data."
            
            # 4.5. Randomize options
            quiz_obj = self._shuffle_quiz_options(quiz_obj)

            if template_name == "reading":
                if isinstance(quiz_obj, dict):
                    quiz_obj["passage"] = data["passage"]
                else:
                    quiz_obj.passage = data["passage"]

            if template_name == "video":
                if isinstance(quiz_obj, dict):
                    quiz_obj["video_url"] = data["video_url"]
                    quiz_obj["video_type"] = data["video_type"]
                    quiz_obj["transcript"] = data["transcript"]
                else:
                    quiz_obj.video_url = data["video_url"]
                    quiz_obj.video_type = data["video_type"]
                    quiz_obj.transcript = data["transcript"]

            # 5. TTS for Listening Quiz (Base64 Embedding)
            audio_url = None
            if template_name == "listening":
                # Detect dialogue data (dataclass or dict)
                is_dialogue = isinstance(quiz_obj, ListeningQuiz) or (isinstance(quiz_obj, dict) and "script" in quiz_obj)

                if is_dialogue:
                    from .tts import tts_service
                    tts_service._refresh()

                    # Use pre-calculated genders, roles, and accents directly from kwargs or quiz_obj
                    if isinstance(quiz_obj, dict):
                        g1 = quiz_obj.get("speaker_1_gender") or kwargs["speaker_1_gender"]
                        g2 = quiz_obj.get("speaker_2_gender") or kwargs["speaker_2_gender"]
                        a1 = quiz_obj.get("speaker_1_accent") or kwargs["speaker_1_accent"]
                        a2 = quiz_obj.get("speaker_2_accent") or kwargs["speaker_2_accent"]
                        
                        role1 = "Creative Director" if g1 == "female" else "Technical Lead"
                        role2 = "Technical Lead" if g2 == "male" else "Creative Director"
                        if role1 == role2:
                            role2 = "Product Manager" if g2 == "female" else "Lead Analyst"
                            
                        quiz_obj["speaker_1_gender"] = g1
                        quiz_obj["speaker_2_gender"] = g2
                        quiz_obj["speaker_1_role"] = role1
                        quiz_obj["speaker_2_role"] = role2
                        quiz_obj["speaker_1_accent"] = a1
                        quiz_obj["speaker_2_accent"] = a2
                        script_dicts = quiz_obj["script"]
                        s1 = quiz_obj.get("speaker_1")
                        s2 = quiz_obj.get("speaker_2")
                    else:
                        g1 = quiz_obj.speaker_1_gender or kwargs["speaker_1_gender"]
                        g2 = quiz_obj.speaker_2_gender or kwargs["speaker_2_gender"]
                        a1 = quiz_obj.speaker_1_accent or kwargs["speaker_1_accent"]
                        a2 = quiz_obj.speaker_2_accent or kwargs["speaker_2_accent"]
                        
                        role1 = "Creative Director" if g1 == "female" else "Technical Lead"
                        role2 = "Technical Lead" if g2 == "male" else "Creative Director"
                        if role1 == role2:
                            role2 = "Product Manager" if g2 == "female" else "Lead Analyst"
                            
                        quiz_obj.speaker_1_gender = g1
                        quiz_obj.speaker_2_gender = g2
                        quiz_obj.speaker_1_role = role1
                        quiz_obj.speaker_2_role = role2
                        quiz_obj.speaker_1_accent = a1
                        quiz_obj.speaker_2_accent = a2
                        script_dicts = [dataclasses.asdict(t) for t in quiz_obj.script]
                        s1 = quiz_obj.speaker_1
                        s2 = quiz_obj.speaker_2

                    audio_binary = tts_service.process_script(
                        script_dicts, 
                        return_binary=True,
                        speaker_1=s1,
                        speaker_2=s2,
                        speaker_1_gender=g1,
                        speaker_2_gender=g2
                    )
                    
                    if audio_binary:
                        b64_str = base64.b64encode(audio_binary).decode("utf-8")
                        audio_url = f"data:audio/mp3;base64,{b64_str}"

            # 6. Render and Save
            language = self.config.get("target_language") or "Chinese"
            html_content = self._render_handout(quiz_obj, template_name, audio_url, language=language)
            asset_stem = data.get("source_stem", core_name) if isinstance(data, dict) else core_name
            handout_filename = f"{asset_stem}_{template_name}_quiz.html"
            handout_dir = self.config.wiki_content_path / core_name / "handouts"
            handout_dir.mkdir(parents=True, exist_ok=True)
            handout_path = handout_dir / handout_filename
            
            with open(handout_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            return str(handout_path)

        except Exception as e:
            return f"Quiz Generation Error: {e}"

    def ask_wiki(self, query):
        """RAG-lite for querying wiki content."""
        inventory = self._get_wiki_inventory()
        inventory_str = "\n".join([f"- {i['name']} ({i['type']})" for i in inventory])
        
        routing_prompt = f"""You are a Librarian. Select up to 5 relevant files for: {query}\nINVENTORY:\n{inventory_str}"""

        try:
            route = llm.chat([{"role": "user", "content": routing_prompt}], schema=RoutingResult, task_name="wiki_routing")
            selected_names = route.selected_files if route else []
            
            # Helper to normalize names for comparison (ignores spaces, underscores, and extension)
            def normalize_name(n):
                if n.lower().endswith('.md'):
                    n = n[:-3]
                return re.sub(r'[^a-zA-Z0-9]', '', n).lower()

            context_parts = []
            for name in selected_names:
                norm_name = normalize_name(name)
                item = next((i for i in inventory if normalize_name(i['name']) == norm_name), None)
                if item:
                    with open(item['path'], "r", encoding="utf-8") as f:
                        context_parts.append(f"--- FILE: {item['name']} ---\n{f.read()}")

            # Robust Fallback: If LLM selected nothing, or all selected names failed to match any inventory file,
            # use a smart local keyword matcher
            if not context_parts:
                # Pre-process query to separate digits (e.g. "4unit" -> "4 unit", "book4" -> "book 4")
                clean_query = re.sub(r'([0-9]+)', r' \1 ', query).lower()
                tokens = []
                # Keep words length >= 2, or any length if they are unit digits
                for word in re.findall(r'[a-zA-Z0-9]+', clean_query):
                    if len(word) >= 2 or word.isdigit():
                        tokens.append(word)
                # Keep Chinese characters
                for char in re.findall(r'[\u4e00-\u9fff]', query):
                    tokens.append(char)

                scored_items = []
                for item in inventory:
                    name_lower = item['name'].lower()
                    score = 0
                    
                    # Category indicators boost matched types
                    if "vocabulary" in tokens and item['type'] == "wiki_vocabulary":
                        score += 5
                    if "grammar" in tokens and item['type'] == "wiki_grammar":
                        score += 5
                    if "summary" in tokens and item['type'] == "wiki_summaries":
                        score += 5
                    
                    for t in tokens:
                        if t in name_lower:
                            score += 10
                            if t.isdigit():
                                score += 10  # heavy priority match on unit/book numbers
                    
                    if score > 15:
                        scored_items.append((item['name'], score))
                
                if scored_items:
                    scored_items.sort(key=lambda x: x[1], reverse=True)
                    # Take top 3 matching items to avoid bloating prompt context
                    top_matches = [name for name, _ in scored_items[:3]]
                    for name in top_matches:
                        item = next((i for i in inventory if i['name'] == name), None)
                        if item:
                            with open(item['path'], "r", encoding="utf-8") as f:
                                context_parts.append(f"--- FILE: {item['name']} ---\n{f.read()}")

            # Smart fallback for source/original text queries not yet resolved
            if not context_parts:
                query_lower = query.lower()
                is_source_query = any(
                    kw in query_lower
                    for kw in ["original text", "full text", "source text", "complete text",
                                "原文", "全文", "完整内容", "课文", "文章"]
                )

                if is_source_query:
                    # Extract unit identifier from query (e.g. "Book 4 Unit 5" -> "book_4_unit_5")
                    source_tokens = []
                    for word in re.findall(r'[a-zA-Z0-9]+', clean_query):
                        if len(word) >= 2 or word.isdigit():
                            source_tokens.append(word.lower())

                    # Collect names already used as context to avoid duplicates
                    existing_names = {re.sub(r'[^a-z0-9]', '', n).lower() for n in inventory}

                    seen_units = set()
                    for item in inventory:
                        if item['type'] != 'raw_source':
                            continue
                        # Check path and name for unit identifier tokens
                        item_path = str(item['path']).replace('\\', '/')
                        item_name = (item['name'] + " " + item_path).lower()
                        matched_tokens = [t for t in source_tokens if t in item_name]
                        if not matched_tokens:
                            continue

                        # Derive unit folder name from path (wiki/<unit>/sources/...)
                        wiki_idx = item_path.find('/wiki/')
                        if wiki_idx >= 0:
                            after_wiki = item_path[wiki_idx + 6:]
                            parts = after_wiki.split('/')
                            if len(parts) >= 3 and parts[1] == 'sources':
                                unit_folder = parts[0]
                            else:
                                continue
                        else:
                            # Bare filename like "Book_4_Unit_5.md" -> use stem as unit name
                            unit_folder = Path(item['name']).stem

                        if unit_folder in seen_units:
                            continue
                        norm_key = re.sub(r'[^a-z0-9]', '', unit_folder).lower()
                        if norm_key in existing_names:
                            continue
                        seen_units.add(unit_folder)
                        existing_names.add(norm_key)

                        with open(item['path'], "r", encoding="utf-8") as f:
                            context_parts.append(
                                f"--- SOURCE FILE ({unit_folder}): {item['name']} ---\n{f.read()}"
                            )

            full_context = "\n\n".join(context_parts)
            answer_prompt = f"""Answer based on context:\n{full_context}\n\nQUERY: {query}"""
            return llm.chat([{"role": "user", "content": answer_prompt}], json_format=False)

        except Exception as e:
            return f"Wiki Query Error: {e}"

    def _save_extraction_results(self, data, source_filename, category_override=None):
        """Categorizes and saves extraction results."""
        path_obj = Path(source_filename)
        if path_obj.stem.lower() in ["subtitle", "transcript"] and path_obj.parent.name:
            filename_stem = path_obj.parent.name
        else:
            filename_stem = path_obj.stem
        saved_paths = []
        
        # Use explicit override if provided (most robust)
        category = category_override
        
        # Fallback to dynamic category detection for Virtual Schemas
        if not category:
            category = getattr(data, "_category", None)
        
        # Fallback to class-based detection for legacy/internal schemas
        if not category:
            if isinstance(data, VocabularyExtraction): category = "vocabulary"
            elif isinstance(data, GrammarExtraction): category = "grammar"
            elif isinstance(data, SummaryExtraction): category = "summary"
            elif isinstance(data, MindMapExtraction): category = "mindmap"

        unit_dir = self.config.wiki_content_path / filename_stem
        extractions_dir = unit_dir / "extractions"
        extractions_dir.mkdir(parents=True, exist_ok=True)

        if category == "vocabulary":
            # Deduplicate vocabulary list based on 'word' field (case-insensitive & stripped)
            seen = set()
            deduped_vocabulary = []
            if dataclasses.is_dataclass(data):
                vocab_list = getattr(data, "vocabulary", [])
                for item in vocab_list:
                    word_val = getattr(item, "word", "")
                    if isinstance(word_val, str):
                        w_clean = word_val.strip().lower().replace("[[", "").replace("]]", "")
                        if w_clean and w_clean not in seen:
                            seen.add(w_clean)
                            deduped_vocabulary.append(item)
                data.vocabulary = deduped_vocabulary
            elif isinstance(data, dict):
                vocab_list = data.get("vocabulary", [])
                for item in vocab_list:
                    word_val = item.get("word", "") if isinstance(item, dict) else getattr(item, "word", "")
                    if isinstance(word_val, str):
                        w_clean = word_val.strip().lower().replace("[[", "").replace("]]", "")
                        if w_clean and w_clean not in seen:
                            seen.add(w_clean)
                            deduped_vocabulary.append(item)
                data["vocabulary"] = deduped_vocabulary

            path = extractions_dir / f"{filename_stem}_vocabulary.md"
            content = self._format_as_markdown(data, "vocabulary", source_filename)
            with open(path, "w", encoding="utf-8") as f: f.write(content)
            saved_paths.append(str(path))
        
        elif category == "grammar":
            path = extractions_dir / f"{filename_stem}_grammar.md"
            content = self._format_as_markdown(data, "grammar", source_filename)
            with open(path, "w", encoding="utf-8") as f: f.write(content)
            saved_paths.append(str(path))

        elif category == "summary":
            path = extractions_dir / f"{filename_stem}_summary.md"
            content = self._format_as_markdown(data, "summary", source_filename)
            with open(path, "w", encoding="utf-8") as f: f.write(content)
            saved_paths.append(str(path))

        elif category == "mindmap":
            # 1. Save raw JSON file
            json_path = extractions_dir / f"{filename_stem}_mindmap.json"
            if dataclasses.is_dataclass(data):
                mindmap_dict = dataclasses.asdict(data)
            else:
                mindmap_dict = data
            nodes = mindmap_dict.get("nodes", []) if isinstance(mindmap_dict, dict) else []
            mindmap_dict["item_count"] = len(nodes)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(mindmap_dict, f, indent=2, ensure_ascii=False)
            saved_paths.append(str(json_path))

            # 2. Render and save standalone HTML to extractions directory
            html_content = self._render_mindmap(mindmap_dict)
            html_path = extractions_dir / f"{filename_stem}_mindmap.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            saved_paths.append(str(html_path))

        return saved_paths

    def _format_as_markdown(self, data, category, source_filename=None):
        """
        Converts an extraction object to a markdown string dynamically.
        Iterates over dataclass fields or dict keys to remain field-agnostic.
        """
        source_link = f"[[{source_filename}]]" if source_filename else "None"
        display_title = Path(source_filename).stem.replace("_", " ") if source_filename else "Unit"
        
        # 1. Handle Collection Extractions (Vocabulary, Grammar)
        if category in ["vocabulary", "grammar"]:
            # Identify top-level fields (like overall_cefr_level)
            if dataclasses.is_dataclass(data):
                fields = [(f.name, getattr(data, f.name)) for f in dataclasses.fields(data)]
            else:
                fields = list(data.items())

            # Identify the collection list & count
            items = []
            if dataclasses.is_dataclass(data):
                list_field = next((f for f in dataclasses.fields(data) if isinstance(getattr(data, f.name), list)), None)
                if list_field:
                    items = getattr(data, list_field.name)
            else:
                list_field_name = next((k for k, v in fields if isinstance(v, list)), None)
                if list_field_name: items = data.get(list_field_name, [])

            # Automatic Deduplication: filter out repeated entries with identical primary key (e.g. word / headword)
            seen_keys = set()
            unique_items = []
            for item in items:
                k_val = None
                if dataclasses.is_dataclass(item):
                    for attr in ["word", "name", "concept_name", "quote"]:
                        if hasattr(item, attr) and getattr(item, attr):
                            k_val = str(getattr(item, attr)).strip().lower()
                            break
                elif isinstance(item, dict):
                    for attr in ["word", "name", "concept_name", "quote"]:
                        if attr in item and item[attr]:
                            k_val = str(item[attr]).strip().lower()
                            break
                
                if k_val:
                    if k_val in seen_keys:
                        continue
                    seen_keys.add(k_val)
                unique_items.append(item)
            
            items = unique_items
            item_count = len(items)

            lines = [
                "---",
                f"title: \"{display_title}\"",
                f"source: \"{source_link}\"",
                f"category: [\"{category}\", \"extraction\"]",
                f"item_count: {item_count}"
            ]

            for name, val in fields:
                if name not in ["title", "grammar_patterns", "vocabulary", "concepts", "_category"]:
                    if val: lines.append(f"{name}: \"{val}\"")
            
            lines.extend(["---", "", f"# {category.title()}: {display_title}", ""])

            for item in items:
                if dataclasses.is_dataclass(item):
                    item_fields = [(f.name, getattr(item, f.name)) for f in dataclasses.fields(item) if f.name != "design_audit"]
                else:
                    item_fields = [(k, v) for k, v in item.items() if k != "design_audit"]

                if not item_fields:
                    continue

                # Locate canonical primary header field
                primary_key = None
                for hk in ["word", "category", "name", "concept_name", "title"]:
                    if any(k == hk for k, v in item_fields):
                        primary_key = hk
                        break

                if primary_key:
                    header_entry = next((k, v) for k, v in item_fields if k == primary_key)
                    body_entries = [entry for entry in item_fields if entry[0] != primary_key]
                else:
                    header_entry = item_fields[0]
                    body_entries = item_fields[1:]

                header_name, header_val = header_entry
                if not (str(header_val).startswith("[[") and str(header_val).endswith("]]")):
                    header_val = f"[[{header_val}]]"
                
                lines.append(f"## {header_val}")
                
                # Iterate remaining fields as bullet points
                for fname, fval in body_entries:
                    label = fname.replace("_", " ").title()
                    if fval:
                        lines.append(f"- **{label}**: {fval}")
                lines.append("")
            
            return "\n".join(lines)

        # 2. Handle Summary Extractions (with nested Concepts)
        elif category == "summary":
            if dataclasses.is_dataclass(data):
                title = getattr(data, "title", display_title)
                summary_text = getattr(data, "text_summary_or_plot", "")
                reading_time = getattr(data, "estimated_reading_time", "")
                questions = getattr(data, "essential_questions", [])
                lesson_hook = getattr(data, "lesson_hook", "")
                concepts = getattr(data, "concepts", [])
                overall_cefr = getattr(data, "overall_cefr_level", "B2")
            else:
                title = data.get("title", display_title)
                summary_text = data.get("text_summary_or_plot", "")
                reading_time = data.get("estimated_reading_time", "")
                questions = data.get("essential_questions", [])
                lesson_hook = data.get("lesson_hook", "")
                concepts = data.get("concepts", [])
                overall_cefr = data.get("overall_cefr_level", "B2")

            # Deterministic word count & reading time calculation (approx. 180-200 WPM)
            if hasattr(self, "_last_source_content") and self._last_source_content:
                words = len(re.findall(r'\b\w+\b', self._last_source_content))
                minutes = max(1, round(words / 180))
                reading_time = f"{words} words, approx. {minutes} min{'s' if minutes > 1 else ''} reading time"

            item_count = len(concepts)

            lines = [
                "---",
                f"title: \"{title}\"",
                f"source: \"{source_link}\"",
                "category: [\"summary\", \"extraction\"]",
                f"overall_cefr_level: \"{overall_cefr}\"",
                f"item_count: {item_count}",
                f"estimated_reading_time: \"{reading_time}\"",
                "---",
                "",
                f"# Summary: {title}",
                "",
                "## Narrative Overview",
                summary_text,
                "",
                "## Lesson Hook",
                lesson_hook,
                "",
                "## Essential Questions",
            ]
            for q in questions:
                lines.append(f"- {q}")
            lines.append("")
            
            lines.append("## Core Concepts")
            lines.append("")
            for concept in concepts:
                if dataclasses.is_dataclass(concept):
                    c_name = getattr(concept, "concept_name", "")
                    c_sig = getattr(concept, "educational_significance", "")
                    c_details = getattr(concept, "key_details", [])
                    c_conn = getattr(concept, "related_connections", [])
                else:
                    c_name = concept.get("concept_name", "")
                    c_sig = concept.get("educational_significance", "")
                    c_details = concept.get("key_details", [])
                    c_conn = concept.get("related_connections", [])

                # Add double-brackets around concept name for wikilinks
                c_header = c_name
                if not (c_header.startswith("[[") and c_header.endswith("]]")):
                    c_header = f"[[{c_header}]]"

                lines.append(f"### {c_header}")
                if c_sig:
                    lines.append(f"- **Educational Significance**: {c_sig}")
                if c_details:
                    lines.append("- **Key Details**:")
                    for detail in c_details:
                        lines.append(f"  - {detail}")
                if c_conn:
                    conn_links = []
                    for conn in c_conn:
                        clean_conn = str(conn or "").strip()
                        if "Connect to" in clean_conn or ":" in clean_conn:
                            m = re.search(r'(?:Connect to\s*)?\*?\*?([A-Za-z0-9\s/&#\-]+?)\*?\*?(?:\s*:|\s*$)', clean_conn, re.IGNORECASE)
                            if m and len(m.group(1).strip()) > 1:
                                clean_conn = m.group(1).strip()
                            elif ":" in clean_conn:
                                clean_conn = re.sub(r'^(?:Connect to\s*)?\*?\*?|\*?\*?$', '', clean_conn.split(":")[0]).strip()
                        if clean_conn:
                            if not (clean_conn.startswith("[[") and clean_conn.endswith("]]")):
                                conn_links.append(f"[[{clean_conn}]]")
                            else:
                                conn_links.append(clean_conn)
                    if conn_links:
                        lines.append(f"- **Related Connections**: {', '.join(conn_links)}")
                lines.append("")
                lines.append("&nbsp;")
                lines.append("")

            return "\n".join(lines)

        return str(data)

    def _load_wiki_data(self, core_name, quiz_type):
        """Loads data from the wiki for quiz generation."""
        parts = Path(core_name).parts
        if "wiki" in parts:
            idx = parts.index("wiki")
            if idx + 1 < len(parts):
                core_name = parts[idx + 1]
        elif "raw" in parts:
            idx = parts.index("raw")
            if idx + 1 < len(parts):
                core_name = parts[idx + 1]
        else:
            core_name = parts[0] if parts else core_name
            
        if core_name.endswith(".md"): core_name = core_name[:-3]
        elif core_name.endswith(".txt"): core_name = core_name[:-4]

        # Check standard new location first: wiki/<core_name>/extractions/<core_name>_vocabulary.md
        vocab_path = self.config.wiki_content_path / core_name / "extractions" / f"{core_name}_vocabulary.md"
        if not vocab_path.exists():
            vocab_paths = list(self.config.wiki_content_path.rglob(f"{core_name}_vocabulary.md"))
            vocab_path = vocab_paths[0] if vocab_paths else (self.config.wiki_content_path / core_name / f"{core_name}_vocabulary.md")

        cefr = None
        
        if vocab_path.exists():
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab_content = f.read()
                # Match both quoted and unquoted frontmatter: overall_cefr_level: "B2" or overall_cefr_level: B2
                m = re.search(r'overall_cefr_level:\s*["\']?([A-C][1-2])["\']?', vocab_content, re.IGNORECASE)
                if m:
                    cefr = m.group(1).upper()
        else:
            vocab_content = ""

        if not cefr:
            import logging
            logging.getLogger("librarian").warning(f"Could not resolve overall_cefr_level for {core_name}; falling back to B2.")
            cefr = "B2"

        if quiz_type == "vocabulary":
            if not vocab_path.exists(): return None
            return {"content": vocab_content, "cefr_level": cefr}

        elif quiz_type == "reading":
            # Check standard location: wiki/<core_name>/sources/<core_name>.md or .txt
            source_path = self.config.wiki_content_path / core_name / "sources" / f"{core_name}.md"
            if not source_path.exists():
                source_path = self.config.wiki_content_path / core_name / "sources" / f"{core_name}.txt"
            if not source_path.exists():
                sources_dir = self.config.wiki_content_path / core_name / "sources"
                if sources_dir.exists() and sources_dir.is_dir():
                    candidates = [f for f in sources_dir.iterdir() if f.is_file() and f.suffix in [".md", ".txt"]]
                    if candidates:
                        source_path = candidates[0]
            if not source_path or not source_path.exists(): return None
            with open(source_path, "r", encoding="utf-8") as f: passage = f.read()
            return {"passage": passage, "cefr_level": cefr}

        elif quiz_type == "translation":
            if not vocab_path.exists(): return None
            grammar_path = self.config.wiki_content_path / core_name / "extractions" / f"{core_name}_grammar.md"
            if not grammar_path.exists():
                grammar_paths = list(self.config.wiki_content_path.rglob(f"{core_name}_grammar.md"))
                grammar_path = grammar_paths[0] if grammar_paths else (self.config.wiki_content_path / core_name / f"{core_name}_grammar.md")
            grammar = ""
            if grammar_path.exists():
                with open(grammar_path, "r", encoding="utf-8") as f: grammar = f.read()
            return {"vocab_list": vocab_content, "grammar_list": grammar, "cefr_level": cefr}

        elif quiz_type == "listening":
            if not vocab_path.exists(): return None
            return {"vocab_list": vocab_content, "cefr_level": cefr}

        elif quiz_type in ["video", "listening"]:
            # For listening quiz, we check if there's an active media file first
            from .config import normalize_name
            normalized_core = normalize_name(core_name)
            unit_wiki_dir = self.config.wiki_content_path / core_name
            
            # Resolve actual unit_wiki_dir case-insensitively
            if self.config.wiki_content_path.exists():
                for child in self.config.wiki_content_path.iterdir():
                    if child.is_dir() and normalize_name(child.name) == normalized_core:
                        unit_wiki_dir = child
                        core_name = child.name  # Sync core_name
                        break

            source_path = None
            active_media_name = None
            if unit_wiki_dir.exists() and unit_wiki_dir.is_dir():
                # Dynamically resolve active media by scanning sources/media (most recently modified first)
                media_dir = unit_wiki_dir / "sources" / "media"
                if media_dir.exists() and media_dir.is_dir():
                    candidates = [f for f in media_dir.iterdir() if f.is_file() and f.suffix in [".md", ".txt"]]
                    if candidates:
                        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                        source_path = candidates[0]
                        active_media_name = source_path.name

            # Fallback if no active_media or active_media not found (only for video, or if listening needs it)
            if not source_path and quiz_type == "video":
                if unit_wiki_dir.exists() and unit_wiki_dir.is_dir():
                    # Check sources/media/ first for any md/txt transcript files
                    media_dir = unit_wiki_dir / "sources" / "media"
                    if media_dir.exists() and media_dir.is_dir():
                        for f in media_dir.iterdir():
                            if f.is_file() and f.suffix in [".md", ".txt"]:
                                source_path = f
                                active_media_name = f.name
                                break
                    
                    if not source_path:
                        # Look for explicit video_transcript files first
                        for f in unit_wiki_dir.rglob("*.md"):
                            if f.is_file():
                                try:
                                    with open(f, "r", encoding="utf-8") as f_read:
                                        test_content = f_read.read()
                                    if 'category: "video_transcript"' in test_content or 'video_type:' in test_content or 'video_url:' in test_content or 'source_url:' in test_content:
                                        source_path = f
                                        break
                                except Exception:
                                    pass

                # If no explicit video transcript found, try standard source paths
                if not source_path or not source_path.exists():
                    source_path = self.config.wiki_content_path / core_name / "sources" / f"{core_name}.md"
                    if not source_path.exists():
                        source_path = self.config.wiki_content_path / core_name / "sources" / f"{core_name}.txt"
                    
                    # Fallback to any md file in the unit directory (excluding compiled node outputs)
                    if not source_path.exists() and unit_wiki_dir.exists() and unit_wiki_dir.is_dir():
                        for f in unit_wiki_dir.glob("*.md"):
                            if f.is_file() and not any(suffix in f.name for suffix in ["_vocabulary", "_grammar", "_summary"]):
                                source_path = f
                                break
                        if not source_path.exists():
                            for f in unit_wiki_dir.rglob("*.md"):
                                if f.is_file() and not any(suffix in f.name for suffix in ["_vocabulary", "_grammar", "_summary"]):
                                    source_path = f
                                    break

            if quiz_type == "listening" and not source_path:
                # Normal listening quiz defaults to using vocab list (not a specific transcript file)
                if not vocab_path.exists(): return None
                return {"vocab_list": vocab_content, "cefr_level": cefr}

            if not source_path or not source_path.exists(): return None
            with open(source_path, "r", encoding="utf-8") as f: transcript = f.read()
            # Try multiple common keys for video URL (quoted or unquoted)
            for key in ["source_url", "video_url", "url", "source"]:
                m_url = re.search(fr'{key}:\s*["\']?([^\n"\']+)["\']?', transcript)
                if m_url and m_url.group(1).strip():
                    video_url = m_url.group(1).strip()
                    break
                    
            m_type = re.search(r'video_type:\s*["\']?([^\n"\']+)["\']?', transcript)
            if m_type:
                video_type = m_type.group(1).strip()
            elif video_url:
                # Auto-detect type based on URL
                url_lower = video_url.lower()
                if "youtube" in url_lower or "youtu.be" in url_lower:
                    video_type = "youtube"
                elif url_lower.endswith((".mp4", ".webm", ".ogg", ".mp3", ".wav")) or "/media/" in url_lower:
                    video_type = "local"
                else:
                    video_type = "youtube"  # default fallback if URL exists

            # If video_url is still empty and file is inside the media folder, resolve local companion or fallback
            if not video_url and ("media" in source_path.parts or (active_media_name and source_path.name == active_media_name)):
                parent_dir = source_path.parent
                found_companion = False
                for ext in [".mp4", ".mp3", ".webm", ".ogg", ".wav"]:
                    companion = parent_dir / f"{source_path.stem}{ext}"
                    if companion.exists():
                        video_url = f"/wiki/{core_name}/sources/media/{companion.name}"
                        video_type = "local"
                        found_companion = True
                        break
                if not found_companion:
                    # Fallback default local video url pointing to standard stem
                    video_url = f"/wiki/{core_name}/sources/media/{source_path.stem}.mp4"
                    video_type = "local"
            
            # Rewrite local video URL to point to /wiki/<UnitName>/sources/media/
            if video_type == "local" and video_url:
                import os
                video_filename = os.path.basename(video_url)
                video_url = f"/wiki/{core_name}/sources/media/{video_filename}"
                
            return {"transcript": transcript, "video_url": video_url, "video_type": video_type, "cefr_level": cefr, "source_stem": source_path.stem}

        return None

    def _render_handout(self, quiz_obj, template_name, audio_url=None, language=None):
        """Renders HTML templates with injected data."""
        template_path = Path(__file__).parent / "templates" / f"{template_name}.html"
        
        # 1. Convert to dict if it's a dataclass
        if dataclasses.is_dataclass(quiz_obj):
            data_dict = dataclasses.asdict(quiz_obj)
        else:
            data_dict = quiz_obj # It's already a dict from a JSON-file schema
            
        if not template_path.exists():
            return f"<html><body><pre>{json.dumps(data_dict, indent=2)}</pre></body></html>"

        with open(template_path, "r", encoding="utf-8") as f: html = f.read()
        
        if audio_url: data_dict["audio_url"] = audio_url
        if language: data_dict["target_language"] = language
        
        # Strip design_audit from questions/root if present so internal drafts do not leak to client HTML
        def _strip_design_audit(obj):
            if isinstance(obj, dict):
                return {k: _strip_design_audit(v) for k, v in obj.items() if k != "design_audit"}
            elif isinstance(obj, list):
                return [_strip_design_audit(item) for item in obj]
            return obj

        cleaned_data = _strip_design_audit(data_dict)
        json_data = json.dumps(cleaned_data, ensure_ascii=False)
        
        if "const quizData =" in html:
            return re.sub(r'const quizData = .*?;', lambda _: f'const quizData = {json_data};', html, flags=re.DOTALL)
        return html.replace("</body>", f"<script>const quizData = {json_data};</script></body>")

    def _interpolate_schema(self, schema, mapping):
        """Recursively interpolates placeholders in a JSON schema dict."""
        if not isinstance(schema, dict):
            return schema
        
        def _recurse(obj):
            if isinstance(obj, str):
                val = self._safe_format(obj, mapping)
                # If the resulting value is a pure integer string, convert to int
                # This is critical for JSON Schema fields like maxItems, minLength, etc.
                if re.match(r'^-?\d+$', val):
                    return int(val)
                return val
            elif isinstance(obj, dict):
                return {k: _recurse(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_recurse(v) for v in obj]
            return obj
            
        return _recurse(schema)

    def _safe_format(self, text, mapping):
        """Formats string with mapping, ignoring missing keys and non-stringable values."""
        for k, v in mapping.items():
            if isinstance(v, (str, int, float)):
                text = text.replace(f"{{{k}}}", str(v))
        return text

    def _render_mindmap(self, mindmap_dict):
        """Renders the horizontal Mind Map HTML page with injected data."""
        template_path = Path(__file__).parent / "templates" / "mindmap.html"
            
        if not template_path.exists():
            return f"<html><body><pre>{json.dumps(mindmap_dict, indent=2)}</pre></body></html>"

        with open(template_path, "r", encoding="utf-8") as f:
            html = f.read()
            
        json_data = json.dumps(mindmap_dict, ensure_ascii=False)
        
        if "const mindmapData =" in html:
            return re.sub(r'const mindmapData = .*?;', lambda _: f'const mindmapData = {json_data};', html, flags=re.DOTALL)
        return html.replace("</body>", f"<script>const mindmapData = {json_data};</script></body>")

    def _get_wiki_inventory(self):
        """Scans the wiki and raw directories recursively."""
        inventory = []
        wiki_dir = self.config.wiki_content_path
        if wiki_dir.exists():
            for f in wiki_dir.rglob("*.md"):
                if f.is_file():
                    if "_vocabulary" in f.name:
                        inventory.append({"name": f.stem, "type": "wiki_vocabulary", "path": f})
                    elif "_grammar" in f.name:
                        inventory.append({"name": f.stem, "type": "wiki_grammar", "path": f})
                    elif "_summary" in f.name:
                        inventory.append({"name": f.stem, "type": "wiki_summaries", "path": f})
                    else:
                        inventory.append({"name": f.stem, "type": "wiki_concepts", "path": f})
        
        # Add unit-specific sources to inventory as raw_source
        if wiki_dir.exists():
            for f in wiki_dir.rglob("sources/*.*"):
                if f.is_file() and f.suffix in [".md", ".txt"]:
                    try:
                        rel_name = str(f.relative_to(wiki_dir)).replace("\\", "/")
                        stem = f.stem
                        inventory.append({"name": rel_name, "type": "raw_source", "path": f})
                        inventory.append({"name": stem, "type": "raw_source", "path": f})
                    except Exception:
                        pass
        return inventory

processor = WikiProcessor()
