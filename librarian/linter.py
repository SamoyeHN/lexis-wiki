import os
import re
from pathlib import Path
from .config import config

class WikiLinter:
    def __init__(self):
        self.wiki_dir = config.wiki_content_path

    def get_all_valid_links(self):
        all_links = set()
        for root, _, files in os.walk(self.wiki_dir):
            for file in files:
                if file.endswith('.md'):
                    # Relative path from wiki_dir without .md
                    rel_path = os.path.relpath(os.path.join(root, file), self.wiki_dir)
                    rel_path = rel_path.replace('.md', '').replace('\\', '/')
                    all_links.add(rel_path)
                    # Simple filename for shorthand links
                    all_links.add(file.replace('.md', ''))
        return all_links

    def check_links(self):
        valid_links = self.get_all_valid_links()
        broken_links = []
        
        for root, _, files in os.walk(self.wiki_dir):
            for file in files:
                if file.endswith('.md'):
                    file_path = Path(root) / file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    links = re.findall(r'\[\[(.*?)(?:\|.*?)?\]\]', content)
                    for link in links:
                        link = link.strip()
                        if link not in valid_links and not link.startswith('http'):
                            broken_links.append((str(file_path), link))
        return broken_links

    def normalize_tags(self):
        def to_kebab(tag):
            tag = re.sub(r'\.md$', '', tag, flags=re.IGNORECASE)
            tag = re.sub(r'[\s_]+', '-', tag)
            return tag.lower()

        updated_count = 0
        for root, _, files in os.walk(self.wiki_dir):
            for file in files:
                if file.endswith('.md'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                    except UnicodeDecodeError:
                        continue # Skip binary or weird files
                    
                    match = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
                    if match:
                        frontmatter = match.group(1)
                        tags_match = re.search(r'tags:\s*\[(.*?)\]', frontmatter)
                        if tags_match:
                            tags_str = tags_match.group(1)
                            tags = [to_kebab(t.strip().strip('"').strip("'")) for t in tags_str.split(',') if t.strip()]
                            
                            unique_tags = []
                            seen = set()
                            for t in tags:
                                if t not in seen:
                                    unique_tags.append(t)
                                    seen.add(t)
                            
                            new_tags_line = 'tags: [' + ', '.join(unique_tags) + ']'
                            if new_tags_line != tags_match.group(0):
                                new_frontmatter = frontmatter.replace(tags_match.group(0), new_tags_line)
                                new_content = content.replace(frontmatter, new_frontmatter, 1)
                                with open(file_path, 'w', encoding='utf-8', newline='') as f:
                                    f.write(new_content)
                                updated_count += 1
        return updated_count
    def find_orphans(self):
        """
        Finds orphan markdown nodes and handouts.
        An orphan is a markdown file or handout with no matching source file.
        """
        # Collect all sources
        raw_files = set()
        raw_stems = set()

        # Scan new structure unit sources
        if self.wiki_dir.exists():
            for p in self.wiki_dir.iterdir():
                if p.is_dir() and not p.name.startswith("."):
                    sources_path = p / "sources"
                    if sources_path.exists():
                        for f in sources_path.iterdir():
                            if f.is_file():
                                raw_files.add(f.name)
                                raw_stems.add(f.stem)

        orphans = []

        # 1. Scan markdown files
        for root, _, files in os.walk(self.wiki_dir):
            for file in files:
                if file.endswith('.md'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                    except (UnicodeDecodeError, IOError):
                        continue

                    # Match source: "[[filename]]" with optional single/double quotes or none
                    match = re.search(r'source:\s*["\']?\[\[(.*?)\]\]["\']?', content)
                    if match:
                        source_ref = match.group(1).strip()
                        exists = False
                        if source_ref in raw_files or source_ref in raw_stems:
                            exists = True
                        else:
                            ref_path = Path(source_ref)
                            ref_stem = ref_path.stem
                            if ref_stem in raw_stems:
                                  exists = True
                        
                        if not exists:
                            orphans.append(str(file_path))

        # 2. Scan handouts/
        handouts_dirs = []
        legacy_handouts = self.wiki_dir / "handouts"
        if legacy_handouts.exists():
            handouts_dirs.append(legacy_handouts)
            
        if self.wiki_dir.exists():
            for p in self.wiki_dir.iterdir():
                if p.is_dir() and not p.name.startswith(".") and p.name != "handouts":
                    h_path = p / "handouts"
                    if h_path.exists():
                        handouts_dirs.append(h_path)
                    
        for h_dir in handouts_dirs:
            for f in h_dir.iterdir():
                if f.is_file() and f.name.endswith('.html'):
                    name = f.name
                    core_name = name
                    for suffix in ["_vocabulary_quiz.html", "_reading_quiz.html", "_translation_quiz.html", "_listening_quiz.html"]:
                        if name.endswith(suffix):
                            core_name = name[:-len(suffix)]
                            break
                    else:
                        if name.endswith("_quiz.html"):
                            core_name = name[:-10]
                        else:
                            continue
                    
                    if core_name not in raw_stems:
                        orphans.append(str(f))

        return sorted(list(set(orphans)))

linter = WikiLinter()

