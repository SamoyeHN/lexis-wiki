import os
import json
from pathlib import Path

DEFAULT_CONFIG = {
    "api_type": "ollama", # "ollama" or "openai"
    "api_url": "http://localhost:11434",
    "api_key": "ollama", # Required for OpenAI
    "model": "gemma4:e4b",
    "wiki_dir": "wiki",
    "quiz_defaults": {
        "reading": 10,
        "translation": 5,
        "vocabulary": 20,
        "listening": 5,
        "video": 5
    },
    "compile_defaults": {
        "vocabulary": 20,
        "grammar": 5,
        "concepts": 3,
        "max_parallel": 3
    },
    "tts_engine": "kokoro",
    "tts_url": "http://localhost:8880/v1/audio/speech",
    "tts_voice_a": "af_sarah",
    "tts_voice_b": "am_michael",
    "tts_model": "tts-1",
    "tts_api_key": "any_string",
    "target_language": "Simplified Chinese",

}

class Config:
    def __init__(self):
        self.data = DEFAULT_CONFIG.copy()
        self.project_root = self.find_project_root()
        self.load_config()
        self.ensure_dirs()

    def find_project_root(self):
        """Searches upwards for wiki_config.json to determine the project root."""
        # 1. Search upwards from the current working directory first
        cwd = Path.cwd().resolve()
        for parent in [cwd] + list(cwd.parents):
            if (parent / "wiki_config.json").exists():
                return parent

        # 2. Fallback to searching upwards from the file's directory
        current = Path(__file__).resolve().parent
        for parent in [current] + list(current.parents):
            if (parent / "wiki_config.json").exists():
                return parent
        # Fallback to the directory containing the package if not found
        return current.parent

    def load_config(self):
        config_path = self.project_root / "wiki_config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    user_config = json.load(f)
                    
                    # Migration: ollama_url -> api_url
                    if "ollama_url" in user_config and "api_url" not in user_config:
                        user_config["api_url"] = user_config.pop("ollama_url")
                    
                    self.data.update(user_config)
            except Exception as e:
                print(f"Warning: Could not load {config_path}: {e}")

    def ensure_dirs(self):
        """Create the wiki base directory if it doesn't exist."""
        try:
            self.wiki_content_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create directories: {e}")

    def initialize_project(self, target_path=None):
        """Creates a new project structure in the target path."""
        target_path = Path(target_path or os.getcwd()).resolve()
        target_path.mkdir(parents=True, exist_ok=True)
        config_path = target_path / "wiki_config.json"
        
        if config_path.exists():
            return False, f"Error: Configuration already exists at {config_path}"

        try:
            # 1. Create config file using the latest DEFAULT_CONFIG
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_CONFIG, f, indent=4)
            
            # 2. Re-initialize self with the new root
            self.project_root = target_path
            self.load_config()
            self.ensure_dirs()

            return True, str(config_path)
        except Exception as e:
            return False, str(e)

    def update_model(self, model_name):
        """Updates the model in the config file."""
        return self.update_config("model", model_name)

    def update_config(self, key, value):
        """Generic config update helper."""
        config_path = self.project_root / "wiki_config.json"
        self.data[key] = value
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=4)
            return True, f"{key} updated to {value}"
        except Exception as e:
            return False, str(e)

    def get(self, key, default=None):
        return self.data.get(key, default)

    @property   
    def wiki_path(self):
        # wiki_root is now strictly the project_root determined by find_project_root
        return self.project_root

    @property
    def wiki_content_path(self):
        return self.wiki_path / self.get("wiki_dir", "wiki")

def normalize_name(n: str) -> str:
    import re
    n = re.sub(r'\.(md|txt|html|json)$', '', n, flags=re.I)  # drop extension
    return re.sub(r'[^a-z0-9]', '', n.lower())               # drop case/space/underscore

def extract_unit_stem(path_str: str) -> str:
    """Robustly extracts the unit folder name/stem from any path or file string."""
    from pathlib import Path
    parts = Path(path_str).parts
    if "wiki" in parts:
        idx = parts.index("wiki")
        stem = parts[idx + 1] if idx + 1 < len(parts) else parts[-1]
    elif "raw" in parts:
        idx = parts.index("raw")
        stem = parts[idx + 1] if idx + 1 < len(parts) else parts[-1]
    else:
        stem = parts[0] if parts else path_str

    if stem.endswith(".md"):
        stem = stem[:-3]
    elif stem.endswith(".txt"):
        stem = stem[:-4]
    return stem

def is_supplement_unit(unit_name: str, filename_stem: str, current_unit: str = "") -> bool:
    """
    Robustly determines if a media or transcript file should be saved as a supplement/companion 
    (under sources/media/) rather than overwriting/representing the primary source file under sources/.
    """
    from pathlib import Path
    
    wiki_dir = Path(config.project_root) / "wiki"
    sources_dir = wiki_dir / unit_name / "sources"
    
    # 1. Check if the primary source file already exists and is standard text
    primary_exists_and_is_standard_text = False
    if sources_dir.exists():
        for ext in [".md", ".txt"]:
            candidate = sources_dir / f"{unit_name}{ext}"
            if candidate.exists() and candidate.is_file():
                try:
                    with open(candidate, "r", encoding="utf-8") as f_check:
                        head = f_check.read(1000)
                    if 'category: "video_transcript"' not in head and 'video_type:' not in head and "Transcription is currently in progress" not in head:
                        primary_exists_and_is_standard_text = True
                except Exception:
                    pass
                break

    if primary_exists_and_is_standard_text:
        return True
        
    # 2. If we are operating within an active/current unit and the uploaded name doesn't match the unit name, it's a supplement
    if current_unit:
        if normalize_name(unit_name) != normalize_name(filename_stem):
            return True
            
    # 3. Otherwise, check if any other primary source file exists in sources/ that doesn't match this name
    else:
        if sources_dir.exists():
            for f in sources_dir.iterdir():
                if f.is_file() and f.suffix in [".md", ".txt"]:
                    if f.stem != filename_stem:
                        return True
                        
    return False

config = Config()
