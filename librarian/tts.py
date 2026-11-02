import asyncio
import io
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request

from .config import config

KOKORO_RELEASE_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
KOKORO_MODEL_FILENAME = "kokoro-v1.0.int8.onnx"
KOKORO_VOICES_FILENAME = "voices-v1.0.bin"

# Offline fallback catalog for Edge-TTS English voices (used only if edge_tts.list_voices() is unreachable)
_EDGE_FALLBACK_VOICES = [
    {"ShortName": "en-US-AriaNeural", "Gender": "Female", "Locale": "en-US"},
    {"ShortName": "en-US-AvaMultilingualNeural", "Gender": "Female", "Locale": "en-US"},
    {"ShortName": "en-US-EmmaMultilingualNeural", "Gender": "Female", "Locale": "en-US"},
    {"ShortName": "en-US-JennyNeural", "Gender": "Female", "Locale": "en-US"},
    {"ShortName": "en-US-AvaNeural", "Gender": "Female", "Locale": "en-US"},
    {"ShortName": "en-US-EmmaNeural", "Gender": "Female", "Locale": "en-US"},
    {"ShortName": "en-US-MichelleNeural", "Gender": "Female", "Locale": "en-US"},
    {"ShortName": "en-US-AnaNeural", "Gender": "Female", "Locale": "en-US"},
    {"ShortName": "en-GB-SoniaNeural", "Gender": "Female", "Locale": "en-GB"},
    {"ShortName": "en-GB-LibbyNeural", "Gender": "Female", "Locale": "en-GB"},
    {"ShortName": "en-GB-MaisieNeural", "Gender": "Female", "Locale": "en-GB"},
    {"ShortName": "en-AU-NatashaNeural", "Gender": "Female", "Locale": "en-AU"},
    {"ShortName": "en-CA-ClaraNeural", "Gender": "Female", "Locale": "en-CA"},
    {"ShortName": "en-GB-RyanNeural", "Gender": "Male", "Locale": "en-GB"},
    {"ShortName": "en-US-AndrewMultilingualNeural", "Gender": "Male", "Locale": "en-US"},
    {"ShortName": "en-US-BrianMultilingualNeural", "Gender": "Male", "Locale": "en-US"},
    {"ShortName": "en-US-GuyNeural", "Gender": "Male", "Locale": "en-US"},
    {"ShortName": "en-US-ChristopherNeural", "Gender": "Male", "Locale": "en-US"},
    {"ShortName": "en-US-AndrewNeural", "Gender": "Male", "Locale": "en-US"},
    {"ShortName": "en-US-BrianNeural", "Gender": "Male", "Locale": "en-US"},
    {"ShortName": "en-US-EricNeural", "Gender": "Male", "Locale": "en-US"},
    {"ShortName": "en-US-RogerNeural", "Gender": "Male", "Locale": "en-US"},
    {"ShortName": "en-US-SteffanNeural", "Gender": "Male", "Locale": "en-US"},
    {"ShortName": "en-GB-ThomasNeural", "Gender": "Male", "Locale": "en-GB"},
    {"ShortName": "en-AU-WilliamMultilingualNeural", "Gender": "Male", "Locale": "en-AU"},
    {"ShortName": "en-CA-LiamNeural", "Gender": "Male", "Locale": "en-CA"},
]

# Offline fallback voice IDs for Kokoro v1.0 (if voices-v1.0.bin has not been downloaded yet)
_KOKORO_FALLBACK_VOICE_IDS = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]

_EDGE_FEMALE_PRIORITY = [
    "en-US-AriaNeural",
    "en-US-AvaMultilingualNeural",
    "en-US-EmmaMultilingualNeural",
    "en-US-JennyNeural",
    "en-GB-SoniaNeural",
    "en-GB-LibbyNeural",
]
_EDGE_MALE_PRIORITY = [
    "en-GB-RyanNeural",
    "en-US-AndrewMultilingualNeural",
    "en-US-BrianMultilingualNeural",
    "en-US-GuyNeural",
    "en-US-ChristopherNeural",
    "en-GB-ThomasNeural",
]

_KOKORO_FEMALE_PRIORITY = [
    "af_sarah", "af_heart", "af_bella", "af_nova", "af_nicole", "bf_emma", "bf_isabella"
]
_KOKORO_MALE_PRIORITY = [
    "am_michael", "am_fenrir", "am_puck", "am_adam", "bm_george", "bm_lewis", "bm_daniel"
]

# Natural given-name mapping for abstract Kokoro voice code names so listening dialogues always use real human names
_KOKORO_ABSTRACT_PERSONA_NAMES = {
    "af_alloy": "Allison",
    "af_aoede": "Audrey",
    "af_heart": "Hannah",
    "af_kore": "Claire",
    "af_nova": "Nora",
    "af_river": "Rachel",
    "af_sky": "Skye",
    "am_echo": "Ethan",
    "am_fenrir": "Frank",
    "am_onyx": "Oliver",
    "am_puck": "Parker",
    "am_santa": "Samuel",
    "bm_fable": "Felix",
}


def _run_coro_sync(coro, timeout: float = 45.0):
    """Safely executes an async coroutine from either synchronous or running-loop contexts."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result(timeout=timeout)
    return asyncio.run(coro)


class TTSService:
    def __init__(self):
        self._lock = threading.Lock()
        self._kokoro_instance = None
        self._voice_catalog_cache: dict[str, dict] = {}
        self._silent_mp3_cache: dict[float, bytes] = {}
        self._refresh()

    @property
    def kokoro_dir(self) -> Path:
        return Path(__file__).resolve().parent / "data" / "kokoro"

    def _ensure_kokoro_assets(self, require_model: bool = True) -> tuple[Path, Path]:
        """Ensures Kokoro ONNX model and voices-v1.0.bin exist locally, downloading on demand if missing."""
        k_dir = self.kokoro_dir
        k_dir.mkdir(parents=True, exist_ok=True)
        voices_path = k_dir / KOKORO_VOICES_FILENAME
        model_path = k_dir / KOKORO_MODEL_FILENAME

        targets = [(voices_path, f"{KOKORO_RELEASE_BASE}/{KOKORO_VOICES_FILENAME}")]
        if require_model:
            targets.append((model_path, f"{KOKORO_RELEASE_BASE}/{KOKORO_MODEL_FILENAME}"))

        for path_obj, url in targets:
            if not path_obj.exists() or path_obj.stat().st_size < 1024:
                tmp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")
                print(f"    - Downloading {path_obj.name} from {url} ...")
                urllib.request.urlretrieve(url, tmp_path)
                tmp_path.replace(path_obj)
        return model_path, voices_path

    def _get_kokoro_engine(self):
        """Lazily initializes and caches the in-process kokoro_onnx.Kokoro instance."""
        if self._kokoro_instance is not None:
            return self._kokoro_instance
        with self._lock:
            if self._kokoro_instance is not None:
                return self._kokoro_instance
            from kokoro_onnx import Kokoro
            model_path, voices_path = self._ensure_kokoro_assets(require_model=True)
            self._kokoro_instance = Kokoro(str(model_path), str(voices_path))
            return self._kokoro_instance

    def _extract_edge_voices(self) -> dict:
        """Extracts English Female and Male voices directly from the Python `edge_tts` library."""
        raw_voices = None
        try:
            import edge_tts
            all_voices = _run_coro_sync(edge_tts.list_voices(), timeout=8.0)
            if all_voices:
                raw_voices = [
                    v for v in all_voices
                    if str(v.get("Locale", "")).startswith(("en-US", "en-GB", "en-AU", "en-CA"))
                ]
        except Exception:
            raw_voices = None

        if not raw_voices:
            raw_voices = list(_EDGE_FALLBACK_VOICES)

        region_map = {
            "en-US": ("US", "Accent: American"),
            "en-GB": ("UK", "Accent: British"),
            "en-AU": ("AU", "Accent: Australian"),
            "en-CA": ("CA", "Accent: Canadian"),
        }

        female_list = []
        male_list = []
        seen = set()

        for v in raw_voices:
            short_name = str(v.get("ShortName") or "").strip()
            if not short_name or short_name in seen:
                continue
            seen.add(short_name)

            locale = str(v.get("Locale") or "en-US")
            region_code, accent_str = region_map.get(locale, ("EN", "Accent: Standard"))
            gender_raw = str(v.get("Gender") or "Female").strip().lower()
            is_female = gender_raw == "female"

            # Format persona name from ShortName, e.g. en-US-AvaMultilingualNeural -> Ava (Multilingual)
            raw_persona = short_name.split("-")[-1]
            if raw_persona.endswith("Neural"):
                raw_persona = raw_persona[:-6]
            if raw_persona.endswith("Multilingual"):
                speaker_name = raw_persona[:-12].strip().title()
                persona = f"{speaker_name} (Multilingual)"
            else:
                speaker_name = raw_persona.strip().title()
                persona = speaker_name

            gender_tag = "Female" if is_female else "Male"
            entry = {
                "value": short_name,
                "label": f"{region_code} {gender_tag}: {persona}",
                "speaker_name": speaker_name,
                "gender": "female" if is_female else "male",
                "accent": accent_str,
            }
            if is_female:
                female_list.append(entry)
            else:
                male_list.append(entry)

        female_list.sort(
            key=lambda x: (
                _EDGE_FEMALE_PRIORITY.index(x["value"])
                if x["value"] in _EDGE_FEMALE_PRIORITY
                else 999,
                x["label"],
            )
        )
        male_list.sort(
            key=lambda x: (
                _EDGE_MALE_PRIORITY.index(x["value"])
                if x["value"] in _EDGE_MALE_PRIORITY
                else 999,
                x["label"],
            )
        )

        return {
            "female": female_list,
            "male": male_list,
            "default_female": "en-US-AriaNeural",
            "default_male": "en-GB-RyanNeural",
        }

    def _extract_kokoro_voices(self) -> dict:
        """Extracts English Female and Male voices directly from `kokoro-onnx` (`voices-v1.0.bin`)."""
        voice_ids = []
        try:
            import numpy as np
            voices_path = self.kokoro_dir / KOKORO_VOICES_FILENAME
            if not voices_path.exists():
                _, voices_path = self._ensure_kokoro_assets(require_model=False)
            if voices_path.exists():
                archive = np.load(str(voices_path))
                voice_ids = sorted(list(archive.keys()))
        except Exception:
            voice_ids = []

        if not voice_ids:
            voice_ids = list(_KOKORO_FALLBACK_VOICE_IDS)

        prefix_meta = {
            "af_": ("US", "Female", "female", "Accent: American"),
            "bf_": ("UK", "Female", "female", "Accent: British"),
            "am_": ("US", "Male", "male", "Accent: American"),
            "bm_": ("UK", "Male", "male", "Accent: British"),
        }

        female_list = []
        male_list = []

        for vid in voice_ids:
            prefix = vid[:3].lower()
            if prefix not in prefix_meta:
                continue
            region_code, gender_tag, gender_norm, accent_str = prefix_meta[prefix]
            persona = vid[3:].replace("_", " ").title()
            speaker_name = _KOKORO_ABSTRACT_PERSONA_NAMES.get(vid.lower(), persona)
            label_persona = f"{persona} ({speaker_name})" if speaker_name != persona else persona
            entry = {
                "value": vid,
                "label": f"{region_code} {gender_tag}: {label_persona}",
                "speaker_name": speaker_name,
                "gender": gender_norm,
                "accent": accent_str,
            }
            if gender_norm == "female":
                female_list.append(entry)
            else:
                male_list.append(entry)

        female_list.sort(
            key=lambda x: (
                _KOKORO_FEMALE_PRIORITY.index(x["value"])
                if x["value"] in _KOKORO_FEMALE_PRIORITY
                else 999,
                x["label"],
            )
        )
        male_list.sort(
            key=lambda x: (
                _KOKORO_MALE_PRIORITY.index(x["value"])
                if x["value"] in _KOKORO_MALE_PRIORITY
                else 999,
                x["label"],
            )
        )

        return {
            "female": female_list,
            "male": male_list,
            "default_female": "af_sarah",
            "default_male": "am_michael",
        }

    def get_available_voices(self, engine: str | None = None, force_refresh: bool = False) -> dict:
        """
        Returns the extracted Female & Male voice catalogs from the Python TTS controls.
        If `engine` is specified ('edge' or 'kokoro'), returns that engine's catalog dict;
        otherwise returns {"edge": ..., "kokoro": ...}.
        """
        if force_refresh or "edge" not in self._voice_catalog_cache:
            self._voice_catalog_cache["edge"] = self._extract_edge_voices()
        if force_refresh or "kokoro" not in self._voice_catalog_cache:
            self._voice_catalog_cache["kokoro"] = self._extract_kokoro_voices()

        if engine:
            eng_key = engine.lower()
            return self._voice_catalog_cache.get(eng_key, self._voice_catalog_cache["edge"])
        return self._voice_catalog_cache

    def _refresh(self):
        raw_engine = str(config.get("tts_engine") or "edge").strip().lower()
        if raw_engine not in ("edge", "kokoro"):
            raw_engine = "edge"
            config.update_config("tts_engine", "edge")
        self.tts_engine = raw_engine

        catalog = self.get_available_voices(self.tts_engine)
        female_ids = {v["value"] for v in catalog["female"]}
        male_ids = {v["value"] for v in catalog["male"]}
        default_female = catalog["default_female"]
        default_male = catalog["default_male"]

        voice_a = config.get("tts_voice_a")
        voice_b = config.get("tts_voice_b")

        # Self-heal legacy swapped configs where Voice A was set to a Male voice and Voice B to a Female voice
        if voice_a in male_ids and voice_b in female_ids:
            voice_a, voice_b = voice_b, voice_a
            config.update_config("tts_voice_a", voice_a)
            config.update_config("tts_voice_b", voice_b)

        if voice_a not in female_ids:
            self.voice_a = default_female
            config.update_config("tts_voice_a", default_female)
        else:
            self.voice_a = voice_a

        if voice_b not in male_ids:
            self.voice_b = default_male
            config.update_config("tts_voice_b", default_male)
        else:
            self.voice_b = voice_b

    def get_gender(self, voice_name: str) -> str:
        """Resolves exact gender ('female' or 'male') from extracted voice catalogs."""
        v = str(voice_name or "").strip()
        catalogs = self.get_available_voices()
        for eng_cat in catalogs.values():
            for item in eng_cat.get("female", []):
                if item["value"] == v:
                    return "female"
            for item in eng_cat.get("male", []):
                if item["value"] == v:
                    return "male"
        v_low = v.lower()
        if v_low.startswith(("am_", "bm_")) or any(
            m in v_low for m in ("ryan", "guy", "christopher", "thomas", "andrew", "brian", "eric", "roger", "steffan", "william", "liam")
        ):
            return "male"
        return "female"

    def get_accent(self, voice_name: str) -> str:
        """Resolves speaker accent tag ('Accent: American', 'Accent: British', etc.) from extracted catalogs."""
        v = str(voice_name or "").strip()
        catalogs = self.get_available_voices()
        for eng_cat in catalogs.values():
            for item in eng_cat.get("female", []) + eng_cat.get("male", []):
                if item["value"] == v:
                    return item.get("accent", "Accent: Standard")
        v_low = v.lower()
        if "gb" in v_low or v_low.startswith(("bf_", "bm_")):
            return "Accent: British"
        if "au" in v_low:
            return "Accent: Australian"
        if "ca" in v_low:
            return "Accent: Canadian"
        return "Accent: American"

    def get_speaker_name(self, voice_name: str) -> str:
        """Resolves a clean human first name directly from the TTS voice ID (e.g. 'en-US-AriaNeural' -> 'Aria', 'af_sarah' -> 'Sarah')."""
        v = str(voice_name or "").strip()
        if not v:
            return "Aria"
        catalogs = self.get_available_voices()
        for eng_cat in catalogs.values():
            for item in eng_cat.get("female", []) + eng_cat.get("male", []):
                if item["value"] == v and item.get("speaker_name"):
                    return str(item["speaker_name"])
        v_low = v.lower()
        if v_low in _KOKORO_ABSTRACT_PERSONA_NAMES:
            return _KOKORO_ABSTRACT_PERSONA_NAMES[v_low]
        if v_low.startswith(("af_", "bf_", "am_", "bm_")):
            return v[3:].replace("_", " ").strip().title()
        tail = v.split("-")[-1]
        for suffix in ("MultilingualNeural", "Neural", "Multilingual"):
            if tail.endswith(suffix):
                tail = tail[:-len(suffix)]
        return tail.strip().title() or "Aria"

    def _get_gender(self, voice_name: str) -> str:
        return self.get_gender(voice_name)

    def _get_silent_mp3_bytes(self, duration: float = 0.35) -> bytes:
        """Generates a short silent 24kHz 48kbps mono MP3 frame for natural inter-turn pauses."""
        if duration in self._silent_mp3_cache:
            return self._silent_mp3_cache[duration]
        try:
            proc = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "lavfi",
                    "-i", "anullsrc=r=24000:cl=mono",
                    "-t", f"{duration:.2f}",
                    "-codec:a", "libmp3lame",
                    "-b:a", "48k",
                    "-ar", "24000",
                    "-ac", "1",
                    "-f", "mp3",
                    "pipe:1",
                ],
                capture_output=True,
                check=True,
                timeout=10,
            )
            self._silent_mp3_cache[duration] = proc.stdout
            return proc.stdout
        except Exception:
            return b""

    def _encode_pcm_to_mp3(self, samples, sample_rate: int) -> bytes | None:
        """Encodes float32 PCM audio samples to MP3 bytes in memory via soundfile + ffmpeg."""
        try:
            import soundfile as sf
            wav_buf = io.BytesIO()
            sf.write(wav_buf, samples, sample_rate, format="WAV", subtype="PCM_16")
            wav_bytes = wav_buf.getvalue()
            try:
                proc = subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-f", "wav",
                        "-i", "pipe:0",
                        "-codec:a", "libmp3lame",
                        "-b:a", "64k",
                        "-ar", str(sample_rate),
                        "-ac", "1",
                        "-f", "mp3",
                        "pipe:1",
                    ],
                    input=wav_bytes,
                    capture_output=True,
                    check=True,
                    timeout=30,
                )
                return proc.stdout if proc.stdout else wav_bytes
            except Exception:
                return wav_bytes
        except Exception as e:
            print(f"    - ! Audio encoding error: {e}")
            return None

    def _fetch_edge_native(self, text: str, voice: str) -> bytes | None:
        """Synthesizes MP3 audio in-process via Python `edge_tts`."""
        try:
            import edge_tts
        except ImportError:
            return None

        async def _synthesize() -> bytes:
            comm = edge_tts.Communicate(text, voice)
            buf = bytearray()
            async for chunk in comm.stream():
                if chunk.get("type") == "audio" and chunk.get("data"):
                    buf.extend(chunk["data"])
            return bytes(buf)

        try:
            audio_bytes = _run_coro_sync(_synthesize(), timeout=45.0)
            return audio_bytes if audio_bytes else None
        except Exception as e:
            print(f"    - ! Native edge-tts error for voice '{voice}': {e}")
            return None

    def _synthesize_kokoro_samples(self, text: str, voice: str):
        """Synthesizes float32 PCM samples + sample_rate in-process via `kokoro_onnx`."""
        try:
            kokoro = self._get_kokoro_engine()
            lang = "en-gb" if str(voice).lower().startswith(("bf_", "bm_")) else "en-us"
            samples, sr = kokoro.create(text, voice=voice, speed=1.0, lang=lang)
            return samples, sr
        except Exception as e:
            print(f"    - ! Native kokoro-onnx error for voice '{voice}': {e}")
            return None, None

    def _fetch_kokoro_native(self, text: str, voice: str) -> bytes | None:
        """Synthesizes MP3 audio in-process via Python `kokoro_onnx`."""
        samples, sr = self._synthesize_kokoro_samples(text, voice)
        if samples is None or sr is None:
            return None
        return self._encode_pcm_to_mp3(samples, sr)

    def _sanitize_voice_for_engine(self, engine: str, voice: str | None) -> str:
        catalog = self.get_available_voices(engine)
        all_valid = {
            v["value"]
            for v in (catalog.get("female", []) + catalog.get("male", []))
        }
        if voice and voice in all_valid:
            return voice
        return catalog.get("default_female", "en-US-AriaNeural")

    def _fetch_audio_binary(self, text: str, voice: str | None, engine_override: str | None = None) -> bytes | None:
        """Synthesizes raw MP3 audio binary using the active in-process Python TTS engine."""
        engine = (engine_override or getattr(self, "tts_engine", "edge") or "edge").lower()
        sanitized_voice = self._sanitize_voice_for_engine(engine, voice)

        if engine == "kokoro":
            return self._fetch_kokoro_native(text, sanitized_voice)
        return self._fetch_edge_native(text, sanitized_voice)

    def preview_voice(self, engine: str, voice: str, text: str | None = None) -> bytes | None:
        """Synthesizes a short natural voice preview clip for the Dashboard Settings modal."""
        eng = (engine or "edge").strip().lower()
        sanitized_voice = self._sanitize_voice_for_engine(eng, voice)
        if not text:
            persona = self.get_speaker_name(sanitized_voice)
            text = f"Hi there! I'm {persona}. Let's listen carefully to today's conversation."
        return self._fetch_audio_binary(text, sanitized_voice, engine_override=eng)

    def generate_audio(self, text: str, output_path, voice: str | None = None) -> bool:
        """Single-voice audio file generation."""
        self._refresh()
        content = self._fetch_audio_binary(text, voice or self.voice_a)
        if content:
            with open(output_path, "wb") as f:
                f.write(content)
            return True
        return False

    def process_script(
        self,
        script,
        output_dir=None,
        file_prefix=None,
        speaker_1=None,
        speaker_2=None,
        speaker_1_gender=None,
        speaker_2_gender=None,
        return_binary=False,
    ):
        """
        Processes a full dialogue script with dual-speaker support (Female Voice A + Male Voice B).
        Synthesizes via in-process Python `edge-tts` or `kokoro-onnx` with natural 350ms inter-turn pauses.
        """
        self._refresh()
        if not script:
            return None

        engine = self.tts_engine.lower()
        print(f"    - Generating dual-speaker audio ({engine}) for {len(script)} turns...")

        voice_a_gender = self.get_gender(self.voice_a)
        voice_b_gender = self.get_gender(self.voice_b)

        speaker_map = {}
        s1_g = str(speaker_1_gender or "").lower()
        s2_g = str(speaker_2_gender or "").lower()

        if speaker_1 and speaker_2 and s1_g and s2_g:
            if s1_g == voice_a_gender and s2_g == voice_b_gender:
                speaker_map[speaker_1] = self.voice_a
                speaker_map[speaker_2] = self.voice_b
            elif s1_g == voice_b_gender and s2_g == voice_a_gender:
                speaker_map[speaker_1] = self.voice_b
                speaker_map[speaker_2] = self.voice_a
            else:
                speaker_map[speaker_1] = self.voice_a
                speaker_map[speaker_2] = self.voice_b
        else:
            if speaker_1:
                speaker_map[speaker_1] = self.voice_a
            if speaker_2:
                speaker_map[speaker_2] = self.voice_b

        unique_speakers = []
        for turn in script:
            s = turn.get("speaker", "Unknown")
            if s not in unique_speakers:
                unique_speakers.append(s)

        voice_slots = [self.voice_a, self.voice_b]
        voice_idx = 0
        for s in unique_speakers:
            if s not in speaker_map:
                speaker_map[s] = voice_slots[voice_idx % 2]
                voice_idx += 1

        combined_audio = b""

        if engine == "kokoro":
            import numpy as np
            pcm_segments = []
            sample_rate = 24000
            for i, turn in enumerate(script):
                speaker = turn.get("speaker", "Unknown")
                text = turn.get("text", "").strip()
                if not text:
                    continue
                voice = self._sanitize_voice_for_engine("kokoro", speaker_map.get(speaker, self.voice_a))
                print(f"      [Turn {i+1}] {speaker} ({voice}): {text[:35]}...")
                samples, sr = self._synthesize_kokoro_samples(text, voice)
                if samples is not None and sr is not None:
                    sample_rate = sr
                    if pcm_segments:
                        # Insert 350ms natural conversational pause between speakers
                        pcm_segments.append(np.zeros(int(0.35 * sample_rate), dtype=np.float32))
                    pcm_segments.append(samples)
                else:
                    print(f"      ! Failed to generate audio for turn {i+1}")
            if pcm_segments:
                full_waveform = np.concatenate(pcm_segments)
                combined_audio = self._encode_pcm_to_mp3(full_waveform, sample_rate) or b""
        else:
            # Concurrent synthesis for Edge-TTS (cuts 8-turn latency to ~1.5s)
            turn_tasks = []
            for i, turn in enumerate(script):
                speaker = turn.get("speaker", "Unknown")
                text = turn.get("text", "").strip()
                voice = self._sanitize_voice_for_engine("edge", speaker_map.get(speaker, self.voice_a))
                print(f"      [Turn {i+1}] {speaker} ({voice}): {text[:35]}...")
                turn_tasks.append((i, text, voice))

            async def _synthesize_all_edge():
                import edge_tts
                sem = asyncio.Semaphore(4)

                async def _one(idx: int, txt: str, vc: str) -> tuple[int, bytes]:
                    if not txt:
                        return idx, b""
                    async with sem:
                        comm = edge_tts.Communicate(txt, vc)
                        buf = bytearray()
                        async for chunk in comm.stream():
                            if chunk.get("type") == "audio" and chunk.get("data"):
                                buf.extend(chunk["data"])
                        return idx, bytes(buf)

                results = await asyncio.gather(*[_one(idx, txt, vc) for idx, txt, vc in turn_tasks], return_exceptions=True)
                ordered = []
                for res in results:
                    if isinstance(res, Exception):
                        ordered.append(b"")
                    else:
                        ordered.append(res[1])
                return ordered

            try:
                turn_binaries = _run_coro_sync(_synthesize_all_edge(), timeout=60.0)
            except Exception as e:
                print(f"    - ! Concurrent edge-tts batch error, falling back to sequential: {e}")
                turn_binaries = [self._fetch_edge_native(txt, vc) or b"" for _, txt, vc in turn_tasks]

            pause_mp3 = self._get_silent_mp3_bytes(0.35)
            non_empty_turns = [b for b in turn_binaries if b]
            if non_empty_turns:
                combined_audio = pause_mp3.join(non_empty_turns) if pause_mp3 else b"".join(non_empty_turns)

        if combined_audio:
            if return_binary:
                return combined_audio

            if output_dir and file_prefix:
                audio_filename = f"{file_prefix}_dialogue.mp3"
                audio_path = Path(output_dir) / audio_filename
                with open(audio_path, "wb") as f:
                    f.write(combined_audio)
                print(f"    - Success: {audio_filename} saved.")
                return audio_filename

        return None


tts_service = TTSService()
