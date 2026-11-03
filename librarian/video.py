import os
import re
from pathlib import Path
from urllib.parse import urlparse
from .config import config, normalize_name, extract_unit_stem, is_supplement_unit

class VideoService:
    def __init__(self):
        pass

    def normalize_remote_input(self, url_or_path):
        """Extracts a clean URL from share text (e.g. Bilibili share text) or expands bare BV IDs."""
        raw = str(url_or_path or "").strip()
        if not raw:
            return raw
        # Bare Bilibili BV ID (12 chars starting with BV)
        if re.match(r'^BV[0-9A-Za-z]{10}$', raw):
            return f"https://www.bilibili.com/video/{raw}"
        # Extract embedded http(s) URL if user pasted share text like "【Title】 https://www.bilibili.com/..."
        if not os.path.exists(raw):
            url_match = re.search(r'(https?://[^\s"\'<>]+)', raw)
            if url_match:
                return url_match.group(1).rstrip('.,;!)】]')
        return raw

    def is_remote_url(self, url_or_path):
        """Returns True if the normalized input is an HTTP(S) URL."""
        norm = self.normalize_remote_input(url_or_path)
        return norm.startswith(("http://", "https://"))

    def extract_youtube_id(self, url):
        """Extracts the 11-character YouTube video ID from a YouTube URL or bare 11-char ID."""
        norm = self.normalize_remote_input(url)
        yt_pattern = r'(?:youtube\.com/(?:watch\?.*?v=|embed/|shorts/|v/)|youtu\.be/)([0-9A-Za-z_-]{11})(?:[?&/#]|$)'
        match = re.search(yt_pattern, norm)
        if match:
            return match.group(1)
        # If input is a bare 11-character ID (and not a file path, URL, or Bilibili ID)
        if (
            len(norm) == 11
            and re.match(r'^[0-9A-Za-z_-]+$', norm)
            and not norm.lower().startswith(('http', 'bv', 'av'))
            and not os.path.exists(norm)
        ):
            return norm
        return None

    def _build_ytdlp_headers(self, url):
        """Builds site-appropriate HTTP headers (setting Referer to bilibili.com for Bilibili, etc.)."""
        lower_url = (url or "").lower()
        if "bilibili.com" in lower_url or "b23.tv" in lower_url:
            referer = "https://www.bilibili.com/"
        elif "youtube.com" in lower_url or "youtu.be" in lower_url:
            referer = "https://www.youtube.com/"
        else:
            parsed = urlparse(url or "")
            referer = f"{parsed.scheme}://{parsed.netloc}/" if parsed.scheme and parsed.netloc else "https://www.google.com/"
        return {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Referer': referer,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        }

    @staticmethod
    def format_timestamp(seconds):
        """Formats float seconds into [MM:SS] or [HH:MM:SS]."""
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        if hrs > 0:
            return f"[{hrs:02d}:{mins:02d}:{secs:05.2f}]"
        else:
            return f"[{mins:02d}:{secs:05.2f}]"

    @staticmethod
    def parse_timestamp_to_seconds(ts_str: str) -> float:
        """Parses '[MM:SS.ss]' or '[HH:MM:SS.ss]' into float seconds."""
        s = (ts_str or "").strip().strip("[]")
        parts = s.split(":")
        try:
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            return float(parts[0])
        except Exception:
            return 0.0

    @classmethod
    def reflow_transcript_lines(cls, lines_or_text):
        """
        Reflows acoustic/subtitle-chopped '[MM:SS.ss] fragment' lines into complete
        grammatical sentences with interpolated start timestamps so words and multi-word
        expressions are never split across line breaks.
        Accepts either a list[str] of '[MM:SS.ss] ...' lines or a full markdown/text string.
        """
        is_str_input = isinstance(lines_or_text, str)
        raw_lines = lines_or_text.splitlines() if is_str_input else list(lines_or_text or [])
        if not raw_lines:
            return "" if is_str_input else []

        header_lines = []
        body_lines = []
        in_frontmatter = False
        ts_line_re = re.compile(r"^\s*\[(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)\]\s*(.+)$")

        for idx, line in enumerate(raw_lines):
            stripped = line.strip()
            if idx == 0 and stripped == "---":
                in_frontmatter = True
                header_lines.append(line)
                continue
            if in_frontmatter:
                header_lines.append(line)
                if stripped == "---":
                    in_frontmatter = False
                continue
            if not body_lines and not ts_line_re.match(line):
                header_lines.append(line)
                continue
            body_lines.append(line)

        segments = []
        for line in body_lines:
            m = ts_line_re.match(line)
            if m:
                sec = cls.parse_timestamp_to_seconds(m.group(1))
                segments.append((sec, m.group(2).strip()))
            elif line.strip() and segments:
                prev_sec, prev_txt = segments[-1]
                segments[-1] = (prev_sec, f"{prev_txt} {line.strip()}")

        if not segments:
            return lines_or_text if is_str_input else raw_lines

        _abbrev_no_split = {
            "mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "jr.", "sr.",
            "vs.", "etc.", "e.g.", "i.e.", "u.s.", "u.k.", "a.m.", "p.m.",
        }

        def _is_sentence_end(word: str) -> bool:
            w_clean = re.sub(r'["\'”’)\]]+$', "", word)
            if not w_clean or w_clean[-1] not in (".", "?", "!"):
                return False
            low = w_clean.lower()
            if low in _abbrev_no_split:
                return False
            if re.match(r"^[a-z]\.$", low):
                return False
            return True

        def _fmt_ts(seconds: float) -> str:
            hrs = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = seconds % 60
            if hrs > 0:
                return f"[{hrs:02d}:{mins:02d}:{secs:05.2f}]"
            return f"[{mins:02d}:{secs:05.2f}]"

        # Idempotency guard: if the transcript is already sentence-aligned (e.g., produced
        # via Whisper word_timestamps=True where >=85% of lines end at a sentence boundary
        # and lack mid-line sentence splits), preserve the exact acoustic timestamps untouched.
        terminal_ok = 0
        has_internal_split = False
        for _, txt in segments:
            ws = txt.split()
            if ws and _is_sentence_end(ws[-1]):
                terminal_ok += 1
            if any(_is_sentence_end(w) for w in ws[:-1]):
                has_internal_split = True
        if not has_internal_split and len(segments) >= 3 and (terminal_ok / len(segments)) >= 0.85:
            return lines_or_text if is_str_input else raw_lines

        reflowed = []
        cur_words = []  # list of (sec, word)

        for i, (sec, txt) in enumerate(segments):
            next_sec = segments[i + 1][0] if i + 1 < len(segments) else (sec + 4.0)
            raw_span = max(0.8, next_sec - sec)
            # Cap span by plausible speech duration so trailing music gaps don't skew word offsets
            dur = min(raw_span, max(2.5, len(txt) * 0.09 + 1.2))
            words = txt.split()
            if not words:
                continue
            weights = [
                max(1, len(w)) + (3 if w.endswith((".", "?", "!")) else (1.5 if w.endswith((",", ";", ":")) else 0))
                for w in words
            ]
            total_w = max(sum(weights), 1.0)
            cum_w = 0.0
            for j, w in enumerate(words):
                w_sec = sec + (dur * (cum_w / total_w))
                cum_w += weights[j]
                cur_words.append((w_sec, w))
                if _is_sentence_end(w) or (len(cur_words) >= 42 and w.endswith((",", ";", ":", "—", "-"))):
                    sent_start = cur_words[0][0]
                    sent_str = " ".join(x[1] for x in cur_words)
                    reflowed.append(f"{_fmt_ts(sent_start)} {sent_str}")
                    cur_words = []

        if cur_words:
            sent_start = cur_words[0][0]
            sent_str = " ".join(x[1] for x in cur_words)
            reflowed.append(f"{_fmt_ts(sent_start)} {sent_str}")

        if is_str_input:
            out_lines = header_lines + reflowed
            return "\n".join(out_lines) + ("\n" if lines_or_text.endswith("\n") else "")
        return reflowed

    @classmethod
    def build_word_aligned_sentences(cls, segments) -> list[str]:
        """
        Builds sentence-level '[MM:SS.ss] sentence' lines directly from faster-whisper
        segments with `word_timestamps=True`, anchoring each sentence's timestamp to the
        exact acoustic onset (`w.start`) of its first word.
        """
        _abbrev_no_split = {
            "mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "jr.", "sr.",
            "vs.", "etc.", "e.g.", "i.e.", "u.s.", "u.k.", "a.m.", "p.m.",
        }

        def _is_sentence_end(word: str) -> bool:
            w_clean = re.sub(r'["\'”’)\]]+$', "", word.strip())
            if not w_clean or w_clean[-1] not in (".", "?", "!"):
                return False
            low = w_clean.lower()
            if low in _abbrev_no_split:
                return False
            if re.match(r"^[a-z]\.$", low):
                return False
            return True

        lines = []
        cur_words = []  # list of (start_sec, end_sec, token_str)
        fallback_lines = []

        for seg in segments:
            seg_words = getattr(seg, "words", None)
            if not seg_words:
                fallback_lines.append(f"{cls.format_timestamp(seg.start)} {seg.text.strip()}")
                continue
            for w in seg_words:
                tok = (w.word or "").strip()
                if not tok:
                    continue
                # If there is a long silence gap (>= 1.5s) and we already have a substantial clause,
                # flush before starting the new utterance after the silence.
                if cur_words and (w.start - cur_words[-1][1]) >= 1.5 and len(cur_words) >= 6:
                    sent_start = cur_words[0][0]
                    sent_text = re.sub(r"\s+([,.;:?!])", r"\1", " ".join(x[2] for x in cur_words))
                    lines.append(f"{cls.format_timestamp(sent_start)} {sent_text}")
                    cur_words = []

                cur_words.append((w.start, w.end, tok))
                if _is_sentence_end(tok) or (len(cur_words) >= 42 and tok.endswith((",", ";", ":", "—", "-"))):
                    sent_start = cur_words[0][0]
                    sent_text = re.sub(r"\s+([,.;:?!])", r"\1", " ".join(x[2] for x in cur_words))
                    lines.append(f"{cls.format_timestamp(sent_start)} {sent_text}")
                    cur_words = []

        if cur_words:
            sent_start = cur_words[0][0]
            sent_text = re.sub(r"\s+([,.;:?!])", r"\1", " ".join(x[2] for x in cur_words))
            lines.append(f"{cls.format_timestamp(sent_start)} {sent_text}")

        if not lines and fallback_lines:
            return cls.reflow_transcript_lines(fallback_lines)
        return lines

    def fetch_youtube_transcript(self, video_id):
        """Fetches subtitles from YouTube using youtube-transcript-api."""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            raise ImportError("Please install youtube-transcript-api: pip install youtube-transcript-api")

        try:
            # Try fetching in preferred languages (compatible with different library versions)
            if hasattr(YouTubeTranscriptApi, 'list_transcripts'):
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            else:
                transcript_list = YouTubeTranscriptApi().list(video_id)

            # Try finding English, Chinese, or fallback to any auto-generated
            try:
                transcript = transcript_list.find_transcript(['zh-CN', 'zh-TW', 'en'])
            except Exception:
                transcript = transcript_list.find_generated_transcript(['en', 'zh-CN', 'zh-TW'])
            
            data = transcript.fetch()
            lines = []
            for entry in data:
                try:
                    start = entry['start']
                    text = entry['text']
                except (TypeError, KeyError):
                    start = getattr(entry, 'start', 0.0)
                    text = getattr(entry, 'text', '')
                
                time_str = self.format_timestamp(start)
                text = text.replace('\n', ' ').strip()
                lines.append(f"{time_str} {text}")
            lines = self.reflow_transcript_lines(lines)
            return "\n".join(lines)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch YouTube transcript: {e}")

    def fetch_via_ytdlp(self, url, temp_dir, cookies_from_browser=None, cookies=None):
        """Downloads and parses subtitles using yt-dlp."""
        try:
            import yt_dlp
        except ImportError:
            raise ImportError("Please install yt-dlp: pip install yt-dlp")

        ydl_opts = {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['all', '-danmaku'],
            'skip_download': True,
            'outtmpl': os.path.join(temp_dir, 'subtitle_temp.%(ext)s'),
            'conffile': False,
            'nocheckcertificate': True,
            'quiet': True,
            'no_warnings': True,
            'http_headers': self._build_ytdlp_headers(url),
        }

        if cookies_from_browser:
            ydl_opts['cookiesfrombrowser'] = (cookies_from_browser,)
        if cookies:
            ydl_opts['cookiefile'] = cookies

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=True)
                title = info.get('title', 'Video Transcript')
            except Exception as e:
                raise RuntimeError(f"yt-dlp failed to extract info: {e}")

        # Find the subtitle file downloaded
        sub_file = None
        for f in os.listdir(temp_dir):
            if f.startswith('subtitle_temp') and (f.endswith('.vtt') or f.endswith('.srt')):
                sub_file = os.path.join(temp_dir, f)
                break

        if not sub_file:
            raise RuntimeError("No subtitles found for this video.")

        # Parse vtt or srt
        with open(sub_file, 'r', encoding='utf-8') as f:
            content = f.read()

        if sub_file.endswith('.vtt'):
            lines = self.parse_vtt(content)
        else:
            lines = self.parse_srt(content)

        return title, "\n".join(lines)

    def download_video_via_ytdlp(self, url, temp_dir, cookies_from_browser=None, cookies=None):
        """
        Downloads a remote video (e.g., Bilibili, TED, or uncaptioned YouTube) as MP4 via yt-dlp,
        along with any available subtitle track.
        Returns (title, media_path, transcript_text_or_None).
        """
        try:
            import yt_dlp
        except ImportError:
            raise ImportError("Please install yt-dlp: pip install yt-dlp")

        ydl_opts = {
            'format': 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best[ext=mp4]/best',
            'merge_output_format': 'mp4',
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['all', '-danmaku'],
            'outtmpl': os.path.join(temp_dir, 'video_temp.%(ext)s'),
            'conffile': False,
            'nocheckcertificate': True,
            'quiet': True,
            'no_warnings': True,
            'http_headers': self._build_ytdlp_headers(url),
        }

        if cookies_from_browser:
            ydl_opts['cookiesfrombrowser'] = (cookies_from_browser,)
        if cookies:
            ydl_opts['cookiefile'] = cookies

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=True)
                if isinstance(info, dict) and 'entries' in info and info['entries']:
                    info = info['entries'][0]
                title = (info.get('title') if isinstance(info, dict) else None) or 'Video Transcript'
            except Exception as e:
                raise RuntimeError(f"yt-dlp failed to download video: {e}")

        media_file = None
        sub_file = None
        for f in sorted(os.listdir(temp_dir)):
            full_p = os.path.join(temp_dir, f)
            if not os.path.isfile(full_p):
                continue
            if f.startswith('video_temp') and (f.endswith('.vtt') or f.endswith('.srt')):
                if sub_file is None or '.en' in f or '.ai-en' in f:
                    sub_file = full_p
            elif f.startswith('video_temp') and f.endswith(('.mp4', '.mkv', '.webm', '.m4a', '.mp3', '.wav', '.flv')):
                if media_file is None or f.endswith('.mp4'):
                    media_file = full_p

        if not media_file:
            raise RuntimeError("yt-dlp did not produce a playable media file.")

        transcript_text = None
        if sub_file:
            try:
                with open(sub_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                lines = self.parse_vtt(content) if sub_file.endswith('.vtt') else self.parse_srt(content)
                if lines:
                    transcript_text = "\n".join(lines)
            except Exception:
                transcript_text = None

        return title, Path(media_file), transcript_text

    def parse_vtt(self, content):
        """Simple WebVTT parser."""
        lines = []
        # Match timestamp lines like "00:01:20.000 --> 00:01:23.000"
        blocks = re.split(r'\n\s*\n', content)
        for block in blocks:
            block = block.strip()
            if not block or "-->" not in block:
                continue
            parts = block.split('\n')
            time_line = parts[0]
            if "-->" not in time_line and len(parts) > 1:
                time_line = parts[1]
                text_lines = parts[2:]
            else:
                text_lines = parts[1:]

            time_match = re.search(r'(\d{2}:)?(\d{2}):(\d{2})[.,](\d{3})', time_line)
            if time_match:
                # Convert to seconds
                hrs = int(time_match.group(1).replace(':', '')) if time_match.group(1) else 0
                mins = int(time_match.group(2))
                secs = int(time_match.group(3))
                ms = int(time_match.group(4))
                total_seconds = hrs * 3600 + mins * 60 + secs + ms / 1000.0
                time_str = self.format_timestamp(total_seconds)
                text = " ".join(text_lines).strip()
                # Clean HTML tags like <c> or <i>
                text = re.sub(r'<[^>]+>', '', text)
                if text:
                    lines.append(f"{time_str} {text}")
        return self.reflow_transcript_lines(lines)

    def parse_srt(self, content):
        """Simple SRT parser."""
        lines = []
        blocks = re.split(r'\n\s*\n', content)
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            parts = block.split('\n')
            if len(parts) < 3:
                continue
            time_line = parts[1]
            text_lines = parts[2:]
            time_match = re.search(r'(\d{2}):(\d{2}):(\d{2})[.,](\d{3})', time_line)
            if time_match:
                hrs = int(time_match.group(1))
                mins = int(time_match.group(2))
                secs = int(time_match.group(3))
                ms = int(time_match.group(4))
                total_seconds = hrs * 3600 + mins * 60 + secs + ms / 1000.0
                time_str = self.format_timestamp(total_seconds)
                text = " ".join(text_lines).strip()
                if text:
                    lines.append(f"{time_str} {text}")
        return self.reflow_transcript_lines(lines)

    @staticmethod
    def _ensure_pyav_compat():
        """Patches av.open for PyAV >= 17 compatibility where metadata_errors was removed."""
        try:
            import inspect
            import av
            if getattr(av.open, "_lexis_patched", False):
                return
            sig = inspect.signature(av.open)
            if "metadata_errors" not in sig.parameters:
                orig_open = av.open

                def _compat_open(*args, **kwargs):
                    kwargs.pop("metadata_errors", None)
                    return orig_open(*args, **kwargs)

                _compat_open._lexis_patched = True
                av.open = _compat_open
        except Exception:
            pass

    @staticmethod
    def _decode_audio_via_ffmpeg(filepath, sampling_rate=16000):
        """Fallback audio decoder using ffmpeg -> 16kHz mono float32 numpy array."""
        import subprocess
        import numpy as np
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads", "0",
            "-i", str(filepath),
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", str(sampling_rate),
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, check=True)
        return np.frombuffer(proc.stdout, np.int16).flatten().astype(np.float32) / 32768.0

    def transcribe_local_file(self, filepath):
        """Transcribes local video or audio using faster-whisper."""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper is not installed. To transcribe local files, please install it:\n"
                "pip install faster-whisper"
            )

        self._ensure_pyav_compat()

        print(f"Loading Whisper model ('base')...")
        try:
            model = WhisperModel("base", device="cpu", compute_type="int8")
        except Exception:
            if not os.environ.get("HF_ENDPOINT"):
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            try:
                model = WhisperModel("base", device="cpu", compute_type="int8")
            except Exception as e:
                raise RuntimeError(f"Failed to load Whisper model: {e}")

        print(f"Transcribing file {filepath}...")
        try:
            try:
                segments, info = model.transcribe(str(filepath), beam_size=5, word_timestamps=True)
            except TypeError:
                audio_np = self._decode_audio_via_ffmpeg(filepath)
                segments, info = model.transcribe(audio_np, beam_size=5, word_timestamps=True)

            lines = self.build_word_aligned_sentences(segments)
            return info.language, "\n".join(lines)
        except IndexError as ie:
            if "tuple index out of range" in str(ie):
                raise RuntimeError(
                    f"Transcription failed: The media file may not contain a valid or compatible audio track. "
                    f"Please ensure the file has audio and can be played, or convert it to a standard format (e.g. WAV/MP3) first."
                ) from ie
            raise ie
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}") from e

    def import_video(self, url_or_path, custom_name=None, cookies_from_browser=None, cookies=None, subtitle=None, current_unit=None):
        """Imports a video transcript and saves it as a Markdown unit inside the self-contained folder structure."""
        import tempfile
        import shutil

        url_or_path = self.normalize_remote_input(url_or_path)
        yt_id = self.extract_youtube_id(url_or_path)
        remote_non_yt = self.is_remote_url(url_or_path) and not yt_id
        
        title = "Video Transcript"
        source_url = url_or_path
        video_type = "local"
        has_subtitles = True
        transcript_text = ""

        # Attempt to find or load local subtitle file
        local_sub_file = subtitle
        if local_sub_file and not os.path.exists(local_sub_file):
            # Check in all existing sources/media directories under wiki/
            if config.wiki_content_path.exists():
                for p in config.wiki_content_path.iterdir():
                    if p.is_dir() and not p.name.startswith("."):
                        sources_media = p / "sources" / "media"
                        if (sources_media / local_sub_file).exists():
                            local_sub_file = str(sources_media / local_sub_file)
                            break

        if not local_sub_file:
            # Auto-detect in current directory or any unit-specific sources/media/ if video ID matches
            search_id = yt_id
            if search_id:
                try:
                    search_dirs = [Path('.')]
                    if config.wiki_content_path.exists():
                        for p in config.wiki_content_path.iterdir():
                            if p.is_dir() and not p.name.startswith("."):
                                search_dirs.append(p / "sources" / "media")
                    
                    for search_dir in search_dirs:
                        if search_dir.exists():
                            for f in os.listdir(search_dir):
                                if search_id in f and (f.endswith('.srt') or f.endswith('.vtt')):
                                    local_sub_file = str(search_dir / f)
                                    break
                        if local_sub_file:
                            break
                except Exception:
                    pass

        if local_sub_file and os.path.exists(local_sub_file):
            print(f"Using local subtitle file: {local_sub_file}")
            try:
                with open(local_sub_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                if local_sub_file.endswith('.vtt'):
                    lines = self.parse_vtt(content)
                else:
                    lines = self.parse_srt(content)
                transcript_text = "\n".join(lines)
                
                # Try to extract title from subtitle filename
                base_name = os.path.splitext(os.path.basename(local_sub_file))[0]
                if yt_id and f"[{yt_id}]" in base_name:
                    title = base_name.split(f"[{yt_id}]")[0].strip()
                else:
                    title = base_name
                
                if not title:
                    title = f"Video {search_id or 'Transcript'}"
            except Exception as sub_err:
                print(f"Error loading local subtitle file: {sub_err}")
                local_sub_file = None

        if yt_id or remote_non_yt:
            remote_url = f"https://www.youtube.com/watch?v={yt_id}" if yt_id else url_or_path
            if yt_id:
                print(f"Detected YouTube video ID: {yt_id}. Downloading video via yt-dlp: {remote_url}")
            else:
                print(f"Downloading remote video via yt-dlp: {remote_url}")
            remote_temp_dir = tempfile.mkdtemp()
            try:
                dl_title, dl_media_path, dl_sub_text = self.download_video_via_ytdlp(
                    remote_url, remote_temp_dir, cookies_from_browser, cookies
                )
                if not local_sub_file or title == "Video Transcript":
                    title = dl_title or (f"YouTube Video {yt_id}" if yt_id else "Video Transcript")

                safe_title = custom_name or title
                safe_title = re.sub(r'[\\/*?:"<>| ]', '_', safe_title)
                safe_title = re.sub(r'_+', '_', safe_title).strip('_')
                if len(safe_title) > 40:
                    safe_title = safe_title[:40].strip('_')
                if not safe_title:
                    safe_title = "Imported_Video"

                if current_unit:
                    unit_stem = extract_unit_stem(current_unit)
                else:
                    unit_stem = safe_title

                wiki_dir = Path(config.project_root) / "wiki"
                wiki_dir.mkdir(parents=True, exist_ok=True)
                sources_dir = wiki_dir / unit_stem / "sources"
                media_dir = sources_dir / "media"
                sources_dir.mkdir(parents=True, exist_ok=True)
                media_dir.mkdir(parents=True, exist_ok=True)

                is_supplement = is_supplement_unit(unit_stem, safe_title, current_unit)
                if is_supplement:
                    target_file = media_dir / f"{safe_title}.md"
                else:
                    target_file = sources_dir / f"{unit_stem}.md"

                dest_media_path = media_dir / f"{target_file.stem}{dl_media_path.suffix}"
                shutil.move(str(dl_media_path), str(dest_media_path))
                local_path = dest_media_path
                video_type = "local"
                source_url = f"/api/media/{dest_media_path.name}"

                if not transcript_text and dl_sub_text:
                    transcript_text = dl_sub_text

                if not transcript_text and yt_id:
                    try:
                        transcript_text = self.fetch_youtube_transcript(yt_id)
                    except Exception:
                        pass

                if not transcript_text:
                    clean_title = safe_title.replace("_", " ").title()
                    placeholder_content = f"""---
title: "{title}"
video_type: "{video_type}"
source_url: "{source_url}"
category: "video_transcript"
---
# {clean_title}

⏳ Transcription is currently in progress. Please wait...
"""
                    with open(target_file, "w", encoding="utf-8") as f_p:
                        f_p.write(placeholder_content)
                    print(f"Created instant placeholder file at: {target_file}")

                    lang, transcript_text = self.transcribe_local_file(dest_media_path)
                    print(f"Transcription complete (detected language: {lang}).")
            except Exception as dl_err:
                if not yt_id:
                    raise
                print(f"YouTube video download failed ({type(dl_err).__name__}: {dl_err}). Falling back to YouTube embed mode...")
                video_type = "youtube"
                source_url = remote_url
                if not local_sub_file:
                    try:
                        transcript_text = self.fetch_youtube_transcript(yt_id)
                        title = f"YouTube Video {yt_id}"
                    except Exception as e:
                        print(f"Could not fetch native transcripts ({type(e).__name__}). Trying fallback via yt-dlp...")
                        try:
                            title, transcript_text = self.fetch_via_ytdlp(source_url, remote_temp_dir, cookies_from_browser, cookies)
                        except Exception:
                            title = f"YouTube Video {yt_id}"
                            transcript_text = "[00:00.00] (YouTube transcript could not be automatically downloaded. Please play the video on the dashboard and add subtitles manually.)"
                            has_subtitles = False
            finally:
                shutil.rmtree(remote_temp_dir, ignore_errors=True)
        else:
            # Treat as local file path
            local_path = Path(url_or_path)
            if not local_path.exists():
                # Scan across all sources/media directories in wiki/
                if config.wiki_content_path.exists():
                    for p in config.wiki_content_path.iterdir():
                        if p.is_dir() and not p.name.startswith("."):
                            sources_media = p / "sources" / "media"
                            check_path = sources_media / Path(url_or_path).name
                            if check_path.exists():
                                local_path = check_path
                                break
                
            if not local_path.exists():
                raise FileNotFoundError(f"Local media file not found: {url_or_path}")
            
            video_type = "local"
            source_url = f"/api/media/{local_path.name}"
            
            # Use the video name as the title if title hasn't been set by subtitle file yet
            if not local_sub_file or title == "Video Transcript":
                title = local_path.stem

            # Sanitize Title for file naming
            safe_title = custom_name or title
            safe_title = re.sub(r'[\\/*?:"<>| ]', '_', safe_title)
            safe_title = re.sub(r'_+', '_', safe_title).strip('_')
            if len(safe_title) > 40:
                safe_title = safe_title[:40].strip('_')

            # Determine the unit stem robustly from safe_title or current_unit using helper
            if current_unit:
                unit_stem = extract_unit_stem(current_unit)
            else:
                unit_stem = safe_title

            # Save to its own folder under wiki/ under the individual node/unit structure
            wiki_dir = Path(config.project_root) / "wiki"
            wiki_dir.mkdir(parents=True, exist_ok=True)

            # Build self-contained sources and sources/media paths
            sources_dir = wiki_dir / unit_stem / "sources"
            media_dir = sources_dir / "media"

            sources_dir.mkdir(parents=True, exist_ok=True)
            media_dir.mkdir(parents=True, exist_ok=True)

            # Determine if the transcript should be a companion/supplement using helper
            is_supplement = is_supplement_unit(unit_stem, safe_title, current_unit)

            # If it's a local video/audio, update its source URL to match the unified renamed filename
            if is_supplement:
                source_url = f"/api/media/{safe_title}{local_path.suffix}"
            else:
                source_url = f"/api/media/{unit_stem}{local_path.suffix}"

            if is_supplement:
                target_file = media_dir / f"{safe_title}.md"
            else:
                target_file = sources_dir / f"{unit_stem}.md"

            if not local_sub_file:
                # INSTANT PLACEHOLDER! Create a temporary file to show up in the library instantly while transcribing.
                clean_title = safe_title.replace("_", " ").title()
                placeholder_content = f"""---
title: "{clean_title}"
video_type: "{video_type}"
source_url: "{source_url}"
category: "video_transcript"
---
# {clean_title}

⏳ Transcription is currently in progress. Please wait...
"""
                with open(target_file, "w", encoding="utf-8") as f_p:
                    f_p.write(placeholder_content)
                print(f"Created instant placeholder file at: {target_file}")

                # Run Whisper transcription
                lang, transcript_text = self.transcribe_local_file(local_path)
                print(f"Transcription complete (detected language: {lang}).")

        # Sanitize Title for file naming (already done for local, but needed for youtube embed fallback)
        if video_type == "youtube":
            safe_title = custom_name or title
            safe_title = re.sub(r'[\\/*?:"<>| ]', '_', safe_title)
            safe_title = re.sub(r'_+', '_', safe_title).strip('_')
            if len(safe_title) > 40:
                safe_title = safe_title[:40].strip('_')
            if not safe_title:
                safe_title = "Imported_Video"

            # Determine the unit stem robustly from safe_title or current_unit using helper
            if current_unit:
                unit_stem = extract_unit_stem(current_unit)
            else:
                unit_stem = safe_title

            # Save to its own folder under wiki/ under the individual node/unit structure
            wiki_dir = Path(config.project_root) / "wiki"
            wiki_dir.mkdir(parents=True, exist_ok=True)

            # Build self-contained sources and sources/media paths
            sources_dir = wiki_dir / unit_stem / "sources"
            media_dir = sources_dir / "media"

            sources_dir.mkdir(parents=True, exist_ok=True)
            media_dir.mkdir(parents=True, exist_ok=True)

            # Determine if the transcript should be a companion/supplement using helper
            is_supplement = is_supplement_unit(unit_stem, safe_title, current_unit)

            if is_supplement:
                target_file = media_dir / f"{safe_title}.md"
            else:
                target_file = sources_dir / f"{unit_stem}.md"

        # Format as raw markdown
        output_lines = [
            "---",
            f"title: \"{title}\"",
            f"video_type: \"{video_type}\"",
            f"source_url: \"{source_url}\"",
            "category: \"video_transcript\"",
            "---",
            "",
            f"# {title}",
            "",
            transcript_text
        ]

        with open(target_file, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))

        print(f"Successfully saved transcript to: {target_file}")

        # Copy original media file to sources/media if local file, renaming it to match the target_file stem
        if video_type == "local":
            try:
                dest_media_path = media_dir / f"{target_file.stem}{local_path.suffix}"
                if local_path.resolve() != dest_media_path.resolve():
                    print(f"Copying local media file from {local_path} to {dest_media_path}...")
                    shutil.copy2(local_path, dest_media_path)
            except Exception as copy_err:
                print(f"Error copying local media file: {copy_err}")

        # Copy local/manual subtitle file if found, renaming it to match the target_file stem
        if local_sub_file and os.path.exists(local_sub_file):
            try:
                dest_sub_path = media_dir / f"{target_file.stem}{Path(local_sub_file).suffix}"
                if Path(local_sub_file).resolve() != dest_sub_path.resolve():
                    print(f"Copying subtitle file from {local_sub_file} to {dest_sub_path}...")
                    shutil.copy2(local_sub_file, dest_sub_path)
            except Exception as copy_sub_err:
                print(f"Error copying subtitle file: {copy_sub_err}")

        return str(target_file), has_subtitles

video_service = VideoService()
