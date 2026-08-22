import os
import re
from pathlib import Path
from .config import config, normalize_name, extract_unit_stem, is_supplement_unit

class VideoService:
    def __init__(self):
        pass

    def extract_youtube_id(self, url):
        """Extracts the 11-character YouTube video ID from a URL."""
        pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        # If url is already just an ID
        if len(url) == 11 and re.match(r'^[0-9A-Za-z_-]+$', url):
            return url
        return None

    def format_timestamp(self, seconds):
        """Formats float seconds into [MM:SS] or [HH:MM:SS]."""
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        if hrs > 0:
            return f"[{hrs:02d}:{mins:02d}:{secs:05.2f}]"
        else:
            return f"[{mins:02d}:{secs:05.2f}]"

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
            'subtitleslangs': ['all'],
            'skip_download': True,
            'outtmpl': os.path.join(temp_dir, 'subtitle_temp.%(ext)s'),
            'conffile': False,
            'nocheckcertificate': True,
            'quiet': True,
            'no_warnings': True,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Referer': 'https://www.youtube.com/',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            }
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
        lines = []
        with open(sub_file, 'r', encoding='utf-8') as f:
            content = f.read()

        if sub_file.endswith('.vtt'):
            lines = self.parse_vtt(content)
        else:
            lines = self.parse_srt(content)

        return title, "\n".join(lines)

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
        return lines

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
        return lines

    def transcribe_local_file(self, filepath):
        """Transcribes local video or audio using faster-whisper."""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper is not installed. To transcribe local files, please install it:\n"
                "pip install faster-whisper"
            )

        print(f"Loading Whisper model ('base')...")
        try:
            model = WhisperModel("base", device="cpu", compute_type="int8")
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}")

        print(f"Transcribing file {filepath}...")
        try:
            segments, info = model.transcribe(str(filepath), beam_size=5)
            
            lines = []
            for segment in segments:
                time_str = self.format_timestamp(segment.start)
                lines.append(f"{time_str} {segment.text.strip()}")
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

        yt_id = self.extract_youtube_id(url_or_path)
        
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

        if yt_id:
            print(f"Detected YouTube video ID: {yt_id}")
            video_type = "youtube"
            source_url = f"https://www.youtube.com/watch?v={yt_id}"
            if not local_sub_file:
                try:
                    transcript_text = self.fetch_youtube_transcript(yt_id)
                    title = f"YouTube Video {yt_id}"
                except Exception as e:
                    print(f"Could not fetch native transcripts ({type(e).__name__}). Trying fallback via yt-dlp...")
                    temp_dir = tempfile.mkdtemp()
                    try:
                        title, transcript_text = self.fetch_via_ytdlp(source_url, temp_dir, cookies_from_browser, cookies)
                    except Exception as ytdlp_err:
                        print(f"yt-dlp fallback failed ({type(ytdlp_err).__name__})")
                        title = f"YouTube Video {yt_id}"
                        transcript_text = "[00:00.00] (YouTube transcript could not be automatically downloaded. Please play the video on the dashboard and add subtitles manually.)"
                        has_subtitles = False
                    finally:
                        shutil.rmtree(temp_dir)
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

        # Sanitize Title for file naming (already done for local, but needed for youtube)
        if yt_id:
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
