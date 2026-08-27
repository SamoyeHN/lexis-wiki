import requests
import os
import json
from pathlib import Path
from .config import config

class TTSService:
    def __init__(self):
        # Refresh config dynamically to ensure we have the latest URL and voices
        self._refresh()

    def _refresh(self):
        self.tts_engine = config.get("tts_engine") or "kokoro"
        self.model = config.get("tts_model") or "tts-1"
        self.api_key = config.get("tts_api_key") or "any_string"

        engine = self.tts_engine.lower()
        if engine == "kokoro":
            valid_voices = ["af_sarah", "af_bella", "af_nicole", "af_sky", "am_adam", "am_michael", "bf_emma", "bf_isabella", "bm_george", "bm_lewis"]
            default_a = "af_sarah"
            default_b = "am_michael"
            default_url = "http://localhost:8880/v1/audio/speech"
        elif engine == "edge":
            valid_voices = ["en-US-AriaNeural", "en-US-JennyNeural", "en-US-GuyNeural", "en-US-ChristopherNeural", "en-GB-SoniaNeural", "en-GB-LibbyNeural", "en-GB-RyanNeural", "en-GB-ThomasNeural"]
            default_a = "en-US-AriaNeural"
            default_b = "en-GB-RyanNeural"
            default_url = "http://localhost:5050/v1/audio/speech"
        else:
            valid_voices = []
            default_a = "af_sarah"
            default_b = "am_michael"
            default_url = "http://localhost:8880/v1/audio/speech"

        # Prevent invalid voice names from being saved and self-heal the config
        voice_a = config.get("tts_voice_a")
        if valid_voices and voice_a not in valid_voices:
            self.voice_a = default_a
            config.update_config("tts_voice_a", default_a)
        else:
            self.voice_a = voice_a or default_a

        voice_b = config.get("tts_voice_b")
        if valid_voices and voice_b not in valid_voices:
            self.voice_b = default_b
            config.update_config("tts_voice_b", default_b)
        else:
            self.voice_b = voice_b or default_b

        # Retrieve user configured url, fallback to standard engine localhost default
        self.api_url = config.get("tts_url") or default_url

    def _fetch_audio_binary(self, text, voice):
        """Fetches raw audio binary from the TTS API with strict voice validation."""
        engine = getattr(self, "tts_engine", "kokoro").lower()
        
        # Verify voice against exact engine lists, fallback immediately to default if invalid
        if engine == "kokoro":
            valid_voices = ["af_sarah", "af_bella", "af_nicole", "af_sky", "am_adam", "am_michael", "bf_emma", "bf_isabella", "bm_george", "bm_lewis"]
            sanitized_voice = voice if voice in valid_voices else self.voice_a
        elif engine == "edge":
            valid_voices = ["en-US-AriaNeural", "en-US-JennyNeural", "en-US-GuyNeural", "en-US-ChristopherNeural", "en-GB-SoniaNeural", "en-GB-LibbyNeural", "en-GB-RyanNeural", "en-GB-ThomasNeural"]
            sanitized_voice = voice if voice in valid_voices else self.voice_a
        else:
            sanitized_voice = voice or self.voice_a

        payload = {
            "model": self.model,
            "input": text,
            "voice": sanitized_voice,
            "response_format": "mp3",
            "speed": 1.0
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"    - ! TTS Fetch Error for voice '{voice}' (mapped to '{sanitized_voice}') via {engine}: {e}")
            return None

    def generate_audio(self, text, output_path, voice=None):
        """Legacy support for single-voice generation if needed."""
        self._refresh()
        content = self._fetch_audio_binary(text, voice or self.voice_a)
        if content:
            with open(output_path, "wb") as f:
                f.write(content)
            return True
        return False

    def _get_gender(self, voice_name):
        v = str(voice_name or "").lower()
        if any(v.startswith(p) for p in ["af_", "bf_"]): return "female"
        if any(v.startswith(p) for p in ["am_", "bm_"]): return "male"
        if any(x in v for x in ["aria", "jenny", "sonia", "libby"]): return "female"
        if any(x in v for x in ["guy", "christopher", "ryan", "thomas"]): return "male"
        return "female"

    def process_script(self, script, output_dir=None, file_prefix=None, speaker_1=None, speaker_2=None, speaker_1_gender=None, speaker_2_gender=None, return_binary=False):
        """
        Processes a full dialogue script with dual-speaker support.
        Concatenates turns into a single MP3 binary or file.
        """
        self._refresh()
        
        combined_audio = b""
        print(f"    - Generating dual-speaker audio for {len(script)} turns...")

        voice_a_gender = self._get_gender(self.voice_a)
        voice_b_gender = self._get_gender(self.voice_b)

        # Dynamically map speakers to voices based on gender matching
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

        # Detect any additional/unmapped speakers from the script
        unique_speakers = []
        for turn in script:
            s = turn.get("speaker", "Unknown")
            if s not in unique_speakers:
                unique_speakers.append(s)
        
        # Fill map for remaining speakers if not already explicitly mapped
        voice_slots = [self.voice_a, self.voice_b]
        voice_idx = 0
        for s in unique_speakers:
            if s not in speaker_map:
                speaker_map[s] = voice_slots[voice_idx % 2]
                voice_idx += 1

        for i, turn in enumerate(script):
            speaker = turn.get("speaker", "Unknown")
            text = turn.get("text", "")
            
            # Use mapped voice, fallback to voice_a
            voice = speaker_map.get(speaker, self.voice_a)
            
            print(f"      [Turn {i+1}] {speaker}: {text[:30]}...")
            turn_audio = self._fetch_audio_binary(text, voice)
            
            if turn_audio:
                combined_audio += turn_audio
            else:
                print(f"      ! Failed to generate audio for turn {i+1}")

        if combined_audio:
            if return_binary:
                return combined_audio

            if output_dir and file_prefix:
                audio_filename = f"{file_prefix}_dialogue.mp3"
                audio_path = output_dir / audio_filename
                with open(audio_path, "wb") as f:
                    f.write(combined_audio)
                print(f"    - Success: {audio_filename} saved.")
                return audio_filename
        
        return None

tts_service = TTSService()
