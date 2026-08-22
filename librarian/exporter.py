import os
import sys
import re
import shutil
import tempfile
import subprocess
from pathlib import Path
from .config import config

LAUNCHER_TEMPLATE = """# -*- coding: utf-8 -*-
import os
import sys
import threading
import http.server
import socketserver
import webbrowser
import socket
import mimetypes

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

class RangeHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args, **kwargs):
        pass # Silence verbose terminal logging

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
            
        ctype = self.guess_type(path)
        try:
            f = open(path, 'rb')
        except OSError:
            self.send_error(404, "File not found")
            return None
            
        range_header = self.headers.get('Range')
        if range_header and range_header.startswith('bytes='):
            try:
                size = os.path.getsize(path)
                ranges = range_header.replace('bytes=', '').split('-')
                start = int(ranges[0]) if ranges[0] else 0
                end = int(ranges[1]) if ranges[1] else size - 1
                if start >= size:
                    self.send_error(416, "Requested range not satisfiable")
                    f.close()
                    return None
                if end >= size:
                    end = size - 1
                length = end - start + 1
                
                self.send_response(206)
                self.send_header('Content-type', ctype)
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
                self.send_header('Content-Length', str(length))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                self._range_start = start
                self._range_end = end
                return f
            except Exception:
                f.seek(0)
                if hasattr(self, '_range_start'):
                    del self._range_start
                if hasattr(self, '_range_end'):
                    del self._range_end
                
        return super().send_head()

    def copyfile(self, source, outputfile):
        if hasattr(self, '_range_start') and hasattr(self, '_range_end'):
            source.seek(self._range_start)
            remaining = self._range_end - self._range_start + 1
            buffer_size = 64 * 1024
            while remaining > 0:
                chunk_size = min(buffer_size, remaining)
                data = source.read(chunk_size)
                if not data:
                    break
                outputfile.write(data)
                remaining -= len(data)
        else:
            super().copyfile(source, outputfile)

def main():
    # Get the directory where PyInstaller unpacked the files
    base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    
    port = find_free_port()
    os.chdir(base_dir)
    
    class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        allow_reuse_address = True
        
    handler = RangeHTTPRequestHandler
    
    httpd = ThreadingHTTPServer(("", port), handler)
    url = f"http://localhost:{port}/index.html"
    
    print("=" * 60)
    print("               LEXIS STANDALONE HANDOUT VIEWER")
    print("=" * 60)
    print(f" * Serving Interactive Quiz on port {port}...")
    print(f" * Opening default browser: {url}")
    print(" * To close, simply close this console window.")
    print("=" * 60)
    
    def open_browser():
        try:
            webbrowser.open(url)
        except Exception:
            pass
            
    threading.Thread(target=open_browser, daemon=True).start()
    
    try:
        httpd.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        print("\\nServer shutting down.")

if __name__ == '__main__':
    main()
"""

def detect_video_path_from_html(html_content, html_path):
    """
    Scans the HTML file content or scans unit-specific sources/media to find referenced media files.
    """
    html_path = Path(html_path)
    # Check unit-specific media folder first!
    unit_dir = html_path.parent.parent
    media_dir = unit_dir / "sources" / "media"
    if media_dir.exists() and media_dir.is_dir():
        for f in media_dir.iterdir():
            if f.is_file() and f.suffix in [".mp4", ".mp3", ".mkv", ".avi", ".mov"]:
                return f

    # Fallback: scan references in HTML
    pattern = r'["\']/?(?:api/media|raw/media)/([^"\'\?#]+)["\']'
    matches = re.findall(pattern, html_content)
    if matches:
        filename = matches[0]
        if media_dir.exists() and media_dir.is_dir():
            full_path = media_dir / filename
            if full_path.exists():
                return full_path
                
    # Fallback to search any mp4/mp3 inside raw/media/ just in case
    html_stem = Path(html_path).stem.lower().replace("_quiz", "").replace("_handout", "")
    legacy_media_dir = Path(config.project_root) / "raw" / "media"
    if legacy_media_dir.exists():
        for f in legacy_media_dir.glob("*.*"):
            if html_stem in f.name.lower() or f.stem.lower() in html_stem:
                return f
                
    return None

def export_standalone_exe(html_path, video_path=None, output_path=None):
    """
    Bundles an HTML quiz file and its associated media file into a single standalone EXE file using PyInstaller.
    """
    # Resolve virtual URL paths (e.g., starting with /wiki/ or wiki/) to physical project paths
    path_str = str(html_path).replace("\\", "/")
    if "wiki/" in path_str:
        relative_part = path_str.split("wiki/", 1)[1]
        resolved_path = Path(config.project_root) / "wiki" / relative_part
        if resolved_path.exists():
            html_path = resolved_path
        else:
            html_path = Path(html_path)
    else:
        if path_str.startswith("/"):
            path_str = path_str[1:]
        resolved_path = Path(config.project_root) / path_str
        if resolved_path.exists():
            html_path = resolved_path
        else:
            html_path = Path(html_path)

    if not html_path.exists():
        # Search inside wiki/handouts/
        handouts_dir = Path(config.project_root) / "wiki" / "handouts"
        if handouts_dir.exists():
            # 1. Exact match on filename
            exact_match = handouts_dir / html_path.name
            if exact_match.exists():
                html_path = exact_match
                print(f"Detected handout inside wiki/handouts: {html_path.name}")
            else:
                # 2. Fuzzy search (case-insensitive substring match)
                matches = []
                search_stem = html_path.stem.lower()
                for f in handouts_dir.glob("*.html"):
                    if search_stem in f.name.lower():
                        matches.append(f)
                
                if len(matches) == 1:
                    html_path = matches[0]
                    print(f"Fuzzy-matched handout in wiki/handouts: {html_path.name}")
                elif len(matches) > 1:
                    print(f"Multiple matches found in wiki/handouts for '{html_path.name}':")
                    for idx, m in enumerate(matches):
                        print(f"  [{idx + 1}] {m.name}")
                    html_path = matches[0]
                    print(f"Defaulting to first match: {html_path.name}")
                else:
                    print(f"Error: HTML handout '{html_path}' not found directly or fuzzy-matched in {handouts_dir}")
                    return False
        else:
            print(f"Error: HTML handout not found at '{html_path}' and handouts directory {handouts_dir} does not exist.")
            return False

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # 1. Resolve video path
    resolved_video = None
    if video_path:
        resolved_video = Path(video_path)
        if not resolved_video.exists():
            # Try finding in raw/media/
            resolved_video = Path(config.project_root) / "raw" / "media" / video_path
            if not resolved_video.exists():
                print(f"Warning: Specified video path '{video_path}' not found. Attempting auto-detection.")
                resolved_video = None

    if not resolved_video:
        resolved_video = detect_video_path_from_html(html_content, html_path)

    if resolved_video:
        print(f"Found associated media file: {resolved_video.name} ({resolved_video.stat().st_size / 1024 / 1024:.2f} MB)")
    else:
        print("Note: No local video/audio file detected for this handout. Packing HTML only.")

    # 2. Check for PyInstaller dependency
    try:
        subprocess.run([sys.executable, "-m", "PyInstaller", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception:
        print("\nError: PyInstaller is required to export standalone EXEs.")
        print("Please install it on your computer by running:")
        print("  pip install pyinstaller")
        return False

    # 3. Create a temporary build workspace
    temp_dir = Path(tempfile.mkdtemp(prefix="lexis_build_"))
    try:
        # Copy HTML to index.html in build workspace
        shutil.copy2(html_path, temp_dir / "index.html")

        # Write the launcher script
        launcher_file = temp_dir / "launcher.py"
        with open(launcher_file, "w", encoding="utf-8") as f:
            f.write(LAUNCHER_TEMPLATE)

        # Set output file name and directory
        html_stem = html_path.stem
        exe_name = f"{html_stem}_standalone"
        
        if output_path:
            target_output_dir = Path(output_path).parent
            final_exe_name = Path(output_path).name
        else:
            # Save inside the unit's actual local handouts directory
            target_output_dir = html_path.parent
            final_exe_name = f"{exe_name}.exe"
            
        target_output_dir.mkdir(parents=True, exist_ok=True)
        final_exe_path = target_output_dir / final_exe_name

        print(f"Building single-file standalone EXE...")
        print("This may take up to a minute depending on the video file size...")

        # Setup PyInstaller arguments
        cmd = [
            sys.executable,
            "-m",
            "PyInstaller",
            "--onefile",
            "--name", exe_name,
            f"--add-data={temp_dir / 'index.html'};.",
            "--distpath", str(target_output_dir),
            "--workpath", str(temp_dir / "build"),
            "--specpath", str(temp_dir),
            "--clean"
        ]

        # Add the media file to the virtual raw/media/ directory inside the EXE
        if resolved_video:
            # Under Windows, PyInstaller --add-data syntax is: "source_path;destination_subfolder"
            cmd.append(f"--add-data={resolved_video};raw/media")

        # Point to launcher entry point
        cmd.append(str(launcher_file))

        # Run compilation
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        if result.returncode == 0:
            # Rename if custom output path is provided
            default_built_exe = target_output_dir / f"{exe_name}.exe"
            if default_built_exe.exists() and default_built_exe.resolve() != final_exe_path.resolve():
                if final_exe_path.exists():
                    os.remove(final_exe_path)
                shutil.move(default_built_exe, final_exe_path)

            print("\n" + "="*50)
            print(" SUCCESS: Standalone EXE package built successfully!")
            print(f" Standalone File: {final_exe_path}")
            print("="*50 + "\n")
            print("Share this single file with anyone! They can double-click it to run the quiz and stream the video.")
            return final_exe_path
        else:
            print("\nError during PyInstaller packaging:")
            print(result.stdout)
            return False

    finally:
        # Clean up temporary build workspace
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass
