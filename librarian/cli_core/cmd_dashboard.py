"""
CLI subparser and handler for the interactive web dashboard.
"""

import argparse
import datetime
import io
import logging
import os
import re
import signal
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

from ..config import config
from ..dashboard import DashboardHTTPRequestHandler, DashboardState


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    dashboard_parser = subparsers.add_parser("dashboard", help="Start the interactive local dashboard server")
    dashboard_parser.add_argument("--port", type=int, default=None, help="Port to run the dashboard server on (default: from wiki_config.json or 8000)")
    dashboard_parser.add_argument("--host", type=str, default=None, help="Host/IP to bind the dashboard server on (default: from wiki_config.json or localhost)")
    dashboard_parser.set_defaults(handler=handle_dashboard)


class _Tee:
    """File-like stream that writes to both a console stream (if any)
    and the log file with ISO timestamps. Each sink is independent: failure on one
    (e.g. console closed) never breaks the other."""
    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = log_file
        self._at_line_start = True

    def write(self, data):
        # Write to console as-is
        if self._stream is not None:
            try:
                self._stream.write(data)
                self._stream.flush()
            except Exception:
                pass

        # Write to log file with ISO timestamp prefix on each new line
        if self._log is not None:
            try:
                lines = data.split("\n")
                for idx, line in enumerate(lines):
                    if idx > 0:
                        self._log.write("\n")
                        self._at_line_start = True
                    if line:
                        if self._at_line_start:
                            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            self._log.write(f"[{ts}] ")
                            self._at_line_start = False
                        self._log.write(line)
                self._log.flush()
            except Exception:
                pass
        return len(data)

    def flush(self):
        for target in (self._stream, self._log):
            if target is not None:
                try:
                    target.flush()
                except Exception:
                    pass

    def isatty(self):
        return bool(self._stream is not None and self._stream.isatty())


def _free_port(target_port: int) -> None:
    current_pid = os.getpid()
    if sys.platform == "win32":
        try:
            raw_out = subprocess.check_output("netstat -ano -p tcp", shell=True, stderr=subprocess.DEVNULL)
            try:
                out = raw_out.decode("utf-8", errors="ignore")
            except Exception:
                out = raw_out.decode("gbk", errors="ignore")
            for line in out.splitlines():
                line = line.strip()
                upper_line = line.upper()
                # Strictly target sockets in LISTENING / 监听 state
                # Avoid misidentifying client-side outbound ports or transient TIME_WAIT states
                is_listening = ("LISTENING" in upper_line or "监听" in line)
                if is_listening and (f":{target_port} " in line or f":{target_port}\t" in line):
                    parts = line.split()
                    try:
                        pid = int(parts[-1])
                        if pid and pid != current_pid and pid != 0:
                            print(f"⚠️ Port {target_port} is occupied by an older process (PID {pid}). Terminating old instance...")
                            subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
                            time.sleep(1.0)
                    except (ValueError, IndexError):
                        pass
        except Exception:
            pass
    else:
        pids_to_kill = set()
        for cmd, pat in (
            (f"lsof -ti tcp:{target_port}", r"\b(\d+)\b"),
            (f"ss -lptn 'sport = :{target_port}'", r"pid=(\d+)"),
            (f"fuser {target_port}/tcp", r"\b(\d+)\b"),
        ):
            try:
                out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
                for m in re.findall(pat, out):
                    pids_to_kill.add(int(m))
                if pids_to_kill:
                    break
            except Exception:
                continue
        for pid in pids_to_kill:
            if pid and pid != current_pid:
                try:
                    print(f"⚠️ Port {target_port} is occupied by an older process (PID {pid}). Terminating old instance...")
                    os.kill(pid, signal.SIGKILL)
                    time.sleep(0.5)
                except (ProcessLookupError, PermissionError):
                    pass


class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    # Handler threads must be daemons: keep-alive connections otherwise block
    # the interpreter from exiting (threading._shutdown waits for non-daemon
    # threads stuck in readline()), leaving a zombie process after shutdown.
    daemon_threads = True


def _try_bind_port(target_port: int) -> ThreadingHTTPServer:
    for attempt in range(3):
        try:
            return ThreadingHTTPServer(("", target_port), DashboardHTTPRequestHandler)
        except OSError as bind_err:
            # WinError 10013 is permission/excluded port - cleanup won't help, don't waste time retrying 3 times
            is_permission_err = getattr(bind_err, "winerror", None) == 10013 or "10013" in str(bind_err)
            if not is_permission_err and attempt < 2:
                print(f"Port {target_port} busy, retrying cleanup (attempt {attempt+1}/3)...")
                _free_port(target_port)
                time.sleep(0.8)
            else:
                raise bind_err


def handle_dashboard(args: argparse.Namespace) -> None:
    import webbrowser

    cfg_dash_url = config.get("dashboard_url") or config.get("server_url") or ""
    parsed_dash = urlparse(cfg_dash_url) if cfg_dash_url else None
    cfg_host = config.get("dashboard_host") or (parsed_dash.hostname if parsed_dash and parsed_dash.hostname else None) or "localhost"
    cfg_port = config.get("dashboard_port") or (parsed_dash.port if parsed_dash and parsed_dash.port else None) or 8000
    port = int(args.port if args.port is not None else cfg_port)
    display_host = args.host if args.host is not None else cfg_host

    # Dual-output logging: append dashboard output to logs/dashboard.log with ISO timestamps and size rotation;
    # when run in a terminal, also mirror it to the console. Under pythonw
    # (VBS launcher) stdout/stderr are None, so the log file is the only sink.
    try:
        logs_dir = Path(config.project_root) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / "dashboard.log"

        # Log rotation: rotate if dashboard.log exceeds 5MB, keep up to 3 backups
        MAX_LOG_BYTES = 5 * 1024 * 1024
        BACKUP_COUNT = 3
        if log_path.exists() and log_path.stat().st_size >= MAX_LOG_BYTES:
            try:
                for i in range(BACKUP_COUNT - 1, 0, -1):
                    s = logs_dir / f"dashboard.log.{i}"
                    d = logs_dir / f"dashboard.log.{i+1}"
                    if s.exists():
                        if d.exists():
                            d.unlink()
                        s.rename(d)
                b1 = logs_dir / "dashboard.log.1"
                if b1.exists():
                    b1.unlink()
                log_path.rename(b1)
            except Exception:
                pass

        _log_file = open(log_path, "a", encoding="utf-8", buffering=1)

        sys.stdout = _Tee(sys.stdout, _log_file)
        sys.stderr = _Tee(sys.stderr, _log_file)

        # Ensure python logging library outputs with level and logger name (_Tee prefixes the timestamp)
        logging.basicConfig(
            format="[%(levelname)s] [%(name)s] %(message)s",
            level=logging.INFO,
            force=True,
        )
    except Exception:
        # Last-resort safety for headless mode: a print() must never crash
        # the server if the log file cannot be opened.
        _null_stream = io.StringIO()
        if sys.stdout is None:
            sys.stdout = _null_stream
        if sys.stderr is None:
            sys.stderr = _null_stream

    # 1. Clean up any existing dashboard instance or process occupying the port
    _free_port(port)

    print(f"Starting Lexis Interactive Dashboard on port {port}...")

    server_created = False
    original_port = port
    httpd = None

    try:
        httpd = _try_bind_port(port)
        server_created = True
    except OSError as bind_err:
        # Fallback to alternate ports if default/specified port is occupied or blocked by Windows excluded ports
        print(f"⚠️ Port {port} is unavailable ({bind_err}). Probing alternative ports...")
        fallback_ports = [8888, 8089, 8090, 8008, 8800, 5000, 3000]
        for candidate_port in fallback_ports:
            if candidate_port == original_port:
                continue
            try:
                _free_port(candidate_port)
                httpd = _try_bind_port(candidate_port)
                port = candidate_port
                server_created = True
                print(f"✅ Successfully bound to alternative port {port}.")
                break
            except OSError:
                continue

        if not server_created:
            raise bind_err

    with httpd:
        httpd.timeout = 0.5  # Check for KeyboardInterrupt every 0.5s on Windows
        url = f"http://{display_host}:{port}"
        print(f"Dashboard server is running at: {url}")
        print(f"LLM API endpoint (wiki_config.json): {config.get('api_url')} (model: {config.get('model')})")
        print(f"TTS Engine (native Python): {config.get('tts_engine')} (Female: {config.get('tts_voice_a')}, Male: {config.get('tts_voice_b')})")
        print("Closing the browser window will automatically stop this server and free the port.")
        print("(Ctrl+C also works.)")

        # Detect WSL so we can launch the Windows host browser and give a generous connect window
        is_wsl = bool(os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"))
        if not is_wsl and os.name != "nt":
            try:
                with open("/proc/version", "r", encoding="utf-8", errors="ignore") as _vf:
                    is_wsl = "microsoft" in _vf.read().lower()
            except Exception:
                pass

        # Open browser automatically in a separate thread
        # (skipped when LEXIS_NO_BROWSER is set - e.g. automated tests)
        def open_browser():
            import shutil as _shutil
            if os.environ.get("LEXIS_NO_BROWSER"):
                return
            if is_wsl:
                for wsl_cmd in (
                    (["wslview", url] if _shutil.which("wslview") else None),
                    (["cmd.exe", "/c", "start", "", url] if _shutil.which("cmd.exe") else None),
                    (["powershell.exe", "-NoProfile", "-Command", f"Start-Process '{url}'"] if _shutil.which("powershell.exe") else None),
                ):
                    if not wsl_cmd:
                        continue
                    try:
                        res = subprocess.run(wsl_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8)
                        if res.returncode == 0:
                            return
                    except Exception:
                        continue
            try:
                webbrowser.open(url)
            except Exception:
                pass

        threading.Thread(target=open_browser, daemon=True).start()

        # Auto-shutdown only on explicit tab close (/api/shutdown beacon) or Ctrl+C:
        #  - Beacon:    /api/shutdown fired by the page's beforeunload handler. It is
        #               ARMED, not immediate: an F5 refresh also fires the beacon, but
        #               the browser reconnects within ~1s; the first new request
        #               (touch()) cancels the pending shutdown and the page reloads
        #               normally. If nothing comes back within SHUTDOWN_GRACE the
        #               client truly left, and the server exits.
        #  - No Idle Timeout: Server stays alive while tabs remain open (even when
        #               browsers throttle background tab timers).
        #  - Safety:    never kill in-progress background jobs (compile / re-audit);
        #               the server stays up until they finish, then exits automatically.
        SHUTDOWN_GRACE = 45        # seconds after the beacon to wait for a reconnect (covers slow refresh / tab transitions)

        def _should_exit():
            shutdown_at, _, has_running = DashboardState.lifecycle()
            if has_running:
                return False  # hold until running background jobs complete
            if shutdown_at is not None:
                # Grace period: if the client reconnects (F5 refresh), touch() has
                # already cleared shutdown_at and we stay up.
                return (time.time() - shutdown_at) > SHUTDOWN_GRACE
            return False

        # Poll with a socket-level timeout: handle_request() then returns
        # after 0.2s of silence instead of blocking forever, so the exit
        # checks above (beacon grace period, idle timeout) stay responsive
        # and Ctrl+C is handled promptly.
        httpd.timeout = 0.2
        try:
            while not _should_exit():
                httpd.handle_request()
        except KeyboardInterrupt:
            print("\nDashboard server stopped (Ctrl+C).")
            sys.exit(0)

        # If background jobs are still running (browser closed mid-compile), finish them first
        _, _, has_running = DashboardState.lifecycle()
        if has_running:
            print("\nBrowser closed; background job(s) still running — finishing before exit...")
            try:
                while True:
                    _, _, still_running = DashboardState.lifecycle()
                    if not still_running:
                        break
                    httpd.handle_request()
            except KeyboardInterrupt:
                pass

        print("\nDashboard server stopped — port released.")
