:: ============================================================
::   Lexis One-Click Setup
:: ------------------------------------------------------------
::   Double-click this file (or run install.cmd in a terminal)
::   to install everything automatically:
::     pip install -e .              lexis (editable install)
::     pip install pyinstaller       standalone EXE export
::               pytest              test suite
::     lexis init .                  initialize the project (safe
::                                   to re-run)
::   Then it ASKS whether to also install the local video/audio
::   transcription extra (faster-whisper) - a large download, so
::   it is NOT installed by default. Answer "y" to install it now,
::   or add it later with:  pip install faster-whisper
::   Dependency versions are all declared in pyproject.toml
::   (single source of truth).
::   NOTE: avoid unescaped parentheses inside the if (...) blocks
::   below - cmd treats them as block terminators.
::   After setup, just run `lexis ...` (or double-click
::   start_dashboard.vbs to open the web console).
:: ============================================================
@echo off
setlocal
cd /d "%~dp0"

:: ---- Resolve the Python interpreter: prefer the `py` launcher
:: ---- (it survives the Microsoft Store python-alias issue), and
:: ---- fall back to plain `python` on PATH.
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo [ERROR] Python not found. Please install Python 3.8+ and check "Add Python to PATH".
    pause
    exit /b 1
)

:: ---- 1/2 Install lexis (editable) + small dev tools
echo [1/2] Installing lexis and dev tools - pyinstaller, pytest ...
%PY% -m pip install -e . || goto :fail
%PY% -m pip install pyinstaller pytest || goto :fail

:: ---- Optional: local video/audio transcription.
:: ---- faster-whisper pulls a large dependency tree, so it is
:: ---- opt-in: default answer is N (just press Enter).
echo.
set "EXTRA="
set /p EXTRA="Install local video transcription  faster-whisper  - large download? [y/N] "
if /i not "%EXTRA%"=="y" goto :no_extra
%PY% -m pip install faster-whisper || goto :fail
echo   faster-whisper installed.
goto :extra_done
:no_extra
echo   Skipped. You can add it later with:  %PY% -m pip install faster-whisper
:extra_done

:: ---- 2/2 Initialize the project (safe to re-run)
echo [2/2] Initializing project (lexis init .) ...
%PY% -m librarian.cli init . || goto :fail

echo.
echo ============================================
echo   Setup complete! You can now run:
echo     lexis compile "D:\My Lessons\Book_4_Unit_1.md"
echo   or double-click  start_dashboard.vbs  to open the web console
echo ============================================
pause
exit /b 0

:fail
echo.
echo [ERROR] Setup failed. Please check the error messages above.
pause
exit /b 1