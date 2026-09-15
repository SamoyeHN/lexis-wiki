' ============================================================
'  Lexis Dashboard Launcher (headless - no console window)
' ------------------------------------------------------------
'  Double-click this file (or a shortcut to it):
'    - starts the dashboard server without any visible window
'    - opens http://localhost:8000 in your default browser
'  Closing the browser window automatically stops the server
'  and frees the port.
'  Log: logs/dashboard.log (in this folder)
' ============================================================
Option Explicit

Dim fso, sh, cmd, w, first
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh  = CreateObject("WScript.Shell")

' Always run from the folder containing this script (project root)
sh.CurrentDirectory = fso.GetParentFolderName(WScript.ScriptFullName)

' Pick the interpreter:
'   - prefer pythonw.exe (ships with every standard CPython / Anaconda /
'     MS Store install; has no console at all), located by full path so
'     PATH quirks do not matter;
'   - fall back to plain python.exe - the hidden launch flag (0) below
'     still keeps any console window invisible.
cmd = "python -m librarian.cli dashboard"
On Error Resume Next
Set w = sh.Exec("where.exe pythonw")
If Err.Number = 0 Then
    first = Trim(w.StdOut.ReadAll)
    If InStr(first, "pythonw") > 0 Then
        cmd = """" & Split(first, vbNewLine)(0) & """ -m librarian.cli dashboard"
    End If
End If
On Error GoTo 0

' 0 = hidden window, False = do not wait for it to exit
sh.Run cmd, 0, False