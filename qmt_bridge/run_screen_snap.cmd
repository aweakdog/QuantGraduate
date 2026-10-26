@echo off
rem Screen-read QMT account snapshot and push to 041. Usage: run_screen_snap.cmd morning|close|manual [dry]
cd /d D:\qmtcode
if not exist logs mkdir logs
set DRY=
if "%2"=="dry" set DRY=--dry-run
.venv\Scripts\python.exe -X utf8 -m qmt_bridge.screen_snapshot --kind %1 %DRY% >> logs\screen_snapshot.log 2>&1
