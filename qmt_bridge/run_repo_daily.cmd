@echo off
rem Daily GC001 reverse repo (14:40). Live only if D:\qmtcode\REPO_ENABLED exists. Usage: run_repo_daily.cmd [dry]
cd /d D:\qmtcode
if not exist logs mkdir logs
set DRY=
if "%1"=="dry" set DRY=--dry-run
.venv\Scripts\python.exe -X utf8 -m qmt_bridge.repo_daily %DRY% >> logs\repo_daily.log 2>&1
