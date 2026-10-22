@echo off
rem Ensure QMT trading account is online; called by Task Scheduler (09:10 --force, 13:00 on demand)
cd /d D:\qmtcode
if not exist logs mkdir logs
.venv\Scripts\python.exe -X utf8 -m qmt_bridge.ensure_qmt %* >> logs\ensure_qmt.log 2>&1
