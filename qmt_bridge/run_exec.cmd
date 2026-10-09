@echo off
rem QMT executor, called by Task Scheduler at 14:53:30 (shadow unless both keys are on)
cd /d D:\qmtcode
if not exist logs mkdir logs
.venv\Scripts\python.exe -X utf8 -m qmt_bridge.executor --auto >> logs\exec.log 2>&1
