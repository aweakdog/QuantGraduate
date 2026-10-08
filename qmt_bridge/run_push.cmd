@echo off
rem QMT snapshot push, called by Task Scheduler: run_push.cmd morning^|close
rem log appended to D:\qmtcode\logs\push.log
cd /d D:\qmtcode
if not exist logs mkdir logs
.venv\Scripts\python.exe -X utf8 -m qmt_bridge.push_snapshot --kind %1 >> logs\push.log 2>&1
