@echo off
rem QMT 快照推送 (计划任务调用): run_push.cmd morning|close
rem 日志追加到 D:\qmtcode\logs\push.log
chcp 65001 >NUL
cd /d D:\qmtcode
if not exist logs mkdir logs
.venv\Scripts\python.exe -X utf8 -m qmt_bridge.push_snapshot --kind %1 >> logs\push.log 2>&1
