@echo off
set PYTHONPATH=D:\myAI\WorkBuddy-workspace\quant-strategy
set QUANT_DATA_DIR=D:\myAI\WorkBuddy-workspace\quant-strategy\data
set QUANT_MODE=%1
if "%QUANT_MODE%"=="" set QUANT_MODE=live

C:\Users\admin\.workbuddy\binaries\python\envs\quant\Scripts\python.exe -m pipeline.xgb_scorer
