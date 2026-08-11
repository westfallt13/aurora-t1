@echo off
cd /d "%~dp0cloud"
echo. > STOP
echo Stop signal sent. Training will save a checkpoint and exit
echo within the next few seconds (check the training window).
pause
