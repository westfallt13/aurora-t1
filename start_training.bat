@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

echo ===============================================
echo   Aurora LLM - Local Training
echo ===============================================
echo   Checkpoints only save on a CLEAN stop now (no
echo   periodic checkpoint during training). To stop
echo   safely: click this window, then press Ctrl+C,
echo   and wait for it to finish saving.
echo.
echo   Do NOT just close this window with the X button,
echo   and avoid crashes/reboots while this is running
echo   -- anything other than a clean stop now loses ALL
echo   progress back to the last checkpoint, not just
echo   the recent bit.
echo.
echo   To resume later: just run this script again.
echo   It automatically picks up from the last checkpoint.
echo.
echo   On resume, it downloads the checkpoint from S3 before
echo   anything else -- at this model size that's ~1.3GB and
echo   can take several minutes (observed 3.5-11 min,
echo   network-dependent). Watch for "[resume] downloading
echo   checkpoint: ..." progress lines -- if you see those
echo   ticking up, it's working normally, not hung.
echo ===============================================
echo.

cd cloud
"%~dp0.venv\Scripts\python.exe" train.py ^
  --s3-bucket aurora-llm-checkpoints-752988091124 ^
  --run-name local-110m ^
  --tokenizer-path tokenizer/tokenizer.json ^
  --hidden-size 768 ^
  --num-layers 12 ^
  --num-heads 12 ^
  --intermediate-size 3072 ^
  --batch-size 4 ^
  --grad-accum-steps 2 ^
  --max-steps 540000 ^
  --max-hours 999 ^
  --log-every 25 ^
  --eval-every 250 ^
  --num-workers 2

echo.
echo Training stopped. Press any key to close this window.
pause > nul
