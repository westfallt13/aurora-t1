@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

echo ===============================================
echo  Refreshing GGUF from the latest checkpoint
echo ===============================================
echo  This downloads the latest S3 checkpoint, converts
echo  it to GGUF, and reloads it into Ollama as
echo  "aurora-t1-scratch". Training does NOT need to be
echo  paused for this -- it only reads a snapshot.
echo ===============================================
echo.

set PY="%~dp0.venv\Scripts\python.exe"
set AWS="C:\Program Files\Amazon\AWSCLIV2\aws.exe"

%AWS% s3 cp s3://aurora-llm-checkpoints-752988091124/checkpoints/local-110m/latest.pt cloud\latest_checkpoint.pt --region us-east-1 --no-progress
if errorlevel 1 goto :error

cd cloud
%PY% export_hf_gpt2.py --checkpoint latest_checkpoint.pt --tokenizer-path tokenizer/tokenizer.json --output-dir hf_export --hidden-size 768 --num-layers 12 --num-heads 12 --intermediate-size 3072 --seq-length 512 --vocab-size 32000
if errorlevel 1 goto :error

%PY% ..\tools\llama.cpp\convert_hf_to_gguf.py hf_export --outfile aurora-t1.gguf --outtype f16
if errorlevel 1 goto :error

ollama create aurora-t1-scratch -f Modelfile
if errorlevel 1 goto :error

echo.
echo Done. Try it with:  ollama run aurora-t1-scratch
pause
exit /b 0

:error
echo.
echo Something failed -- see the error above.
pause
exit /b 1
