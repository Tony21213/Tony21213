@echo off
chcp 65001 >nul
rem Модели сегментации КТ: скачать веса TotalSegmentator и перевести в ONNX (один раз).
rem Для перевода нужны PyTorch и nnU-Net — они ставятся в отдельную папку .venv-models
rem и программе не нужны; после подготовки её можно удалить.
rem Результат — папка models в корне проекта (её же build_windows.bat кладёт рядом с exe).
rem Скачивается около 2 ГБ, перевод — 10-30 минут.
setlocal
cd /d "%~dp0\.."

if not exist .venv-models\Scripts\python.exe (
  python -m venv .venv-models || goto :error
  .venv-models\Scripts\python -m pip install --upgrade pip || goto :error
  .venv-models\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu || goto :error
  .venv-models\Scripts\python -m pip install nnunetv2 onnx onnxruntime || goto :error
)
.venv-models\Scripts\python tools\prepare_models.py models --only teeth craniofacial cavities || goto :error
echo.
echo Готово: папка models. Папку .venv-models можно удалить.
pause
exit /b 0

:error
echo Подготовка моделей не удалась.
pause
exit /b 1
