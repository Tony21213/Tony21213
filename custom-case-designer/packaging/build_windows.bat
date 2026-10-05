@echo off
chcp 65001 >nul
rem Сборка Custom Case Designer для Windows 10/11.
rem Нужен Python 3.10-3.12 (python.org, галочка "Add python.exe to PATH").
rem Результат: dist\CustomCaseDesigner\CustomCaseDesigner.exe
setlocal
cd /d "%~dp0\.."

if not exist .venv (
  python -m venv .venv || goto :error
)
call .venv\Scripts\activate.bat || goto :error
python -m pip install --upgrade pip
rem onnxruntime-directml — расчёт на любой видеокарте с DirectX 12 (NVIDIA, AMD, Intel)
pip install -r packaging\requirements-app.txt pyinstaller || goto :error

pyinstaller --noconfirm packaging\casedesigner.spec || goto :error

rem Модели сегментации: если рядом есть папка models (tools\prepare_models.py), кладём её к exe.
if exist models (
  xcopy /E /I /Y models dist\CustomCaseDesigner\models >nul
) else (
  echo Внимание: папки models нет - сегментация будет недоступна, пока её не указать в приложении.
)
echo.
echo Готово: dist\CustomCaseDesigner\CustomCaseDesigner.exe
exit /b 0

:error
echo Сборка не удалась.
exit /b 1
