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

rem Модели сегментации: если в корне проекта уже есть папка models, кладём её к exe;
rem иначе их скачают кнопкой в программе.
if exist models (
  xcopy /E /I /Y models dist\CustomCaseDesigner\models >nul
) else (
  echo Моделей в сборке нет - их скачают кнопкой в программе.
)
echo.
echo Готово: dist\CustomCaseDesigner\CustomCaseDesigner.exe
exit /b 0

:error
echo Сборка не удалась.
exit /b 1
