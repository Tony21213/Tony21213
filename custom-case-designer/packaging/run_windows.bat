@echo off
chcp 65001 >nul
rem Custom Case Designer: запуск из исходников на Windows 10/11 (без сборки exe).
rem Нужен Python 3.10-3.12 с python.org (при установке — галочка "Add python.exe to PATH").
rem Первый запуск ставит библиотеки в папку .venv (несколько минут), дальше — сразу окно программы.
rem Модели сегментации — папка models в корне проекта (packaging\prepare_models.bat).
setlocal
cd /d "%~dp0\.."

if not exist .venv\Scripts\python.exe (
  echo Первый запуск: ставлю библиотеки...
  python -m venv .venv || goto :nopython
  .venv\Scripts\python -m pip install --upgrade pip || goto :error
  .venv\Scripts\python -m pip install -r packaging\requirements-app.txt || goto :error
)
if not exist models (
  echo Внимание: папки models нет - сегментация попросит указать папку моделей. Подготовить: packaging\prepare_models.bat
)
.venv\Scripts\python -m casedesigner.app %*
exit /b %errorlevel%

:nopython
echo Не найден Python. Установите Python 3.12 с python.org (галочка "Add python.exe to PATH") и запустите снова.
pause
exit /b 1

:error
echo Не удалось поставить библиотеки (нужен интернет). Удалите папку .venv и запустите снова.
pause
exit /b 1
