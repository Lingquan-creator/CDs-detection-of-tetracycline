@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."

cd /d "%REPO_ROOT%"
echo Building PicRGBRoiTool for Windows...
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%build_pic_rgb_gui.ps1"
set "BUILD_EXIT_CODE=%ERRORLEVEL%"

echo.
if "%BUILD_EXIT_CODE%"=="0" (
    echo Build succeeded.
    echo Output:
    echo   "%SCRIPT_DIR%dist\PicRGBRoiTool\PicRGBRoiTool.exe"
) else (
    echo Build failed. Exit code: %BUILD_EXIT_CODE%
)
echo.
pause
exit /b %BUILD_EXIT_CODE%
