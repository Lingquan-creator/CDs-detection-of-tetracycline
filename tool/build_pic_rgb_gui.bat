@echo off
setlocal
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_pic_rgb_gui.ps1"
