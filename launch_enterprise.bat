@echo off
setlocal EnableDelayedExpansion

echo ==================================================
echo    BLACKBOX ENTERPRISE LAUNCHER
echo ==================================================

:: 1. Check Directory Context
if not exist "blackbox-deploy" (
    echo [ERROR] 'blackbox-deploy' folder not found.
    echo Please run this script from the root 'blackbox' directory.
    pause
    exit /b 1
)

:: 2. Configuration Prompt
echo.
echo [OPTIONAL] Configure Slack Alerts
set /p SLACK_INPUT="Paste Slack Webhook URL (or press Enter to skip): "

if "%SLACK_INPUT%"=="" (
    set SLACK_WEBHOOK_URL=
    echo [INFO] Slack alerts disabled.
) else (
    set SLACK_WEBHOOK_URL=%SLACK_INPUT%
    echo [INFO] Slack integration enabled.
)

:: 3. Docker Compose Launch
echo.
echo [INFO] Building and Starting Services...
cd blackbox-deploy

docker-compose -f compose/docker-compose.yml up -d --build

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker Launch Failed. Is Docker Desktop running?
    pause
    exit /b 1
)

:: 4. Verification
echo.
echo ==================================================
echo [SUCCESS] Stack is Online!
echo.
echo   [HUD]      http://localhost:3000
echo   [API]      http://localhost:8080
echo   [DB]       http://localhost:8123
echo ==================================================
echo.
echo [INFO] Checking Relay logs...
timeout /t 3 /nobreak >nul
docker logs blackbox-relay

pause