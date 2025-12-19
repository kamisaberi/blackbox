@echo off
setlocal EnableDelayedExpansion

echo ==================================================
echo    Setting up Blackbox Relay (SOAR)
echo ==================================================

:: 1. Check Go
where go >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Go is not installed.
    exit /b 1
)

:: 2. Initialize Module
if not exist go.mod (
    echo [INFO] Initializing module...
    go mod init blackbox-relay
)

:: 3. Install Dependencies
echo [INFO] Installing Redis Client...
go get github.com/redis/go-redis/v9

:: 4. Tidy
echo [INFO] Tidying dependencies...
go mod tidy

echo.
echo [SUCCESS] Ready to build.
pause