@echo off
setlocal EnableDelayedExpansion

:: Define Module Name
set MODULE_NAME=blackbox-vacuum

echo ==================================================
echo    Setting up Blackbox Vacuum Environment
echo ==================================================

:: 1. Check if Go is installed
where go >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Go is not installed or not in PATH.
    echo Please install Go from https://go.dev/dl/
    exit /b 1
)

:: 2. Initialize Go Module
if not exist go.mod (
    echo [INFO] Initializing module: %MODULE_NAME%
    go mod init %MODULE_NAME%
    if !ERRORLEVEL! NEQ 0 goto :fail
) else (
    echo [INFO] go.mod already exists. Skipping init.
)

:: 3. Install Dependencies
echo.
echo [INFO] Installing Core Networking (Gin)...
go get github.com/gin-gonic/gin
if !ERRORLEVEL! NEQ 0 goto :fail

echo [INFO] Installing AWS SDK...
go get github.com/aws/aws-sdk-go-v2
go get github.com/aws/aws-sdk-go-v2/config
go get github.com/aws/aws-sdk-go-v2/service/cloudtrail
if !ERRORLEVEL! NEQ 0 goto :fail

echo [INFO] Installing SQL Driver (MSSQL)...
go get github.com/denisenkom/go-mssqldb
if !ERRORLEVEL! NEQ 0 goto :fail

echo [INFO] Installing Utilities (DotEnv)...
go get github.com/joho/godotenv
if !ERRORLEVEL! NEQ 0 goto :fail

:: 4. Tidy up
echo.
echo [INFO] Tidying up dependencies...
go mod tidy
if !ERRORLEVEL! NEQ 0 goto :fail

echo.
echo ==================================================
echo [SUCCESS] Setup Complete! You are ready to build.
echo ==================================================
pause
exit /b 0

:fail
echo.
echo [ERROR] Something went wrong. Check the logs above.
pause
exit /b 1