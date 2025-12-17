@echo off
setlocal EnableDelayedExpansion

:: =========================================================
:: BLACKBOX VACUUM SETUP SCRIPT (Windows)
:: =========================================================

set MODULE_NAME=blackbox-vacuum

echo.
echo ==================================================
echo    Setting up Blackbox Vacuum Environment
echo ==================================================
echo.

:: 1. Check if Go is installed
where go >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Go is not installed or not in PATH.
    echo Please install Go from https://go.dev/dl/
    pause
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
echo [INFO] Installing Core & Web Framework...
go get github.com/gin-gonic/gin
go get github.com/joho/godotenv
if !ERRORLEVEL! NEQ 0 goto :fail

echo.
echo [INFO] Installing AWS SDK...
go get github.com/aws/aws-sdk-go-v2
go get github.com/aws/aws-sdk-go-v2/config
go get github.com/aws/aws-sdk-go-v2/service/cloudtrail
if !ERRORLEVEL! NEQ 0 goto :fail

echo.
echo [INFO] Installing Azure SDK...
go get github.com/Azure/azure-sdk-for-go/sdk/azidentity
go get github.com/Azure/azure-sdk-for-go/sdk/resourcemanager/monitor/armmonitor
if !ERRORLEVEL! NEQ 0 goto :fail

echo.
echo [INFO] Installing Kubernetes Client...
go get k8s.io/client-go@latest
go get k8s.io/api@latest
go get k8s.io/apimachinery@latest
if !ERRORLEVEL! NEQ 0 goto :fail

echo.
echo [INFO] Installing Database Drivers...
go get github.com/denisenkom/go-mssqldb
go get github.com/lib/pq
go get go.mongodb.org/mongo-driver/mongo
go get github.com/redis/go-redis/v9
if !ERRORLEVEL! NEQ 0 goto :fail

echo.
echo [INFO] Installing SaaS SDKs (Slack)...
go get github.com/slack-go/slack
if !ERRORLEVEL! NEQ 0 goto :fail

:: 4. Cleanup
echo.
echo [INFO] Tidying up dependencies...
go mod tidy
if !ERRORLEVEL! NEQ 0 goto :fail

echo.
echo ==================================================
echo [SUCCESS] Setup Complete! You are ready to build.
echo ==================================================
echo.
pause
exit /b 0

:fail
echo.
echo [ERROR] Something went wrong. Check the logs above.
pause
exit /b 1