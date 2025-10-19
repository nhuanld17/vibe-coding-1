# Test script for Missing Person AI API
# Run this script to test the API

Write-Host "Testing Missing Person AI API" -ForegroundColor Green
Write-Host "=" * 50

# 1. Test health endpoint
Write-Host "`nTesting /health endpoint..." -ForegroundColor Cyan
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
    Write-Host "✅ Health Check: " -NoNewline -ForegroundColor Green
    Write-Host ($health | ConvertTo-Json)
} catch {
    Write-Host "❌ Health check failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# 2. Test root endpoint
Write-Host "`nTesting / endpoint..." -ForegroundColor Cyan
try {
    $root = Invoke-RestMethod -Uri "http://localhost:8000/" -Method Get
    Write-Host "✅ Root endpoint: " -NoNewline -ForegroundColor Green
    Write-Host ($root | ConvertTo-Json)
} catch {
    Write-Host "❌ Root endpoint failed: $($_.Exception.Message)" -ForegroundColor Red
}

# 3. Test /test endpoint
Write-Host "`nTesting /test endpoint..." -ForegroundColor Cyan
try {
    $test = Invoke-RestMethod -Uri "http://localhost:8000/test" -Method Get
    Write-Host "✅ Test endpoint: " -NoNewline -ForegroundColor Green
    Write-Host ($test | ConvertTo-Json)
} catch {
    Write-Host "❌ Test endpoint failed: $($_.Exception.Message)" -ForegroundColor Red
}

# 4. Check Qdrant
Write-Host "`nTesting Qdrant connection..." -ForegroundColor Cyan
try {
    $qdrant = Invoke-RestMethod -Uri "http://localhost:6333/" -Method Get
    Write-Host "✅ Qdrant is running: " -NoNewline -ForegroundColor Green
    Write-Host ($qdrant | ConvertTo-Json)
} catch {
    Write-Host "❌ Qdrant connection failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n" + "=" * 50
Write-Host "API Testing Complete!" -ForegroundColor Green
Write-Host "`nAvailable endpoints:" -ForegroundColor Yellow
Write-Host "  - API Docs: http://localhost:8000/docs"
Write-Host "  - API Root: http://localhost:8000/"
Write-Host "  - Health: http://localhost:8000/health"
Write-Host "  - Qdrant Dashboard: http://localhost:6333/dashboard"
Write-Host "`nTo upload images, use Swagger UI at http://localhost:8000/docs" -ForegroundColor Cyan

