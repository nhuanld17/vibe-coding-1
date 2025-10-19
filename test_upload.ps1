# Test upload API
Write-Host "Testing Missing Person API Upload" -ForegroundColor Green

# Test with first image from FGNET dataset
$imagePath = "datasets\FGNET_organized\person_001\age_02.jpg"

if (-Not (Test-Path $imagePath)) {
    Write-Host "ERROR: Image not found at $imagePath" -ForegroundColor Red
    Write-Host "Please check the path" -ForegroundColor Yellow
    exit 1
}

Write-Host "Image found: $imagePath" -ForegroundColor Cyan

# Create metadata
$metadata = @{
    case_id = "TEST_001"
    name = "Test Person 001"
    age_at_disappearance = 2
    year_disappeared = 2023
    gender = "male"
    location_last_seen = "Test Location"
    contact = "test@example.com"
} | ConvertTo-Json

Write-Host "`nMetadata:" -ForegroundColor Cyan
Write-Host $metadata

Write-Host "`nUploading to API..." -ForegroundColor Yellow

try {
    # Read file as bytes
    $fileBytes = [System.IO.File]::ReadAllBytes((Resolve-Path $imagePath))
    $fileName = Split-Path $imagePath -Leaf
    
    # Create multipart form data
    $boundary = [System.Guid]::NewGuid().ToString()
    $LF = "`r`n"
    
    $bodyLines = @(
        "--$boundary",
        "Content-Disposition: form-data; name=`"image`"; filename=`"$fileName`"",
        "Content-Type: image/jpeg$LF"
    ) -join $LF
    
    $bodyLines += [System.Text.Encoding]::GetEncoding("iso-8859-1").GetString($fileBytes)
    
    $bodyLines += @(
        "$LF--$boundary",
        "Content-Disposition: form-data; name=`"metadata`"$LF",
        $metadata,
        "--$boundary--$LF"
    ) -join $LF
    
    $response = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/upload/missing" `
        -Method Post `
        -ContentType "multipart/form-data; boundary=$boundary" `
        -Body ([System.Text.Encoding]::GetEncoding("iso-8859-1").GetBytes($bodyLines))
    
    Write-Host "`nSUCCESS!" -ForegroundColor Green
    Write-Host ($response | ConvertTo-Json -Depth 5)
    
} catch {
    Write-Host "`nERROR!" -ForegroundColor Red
    
    if ($_.Exception.Response) {
        Write-Host "Status Code: $($_.Exception.Response.StatusCode.value__)" -ForegroundColor Red
    }
    
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    
    # Try to get error details
    if ($_.ErrorDetails) {
        Write-Host "`nError Details:" -ForegroundColor Yellow
        Write-Host $_.ErrorDetails.Message
    }
}

