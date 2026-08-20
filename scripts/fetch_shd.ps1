# SHD Data Fetcher für Windows
# Lädt echte SHD-Daten von ZenkeLab herunter und konvertiert sie.

param()

$ErrorActionPreference = "Stop"
$DATA_DIR = "data/shd"
$BASE_URL = "https://zenkelab.org/datasets"

# Erstelle Verzeichnis
if (-not (Test-Path $DATA_DIR)) {
    New-Item -ItemType Directory -Path $DATA_DIR -Force | Out-Null
}

function Expand-Gzip {
    param(
        [string]$InputFile,
        [string]$OutputFile
    )
    
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    
    $inputStream = [System.IO.File]::OpenRead($InputFile)
    $gzipStream = New-Object System.IO.Compression.GZipStream($inputStream, [System.IO.Compression.CompressionMode]::Decompress)
    $outputStream = [System.IO.File]::Create($OutputFile)
    
    $gzipStream.CopyTo($outputStream)
    
    $outputStream.Close()
    $gzipStream.Close()
    $inputStream.Close()
}

foreach ($split in @("train", "test")) {
    $gzFile = "$DATA_DIR/shd_${split}.h5.gz"
    $h5File = "$DATA_DIR/shd_${split}.h5"
    $url = "$BASE_URL/shd_${split}.h5.gz"
    
    if (Test-Path $h5File) {
        Write-Host "[SKIP] shd_${split}.h5 existiert bereits."
        continue
    }
    
    if (-not (Test-Path $gzFile)) {
        Write-Host "[DOWNLOAD] $url"
        Invoke-WebRequest -Uri $url -OutFile $gzFile -UseBasicParsing
    }
    
    Write-Host "[EXTRACT] $gzFile -> $h5File"
    Expand-Gzip -InputFile $gzFile -OutputFile $h5File
    Remove-Item $gzFile
}

Write-Host "[CONVERT] HDF5 -> JSON"
& python scripts/convert_shd.py

Write-Host "[DONE] Echte SHD-Daten in $DATA_DIR/shd.json"
