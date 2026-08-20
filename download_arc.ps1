New-Item -ItemType Directory -Force -Path data\arc | Out-Null

$git = Get-Command git -ErrorAction SilentlyContinue
if ($git) {
    Write-Output "Git found. Cloning ARC repository..."
    if (-not (Test-Path data\arc\ARC)) {
        git clone --depth 1 https://github.com/fchollet/ARC.git data\arc\ARC 2>&1
    }
    if (Test-Path data\arc\ARC\data\training) {
        Copy-Item data\arc\ARC\data\training\*.json data\arc\ -Force
        Write-Output "Copied training tasks"
    }
    if (Test-Path data\arc\ARC\data\evaluation) {
        Copy-Item data\arc\ARC\data\evaluation\*.json data\arc\ -Force
        Write-Output "Copied evaluation tasks"
    }
    Remove-Item data\arc\ARC -Recurse -Force
    Write-Output "Cleaned up ARC clone"
} else {
    Write-Output "Git not found. Using fallback: downloading individual files..."
    $baseUrl = "https://raw.githubusercontent.com/fchollet/ARC/master/data/training"
    $taskIds = @("00d62c1b","01d72849","01c9dda9","01c5c2a6","01c9485c","01d2a3f2","01d26d7c","01d18664","01d2abde","01d2013b","01d26e7c","01d2825c","01d5e2b5","01d6b12e","01d80f09","01d9e4e7","01daa01a","01ddac80","01e2a64c","01e3a0d5","01e3a0d6","01e3a0d7","01e3a0d8","01e3a0d9","01e3a0da","01e3a0db","01e3a0dc","01e3a0dd","01e3a0de","01e3a0df","01e3a0e0","01e3a0e1","01e3a0e2","01e3a0e3","01e3a0e4","01e3a0e5","01e3a0e6","01e3a0e7","01e3a0e8","01e3a0e9","01e3a0ea","01e3a0eb","01e3a0ec","01e3a0ed","01e3a0ee","01e3a0ef","01e3a0f0","01e3a0f1","01e3a0f2","01e3a0f3")
    
    foreach ($task in $taskIds) {
        $output = "data/arc/$task.json"
        if (-not (Test-Path $output)) {
            try {
                Invoke-WebRequest -Uri "$baseUrl/$task.json" -OutFile $output -UseBasicParsing -ErrorAction Stop
                Write-Output "Downloaded $task"
            } catch {
                Write-Output "Failed $task : $_"
            }
        }
    }
}

$count = (Get-ChildItem data\arc\*.json).Count
Write-Output ""
Write-Output "Total ARC tasks downloaded: $count"

Get-ChildItem data\arc\*.json | Select-Object -First 5 | ForEach-Object { Write-Output "  $($_.Name)" }
if ($count -gt 5) { Write-Output "  ... and $($count - 5) more" }
