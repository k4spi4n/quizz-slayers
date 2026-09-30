# EDUX Slayers - Cập nhật extension tại chỗ.
# Chép đè bản mới nhất từ GitHub Releases vào CHÍNH thư mục này, nên trình duyệt vẫn
# coi là cùng một extension (cùng ID) -> cấu hình AI, API key và cài đặt được giữ nguyên.
param([switch]$Force)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Repo = 'k4spi4n/quizz-slayers'
$ZipUrl = "https://github.com/$Repo/releases/latest/download/edux-extension.zip"
$ExtDir = $PSScriptRoot

function Finish([int]$Code) {
    Write-Host ''
    Read-Host 'Nhấn Enter để đóng' | Out-Null
    exit $Code
}

Write-Host '=== EDUX Slayers - Cập nhật extension ===' -ForegroundColor Cyan

$ManifestPath = Join-Path $ExtDir 'manifest.json'
if (-not (Test-Path $ManifestPath)) {
    Write-Host "Không thấy manifest.json trong $ExtDir." -ForegroundColor Red
    Write-Host 'Hãy đặt update.bat trong thư mục extension rồi chạy lại.'
    Finish 1
}

if (Test-Path (Join-Path $ExtDir '..\.git')) {
    Write-Host 'Thư mục này thuộc repo git. Hãy cập nhật bằng:  git pull' -ForegroundColor Yellow
    Write-Host 'rồi bấm Áp dụng trong popup (hoặc Reload extension).'
    Finish 0
}

$Current = (Get-Content $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json).version
Write-Host "Phiên bản hiện tại: v$Current"
Write-Host "Thư mục extension : $ExtDir"

$Temp = Join-Path ([IO.Path]::GetTempPath()) ("edux-update-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $Temp | Out-Null
try {
    $Zip = Join-Path $Temp 'edux-extension.zip'
    $Unpacked = Join-Path $Temp 'files'

    Write-Host 'Đang tải bản mới nhất từ GitHub...'
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $ZipUrl -OutFile $Zip -UseBasicParsing
    Expand-Archive -Path $Zip -DestinationPath $Unpacked -Force

    # Hỗ trợ cả zip phẳng lẫn zip có 1 thư mục bọc ngoài
    $Source = $Unpacked
    if (-not (Test-Path (Join-Path $Source 'manifest.json'))) {
        $Inner = Get-ChildItem $Unpacked -Directory | Where-Object { Test-Path (Join-Path $_.FullName 'manifest.json') } | Select-Object -First 1
        if (-not $Inner) { throw 'File zip tải về không chứa manifest.json.' }
        $Source = $Inner.FullName
    }

    $Latest = (Get-Content (Join-Path $Source 'manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json).version
    Write-Host "Phiên bản mới nhất: v$Latest"

    if (-not $Force -and ([version]$Latest -le [version]$Current)) {
        Write-Host 'Bạn đang dùng bản mới nhất, không cần cập nhật.' -ForegroundColor Green
        Finish 0
    }

    Copy-Item -Path (Join-Path $Source '*') -Destination $ExtDir -Recurse -Force

    Write-Host ''
    Write-Host "Đã cập nhật v$Current -> v$Latest" -ForegroundColor Green
    Write-Host 'Bước cuối: mở popup EDUX Slayers và bấm "Áp dụng"'
    Write-Host '(hoặc bấm nút Reload của extension trong trang chrome://extensions).'
    Write-Host 'Cấu hình AI và API key được giữ nguyên. KHÔNG bấm Remove extension.' -ForegroundColor Yellow
    Finish 0
}
catch {
    Write-Host ''
    Write-Host "Cập nhật thất bại: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host 'Bạn có thể tải thủ công edux-extension.zip từ:'
    Write-Host "  https://github.com/$Repo/releases/latest"
    Write-Host 'rồi giải nén ĐÈ LÊN thư mục extension hiện tại.'
    Finish 1
}
finally {
    Remove-Item $Temp -Recurse -Force -ErrorAction SilentlyContinue
}
