# 启动本项目专用的 Label Studio 标注服务。
#
# 使用方法：
#   在 PowerShell 中进入项目根目录后执行：
#   powershell -ExecutionPolicy Bypass -File scripts\start_label_studio.ps1
#
# 说明：
#   1. 本脚本使用独立的 Conda 环境 LabelStudio，不污染 BookOCR 训练环境。
#   2. LABEL_STUDIO_LOCAL_FILES_* 两个变量允许 Label Studio 读取本项目下的本地图片。
#   3. 数据库和上传文件写入 data/interim/label_studio_data，该目录已被 .gitignore 忽略。

$ErrorActionPreference = "Stop"

# 根据脚本位置反推出项目根目录，避免从 PyCharm 或其他目录运行时路径错乱。
# 这里不用单独依赖 $PSScriptRoot，是为了兼容旧版 Windows PowerShell 的边界情况。
$ScriptPath = $PSCommandPath
if ([string]::IsNullOrWhiteSpace($ScriptPath)) {
    $ScriptPath = $MyInvocation.MyCommand.Path
}

if ([string]::IsNullOrWhiteSpace($ScriptPath)) {
    # 极少数启动方式下脚本路径可能为空，此时退回到当前目录。
    $ProjectRoot = (Resolve-Path ".").Path
} else {
    $ScriptDir = Split-Path -Parent $ScriptPath
    $ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
}
$LabelStudioExe = "D:\Downloads\Anaconda\envs\LabelStudio\Scripts\label-studio.exe"
$DataDir = Join-Path $ProjectRoot "data\interim\label_studio_data"
$LabelConfig = Join-Path $ProjectRoot "configs\label_studio\book_spine.xml"

if (-not (Test-Path $LabelStudioExe)) {
    throw "Label Studio executable not found: $LabelStudioExe"
}

if (-not (Test-Path $LabelConfig)) {
    throw "Label config not found: $LabelConfig"
}

New-Item -ItemType Directory -Force -Path $DataDir | Out-Null

# 开启本地文件访问。任务 JSON 中的 /data/local-files/?d=... 会以 ProjectRoot 为根目录解析。
# Label Studio 1.23 实际读取 LOCAL_FILES_*；同时保留 LABEL_STUDIO_* 兼容旧文档和旧版本。
$env:LOCAL_FILES_SERVING_ENABLED = "true"
$env:LOCAL_FILES_DOCUMENT_ROOT = $ProjectRoot
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = $ProjectRoot

Write-Host "Project root: $ProjectRoot"
Write-Host "Label Studio data dir: $DataDir"
Write-Host "Local files document root: $env:LOCAL_FILES_DOCUMENT_ROOT"
Write-Host "Label config: $LabelConfig"
Write-Host "URL: http://127.0.0.1:8080"
Write-Host ""
Write-Host "First open requires a local account. Then create a project and import data/interim/label_studio_tasks.json."
Write-Host "Press Ctrl+C to stop the service."

& $LabelStudioExe start `
    --data-dir $DataDir `
    --label-config $LabelConfig `
    --port 8080 `
    --no-browser
