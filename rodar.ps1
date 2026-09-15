Set-Location $PSScriptRoot

$logs = Join-Path $PSScriptRoot "integracao\logs"
if (-not (Test-Path $logs)){
    New-Item -ItemType Directory -Path $logs | Out-Null
}

$data = Get-Date -Format "yyyy-MM-dd_HHmm"
$arquivo = Join-Path $logs "monitor_$data.log"

Write-Host "monitor iniciado - log em $arquivo"
Write-Host "para acompanhar: Get-Content `"$arquivo`" -Wait -Tail 20"

python integracao/monitor.py --modelos preditivo/modelos --ciclos 0 --intervalo 60 *> $arquivo