param(
    [string]$Python = "python",
    [string]$Config = "configs\experiment.yaml"
)

$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $PSScriptRoot

function Invoke-PythonStep {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python 命令执行失败：$Python $($Arguments -join ' ')"
    }
}

Set-Location $Project
Invoke-PythonStep @("scripts\check_environment.py")
Invoke-PythonStep @("scripts\prepare_camus.py", "--config", $Config)
Invoke-PythonStep @("scripts\cache_embeddings.py", "--config", $Config)
Invoke-PythonStep @("scripts\train_autobox.py", "--config", $Config)
Invoke-PythonStep @("scripts\evaluate_autobox.py", "--config", $Config)
Invoke-PythonStep @("scripts\make_table.py")
