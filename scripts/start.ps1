#requires -Version 5.1
<#
  Execução nativa no Windows: Python, PostgreSQL/pgvector e modelos já instalados.
  .\scripts\start.ps1 [-Config config/lab.local.yaml] [-CheckOnly] [-Install]
  -Install instala as dependências Python em .venv a partir do lock versionado.
#>
[CmdletBinding()]
param(
    [string]$Config = "",
    [ValidateRange(1, 65535)][int]$Port = 8000,
    [switch]$Install,
    [switch]$CheckOnly
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
    if (-not (Test-Path "pyproject.toml") -or -not (Test-Path "uv.lock")) {
        throw "Checkout incompleto: pyproject.toml e uv.lock são obrigatórios."
    }
    if (-not $Config) {
        # O Python carrega .env e resolve RAG_CONFIG; não sobrescreva essa escolha aqui.
        $ConfigArgs = @()
    } else {
        if (-not (Test-Path $Config -PathType Leaf)) { throw "YAML não encontrado: $Config" }
        $ConfigArgs = @("--config", $Config)
    }
    if ($Install) {
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
            throw "Instale uv (https://docs.astral.sh/uv/getting-started/installation/) e tente novamente."
        }
        # uv valida a disponibilidade do Python e instala uma versão suportada se necessário.
        & uv sync --frozen --python 3.12
        if ($LASTEXITCODE -ne 0) { throw "Falha ao preparar o ambiente Python a partir do lock." }
    }
    $PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $PythonExe)) { throw "Ambiente ausente. Execute novamente com -Install." }
    & $PythonExe -c "import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] < (3,15) else 1)"
    if ($LASTEXITCODE -ne 0) { throw "Use Python 3.11 a 3.14 (recomendado: 3.12)." }
    & $PythonExe -m rag_lab.cli @ConfigArgs validate
    if ($LASTEXITCODE -ne 0) { throw "Configuração YAML inválida ou dependências ausentes." }
    Write-Host "Validando dependências, porta, PostgreSQL/pgvector, credenciais e modelos..."
    & $PythonExe -m rag_lab.cli @ConfigArgs doctor --port $Port
    if ($LASTEXITCODE -ne 0) {
        throw "Preflight falhou. Confira os itens acima e as instruções do README."
    }
    & $PythonExe -m rag_lab.cli @ConfigArgs init-db
    if ($LASTEXITCODE -ne 0) { throw "Não foi possível inicializar as tabelas do laboratório." }
    if (-not $CheckOnly) {
        & $PythonExe -m rag_lab.cli @ConfigArgs serve --port $Port
        if ($LASTEXITCODE -ne 0) { throw "O servidor encerrou com erro." }
    }
} finally {
    Pop-Location
}
