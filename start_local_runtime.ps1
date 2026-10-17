$ErrorActionPreference = 'Stop'
$kfmRoot = $PSScriptRoot
$kfmOllama = Join-Path $kfmRoot '.tools\ollama\ollama.exe'
if (-not (Test-Path -LiteralPath $kfmOllama)) { throw 'Local Ollama was not found in .tools/ollama.' }
try {
    $kfmRequest = [System.Net.HttpWebRequest]::Create('http://127.0.0.1:11434/api/version')
    $kfmRequest.Proxy = $null
    $kfmRequest.Timeout = 2000
    $kfmResponse = $kfmRequest.GetResponse()
    $kfmReader = [System.IO.StreamReader]::new($kfmResponse.GetResponseStream())
    $kfmVersion = $kfmReader.ReadToEnd() | ConvertFrom-Json
    $kfmReader.Dispose()
    $kfmResponse.Dispose()
    if ($kfmVersion.version) { exit 0 }
} catch { }
$env:OLLAMA_MODELS = Join-Path $kfmRoot '.models'
$env:OLLAMA_NO_CLOUD = '1'
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_CONTEXT_LENGTH = '4096'
$env:OLLAMA_NUM_PARALLEL = '1'
$env:OLLAMA_MAX_LOADED_MODELS = '1'
$env:OLLAMA_FLASH_ATTENTION = '1'
$env:OLLAMA_KEEP_ALIVE = '10m'
$kfmProcess = Start-Process -FilePath $kfmOllama -ArgumentList 'serve' -WorkingDirectory $kfmRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $kfmRoot '.tools\ollama-out.log') -RedirectStandardError (Join-Path $kfmRoot '.tools\ollama-error.log') -PassThru
$kfmProcess.Id | Set-Content -LiteralPath (Join-Path $kfmRoot '.tools\ollama.pid')
for ($kfmAttempt = 0; $kfmAttempt -lt 30; $kfmAttempt++) {
    try {
        $kfmReadyRequest = [System.Net.HttpWebRequest]::Create('http://127.0.0.1:11434/api/version')
        $kfmReadyRequest.Proxy = $null
        $kfmReadyRequest.Timeout = 1000
        $kfmReadyResponse = $kfmReadyRequest.GetResponse()
        $kfmReadyResponse.Dispose()
        exit 0
    } catch {
        Start-Sleep -Milliseconds 500
    }
}
throw 'The local model runtime did not start. See .tools/ollama-error.log.'
