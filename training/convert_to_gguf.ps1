param(
  [string]$MergedModel = "training/output/claude-fable-qwen-merged",
  [string]$LlamaCpp = "C:\Users\Admin\Desktop\my-clone\llama.cpp",
  [string]$OutFile = "models\qwen2.5-3b-claude-fable-f16.gguf"
)

$ErrorActionPreference = "Stop"
$convert = Join-Path $LlamaCpp "convert_hf_to_gguf.py"
if (!(Test-Path $convert)) {
  throw "convert_hf_to_gguf.py not found at $convert"
}

New-Item -ItemType Directory -Force -Path (Split-Path $OutFile) | Out-Null
python $convert $MergedModel --outfile $OutFile --outtype f16
Write-Host "Saved GGUF to $OutFile"
