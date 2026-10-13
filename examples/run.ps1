$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
# Make a failing native command (uv/mimosa) stop the script, not just the last one.
$PSNativeCommandUseErrorActionPreference = $true
Set-Location $PSScriptRoot

uv run mimosa compare ./myog.ihbcp ./pif4.meme `
  --query-type bamm `
  --target-type pwm `
  --min-logerr 2 `
  --fasta ./foreground.fa

uv run mimosa compare ./gata2.ihbcp ./gata4.ihbcp `
  --query-type bamm `
  --target-type bamm `
  --min-logerr 2 `
  --fasta ./foreground.fa

uv run mimosa compare ./foxa2.meme ./pif4.meme `
  --query-type pwm `
  --target-type pwm `
  --min-logerr 2 `
  --fasta ./foreground.fa

uv run mimosa compare ./sitega_stat6.mat ./pif4.meme `
  --query-type sitega `
  --target-type pwm `
  --min-logerr 2 `
  --fasta ./foreground.fa

uv run mimosa compare ./sitega_gata2.mat ./sitega_stat6.mat `
  --query-type sitega `
  --target-type sitega `
  --min-logerr 2 `
  --fasta ./foreground.fa

uv run mimosa compare ./scores_1.fasta ./scores_2.fasta `
  --query-type scores `
  --target-type scores `
  --min-logerr 2
