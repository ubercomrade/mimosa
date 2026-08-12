$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location $PSScriptRoot

uv run mimosa compare ./myog.ihbcp ./pif4.meme `
  --query-type bamm `
  --target-type pwm `
  --metric dice `
  --fasta ./foreground.fa

uv run mimosa compare ./gata2.ihbcp ./gata4.ihbcp `
  --query-type bamm `
  --target-type bamm `
  --metric co `
  --fasta ./foreground.fa

uv run mimosa compare ./foxa2.meme ./pif4.meme `
  --query-type pwm `
  --target-type pwm `
  --metric dice `
  --fasta ./foreground.fa

uv run mimosa compare ./sitega_stat6.mat ./pif4.meme `
  --query-type sitega `
  --target-type pwm `
  --metric cosine `
  --fasta ./foreground.fa

uv run mimosa compare ./sitega_gata2.mat ./sitega_stat6.mat `
  --query-type sitega `
  --target-type sitega `
  --metric co `
  --fasta ./foreground.fa

uv run mimosa compare ./scores_1.fasta ./scores_2.fasta `
  --query-type scores `
  --target-type scores `
  --metric cosine
