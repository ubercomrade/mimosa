$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location $PSScriptRoot

mimosa profile ./myog.ihbcp ./pif4.meme `
  --model1-type bamm `
  --model2-type pwm `
  --metric dice `
  --fasta ./foreground.fa

mimosa profile ./gata2.ihbcp ./gata4.ihbcp `
  --model1-type bamm `
  --model2-type bamm `
  --metric co `
  --fasta ./foreground.fa

mimosa profile ./foxa2.meme ./pif4.meme `
  --model1-type pwm `
  --model2-type pwm `
  --metric dice `
  --fasta ./foreground.fa

mimosa profile ./sitega_stat6.mat ./pif4.meme `
  --model1-type sitega `
  --model2-type pwm `
  --metric cosine `
  --fasta ./foreground.fa

mimosa profile ./sitega_gata2.mat ./sitega_stat6.mat `
  --model1-type sitega `
  --model2-type sitega `
  --metric co `
  --fasta ./foreground.fa

mimosa profile ./scores_1.fasta ./scores_2.fasta `
  --model1-type scores `
  --model2-type scores `
  --metric cosine
