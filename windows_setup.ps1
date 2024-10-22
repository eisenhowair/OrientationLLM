# Version PowerShell (setup.ps1)
# Exécuter en tant qu'administrateur
$ErrorActionPreference = "Stop"

Write-Host "Installation de Python et pip..." -ForegroundColor Green
# Vérifier si Python est installé
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python n'est pas installé. Veuillez d'abord installer Python depuis python.org"
    Exit 1
}

# Vérifier si pip est installé
python -m ensurepip --upgrade

# Installation de Ollama pour Windows
Write-Host "Téléchargement et installation de Ollama..." -ForegroundColor Green
$ollamaUrl = "https://ollama.com/download/windows"
$ollamaInstaller = "ollama-windows.msi"
Invoke-WebRequest -Uri $ollamaUrl -OutFile $ollamaInstaller
Start-Process msiexec.exe -ArgumentList "/i $ollamaInstaller /quiet" -Wait

# Création de l'environnement virtuel
Write-Host "Création de l'environnement virtuel..." -ForegroundColor Green
python -m venv ProjetLLM

# Instructions pour l'utilisateur
Write-Host "`nPour activer l'environnement virtuel:" -ForegroundColor Yellow
Write-Host "1. Exécutez: .\ProjetLLM\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "2. Puis lancez le script d'installation des dépendances" -ForegroundColor Yellow
