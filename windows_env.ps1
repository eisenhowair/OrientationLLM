# Exécuter en tant qu'administrateur
$ErrorActionPreference = "Stop"

# Définir le chemin du projet
$projectPath = Join-Path $HOME "OrientationLLM\ProjetLLM"

# Modifier les permissions (équivalent du chown)
Write-Host "Configuration des permissions..." -ForegroundColor Green
try {
    $acl = Get-Acl $projectPath
    $currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    $accessRule = New-Object System.Security.AccessControl.FileSystemAccessRule(
        $currentUser,
        "FullControl",
        "ContainerInherit,ObjectInherit",
        "None",
        "Allow"
    )
    $acl.SetAccessRule($accessRule)
    Set-Acl $projectPath $acl
}
catch {
    Write-Host "Attention: Erreur lors de la modification des permissions: $_" -ForegroundColor Yellow
    Write-Host "Le script continue..." -ForegroundColor Yellow
}

# Définir le dossier temporaire (équivalent de TMPDIR)
$env:TEMP = "C:\Windows\Temp"
$env:TMP = "C:\Windows\Temp"

# Mise à jour de wheel et setuptools
Write-Host "Mise à jour de wheel et setuptools..." -ForegroundColor Green
python -m pip install --upgrade wheel setuptools

# Installation des dépendances
Write-Host "Installation des dépendances depuis requirements.txt..." -ForegroundColor Green
try {
    python -m pip install --no-cache-dir -r requirements.txt
}
catch {
    Write-Host "Erreur lors de l'installation des dépendances: $_" -ForegroundColor Red
    Write-Host "`nSi vous rencontrez des problèmes d'espace disque, essayez de:" -ForegroundColor Yellow
    Write-Host "1. Nettoyer le dossier temp: Remove-Item $env:TEMP\* -Recurse -Force" -ForegroundColor Yellow
    Write-Host "2. Relancer ce script" -ForegroundColor Yellow
    Exit 1
}

Write-Host "`nInstallation terminée avec succès!" -ForegroundColor Green
Write-Host "Si vous rencontrez des problèmes d'espace disque, relancez ce script après avoir nettoyé le dossier temp" -ForegroundColor Yellow