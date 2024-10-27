# Script d'installation des modèles Ollama
# Nécessite PowerShell 5.1 ou supérieur

# Fonction pour vérifier si Ollama est installé
function Test-OllamaInstallation {
    try {
        $ollamaVersion = ollama version
        Write-Host "Ollama est installé : $ollamaVersion" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "Ollama n'est pas installé ou n'est pas dans le PATH" -ForegroundColor Red
        Write-Host "Veuillez installer Ollama depuis https://ollama.ai/download" -ForegroundColor Yellow
        return $false
    }
}

# Fonction pour vérifier l'espace disque disponible
function Test-DiskSpace {
    $drive = Get-PSDrive C
    $freeSpaceGB = [math]::Round($drive.Free / 1GB, 2)
    $minimumSpaceGB = 20

    Write-Host "Espace disque disponible : $freeSpaceGB GB" -ForegroundColor Cyan
    
    if ($freeSpaceGB -lt $minimumSpaceGB) {
        Write-Host "Attention : Il est recommandé d'avoir au moins ${minimumSpaceGB}GB d'espace libre" -ForegroundColor Yellow
        return $false
    }
    return $true
}

# Liste des modèles à installer
$models = @(
    "llama3.1:8b-instruct-q4_1",
    "llama3:instruct",
    "llama3.2:3b-instruct-q8_0",
    "nemotron-mini",
    "nemotron-mini:4b-instruct-q5_0"
)

# Fonction principale d'installation
function Install-OllamaModels {
    if (-not (Test-OllamaInstallation)) {
        return
    }

    if (-not (Test-DiskSpace)) {
        $response = Read-Host "Voulez-vous continuer malgré le peu d'espace disque ? (O/N)"
        if ($response -ne "O") {
            return
        }
    }

    # Création d'un dossier de logs
    $logDir = ".\ollama_logs"
    if (-not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Path $logDir | Out-Null
    }

    $logFile = Join-Path $logDir "installation_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
    "Installation démarrée le $(Get-Date)" | Out-File $logFile

    $totalModels = $models.Count
    $currentModel = 0

    foreach ($model in $models) {
        $currentModel++
        $progress = [math]::Round(($currentModel / $totalModels) * 100)
        
        Write-Progress -Activity "Installation des modèles Ollama" -Status "Installation de $model" -PercentComplete $progress

        Write-Host "`nInstallation du modèle ($currentModel/$totalModels): $model"
        
        try {
            $startTime = Get-Date
            ollama pull $model 2>&1 | Tee-Object -Append -FilePath $logFile
            $endTime = Get-Date
            $duration = $endTime - $startTime
            
            "Modèle $model installé en $($duration.TotalMinutes.ToString('F2')) minutes" | Out-File $logFile -Append
            Write-Host "Installation réussie de $model" -ForegroundColor Green
        }
        catch {
            Write-Host "Erreur lors de l'installation de $model : $_"
            $_ | Out-File $logFile -Append
        }
    }

    Write-Progress -Activity "Installation des modèles Ollama" -Completed
    Write-Host "`nInstallation terminée. Log disponible dans : $logFile" -ForegroundColor Green
}

# Exécution du script
Clear-Host
Write-Host "Installation des modèles Ollama"
Write-Host "=================================" 

Install-OllamaModels