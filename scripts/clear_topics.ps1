# Path: scripts/clear_topics.ps1
# Script PowerShell pour vider le contenu des topics Kafka
# Supprime et recrée les topics pour les vider complètement

param(
    [string]$BootstrapServers = 'localhost:9092',
    [int]$Partitions = 3,
    [int]$ReplicationFactor = 1,
    [string]$KafkaHome = 'C:\kafka'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Continue'

# Chercher kafka-topics.bat
$topicsBat = Join-Path $KafkaHome 'bin\windows\kafka-topics.bat'
if (-not (Test-Path $topicsBat)) {
    # Essayer sans windows
    $topicsBat = Join-Path $KafkaHome 'bin\kafka-topics.bat'
    if (-not (Test-Path $topicsBat)) {
        # Essayer .sh
        $topicsSh = Join-Path $KafkaHome 'bin\kafka-topics.sh'
        if (Test-Path $topicsSh) {
            Write-Host "[INFO] Utilisation de kafka-topics.sh (Linux/WSL)"
            $topicsBat = $topicsSh
        } else {
            Write-Error "Kafka introuvable dans $KafkaHome"
            Write-Host "Verifiez le chemin Kafka ou specifiez -KafkaHome"
            exit 1
        }
    }
}

# Topics du projet
$topics = @(
    'data.raw.stream',
    'data.cleaned.stream',
    'data.features.hourly',
    'data.predictions.weather'
)

Write-Host ("=" * 60)
Write-Host "VIDAGE DES TOPICS KAFKA"
Write-Host ("=" * 60)
Write-Host ""

foreach ($topic in $topics) {
    Write-Host "[Kafka] Traitement du topic: $topic"
    
    # Supprimer le topic (ignore les erreurs si n'existe pas)
    Write-Host "  -> Suppression du topic (si existe)..."
    $deleteArgs = @(
        "--bootstrap-server", $BootstrapServers,
        "--delete",
        "--topic", $topic
    )
    & $topicsBat $deleteArgs 2>&1 | Out-Null
    
    # Attendre un peu pour que la suppression soit effective
    Start-Sleep -Seconds 1
    
    # Recréer le topic vide
    Write-Host "  -> Creation du topic vide..."
    $createArgs = @(
        "--bootstrap-server", $BootstrapServers,
        "--create",
        "--topic", $topic,
        "--partitions", $Partitions,
        "--replication-factor", $ReplicationFactor,
        "--if-not-exists"
    )
    $result = & $topicsBat $createArgs 2>&1
    
    if ($LASTEXITCODE -eq 0 -or $result -match "already exists") {
        Write-Host "  -> [OK] Topic pret (vide)"
    } else {
        Write-Host "  -> [ATTENTION] $result"
    }
    
    Write-Host ""
}

Write-Host ("=" * 60)
Write-Host "[OK] Tous les topics ont ete vides et recrees"
Write-Host ("=" * 60)
Write-Host ""
Write-Host "Topics existants:"
$listArgs = @("--bootstrap-server", $BootstrapServers, "--list")
& $topicsBat $listArgs

