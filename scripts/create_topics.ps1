# Path: scripts/create_topics.ps1
param(
    [string]$BootstrapServers = 'localhost:9092',
    [int]$Partitions = 3,
    [int]$ReplicationFactor = 1
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$kafkaHome   = 'C:\kafka'
$topicsBat   = Join-Path $kafkaHome 'bin\windows\kafka-topics.bat'

if (-not (Test-Path $topicsBat)) {
    Write-Error "Fichier manquant: $topicsBat"
}

# Topics du projet
$topics = @(
    'data.raw.stream',
    'data.cleaned.stream',
    'data.features.hourly',
    'data.predictions.weather'
)

foreach ($t in $topics) {
    Write-Host "[Kafka] Création du topic '$t' (si absent)..."
    cmd /c ('"{0}" --bootstrap-server {1} --create --topic {2} --partitions {3} --replication-factor {4} --if-not-exists' -f $topicsBat, $BootstrapServers, $t, $Partitions, $ReplicationFactor) | Out-Null
}

Write-Host '[Kafka] Topics existants:'
cmd /c ('"{0}" --bootstrap-server {1} --list' -f $topicsBat, $BootstrapServers)


