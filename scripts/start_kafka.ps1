# Path: scripts/start_kafka.ps1
param(
    [switch]$Clean,
    [int]$WaitSeconds = 8,
    [switch]$Attach
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$kafkaHome   = 'C:\kafka'
$serverProps = Join-Path $kafkaHome 'config\kraft\server.properties'
$storageBat  = Join-Path $kafkaHome 'bin\windows\kafka-storage.bat'
$serverBat   = Join-Path $kafkaHome 'bin\windows\kafka-server-start.bat'
# Déduire log.dirs depuis server.properties (fallback vers dossier par défaut connu)
$serverPropsContent = Get-Content -Path $serverProps -ErrorAction Stop
$logDirsLine = $serverPropsContent | Where-Object { $_ -match '^log\.dirs\s*=\s*' } | Select-Object -First 1
if ($null -ne $logDirsLine) {
    $logDir = ($logDirsLine -split '=',2)[1].Trim()
} else {
    $logDir = Join-Path $kafkaHome 'kafkakraft-combined-logs'
}
# Normaliser les backslashes
$logDir = $logDir -replace '/', '\\'

if (-not (Test-Path $serverProps)) {
    Write-Error "Kafka KRaft server.properties introuvable: $serverProps"
}
if (-not (Test-Path $storageBat)) {
    Write-Error "Fichier manquant: $storageBat"
}
if (-not (Test-Path $serverBat)) {
    Write-Error "Fichier manquant: $serverBat"
}

if ($Clean) {
    Write-Host "[Kafka] Arrêt des processus Java (si présents)..."
    Get-Process -Name java -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    if (Test-Path $logDir) {
        Write-Host "[Kafka] Nettoyage du répertoire de logs: $logDir"
        Remove-Item -Recurse -Force $logDir
    }
}

# Choisir un cluster.id et NE PAS reformater si meta.properties existe déjà (pour ne pas casser les topics)
$metaPath = Join-Path $logDir 'meta.properties'
$metaPathFallback = 'C:\kafka\kraft-combined-logs\meta.properties'
$skipFormat = $false
$metaExists = (Test-Path $metaPathFallback -PathType Leaf) -or (Test-Path $metaPath -PathType Leaf)

if (-not $Clean -and $metaExists) {
    # Toujours réutiliser le stockage existant si meta.properties présent (pas de reformatage)
    if (Test-Path $metaPathFallback -PathType Leaf) {
        $sourceMeta = $metaPathFallback
    } else {
        $sourceMeta = $metaPath
    }
    try {
        $existingLine = Get-Content $sourceMeta | Where-Object { $_ -match '^cluster\.id=' } | Select-Object -First 1
        if ($existingLine) {
            $clusterId = ($existingLine -split '=',2)[1].Trim()
            Write-Host "[Kafka] cluster.id existant détecté: $clusterId"
        } else {
            Write-Host "[Kafka] meta.properties présent sans cluster.id explicite, on réutilise quand même."
        }
    } catch {
        Write-Host "[Kafka] meta.properties présent, réutilisation du stockage."
    }
    $skipFormat = $true
}

if (-not $skipFormat) {
    Write-Host "[Kafka] Génération d'un cluster.id..."
    try {
        $raw = cmd /c ('"{0}" random-uuid' -f $storageBat) 2>$null
    } catch {
        $raw = $null
    }
    if (-not $raw) {
        # Fallback direct Java si kafka-storage.bat échoue (classpath vide dans certaines installs)
        $raw = & java -cp (Join-Path $kafkaHome 'libs\*') kafka.tools.StorageTool random-uuid
    }
    $clusterId = ([string]$raw).Trim()
    if (-not $clusterId) { Write-Error 'Échec génération du cluster.id' }
    Write-Host "[Kafka] cluster.id: $clusterId"

    Write-Host "[Kafka] Formatage du stockage (premier démarrage ou Clean)..."
    $formatCmd = @(
        'java'
        '-cp', (Join-Path $kafkaHome 'libs\*')
        'kafka.tools.StorageTool'
        'format', '-t', $clusterId, '-c', $serverProps, '--ignore-formatted'
    )
    $formatArgs = $formatCmd[1..($formatCmd.Length-1)]
    & $formatCmd[0] @formatArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Error 'Échec du formatage du stockage (vérifier log.dirs et meta.properties)'
    }
} else {
    Write-Host "[Kafka] Stockage déjà formaté, pas de reformatage (topics conservés)."
}

if ($Attach) {
    Write-Host "[Kafka] Démarrage du broker en mode attaché (Ctrl+C pour arrêter)..."
    $startCmd = @(
        'java'
        '-cp', (Join-Path $kafkaHome 'libs\*')
        "-Dlog4j.configuration=file:$(Join-Path $kafkaHome 'config\tools-log4j.properties')"
        "-Dkafka.logs.dir=$(Join-Path $kafkaHome 'logs')"
        'kafka.Kafka'
        $serverProps
    )
    $startArgs = $startCmd[1..($startCmd.Length-1)]
    & $startCmd[0] @startArgs
    exit $LASTEXITCODE
} else {
    Write-Host "[Kafka] Démarrage du broker (fenêtre minimisée)..."
    $javaArgs = @(
        '-cp', (Join-Path $kafkaHome 'libs\*')
        "-Dlog4j.configuration=file:$(Join-Path $kafkaHome 'config\tools-log4j.properties')"
        "-Dkafka.logs.dir=$(Join-Path $kafkaHome 'logs')"
        'kafka.Kafka'
        $serverProps
    )
    Start-Process -FilePath 'java' -ArgumentList $javaArgs -WindowStyle Minimized | Out-Null
    Start-Sleep -Seconds $WaitSeconds
    try {
        $r = Test-NetConnection -ComputerName localhost -Port 9092 -WarningAction SilentlyContinue
        if ($r.TcpTestSucceeded) {
            Write-Host '[Kafka] Broker OK sur localhost:9092'
            exit 0
        } else {
            Write-Error 'Broker non joignable sur localhost:9092'
        }
    } catch {
        Write-Error $_
    }
}


