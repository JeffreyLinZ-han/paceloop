$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $repoRoot '.venv\Scripts\python.exe'
$entryPath = Join-Path $repoRoot 'loop.py'
if (!(Test-Path -LiteralPath $pythonPath)) {
    throw 'Create .venv and install requirements.txt before installing the loop command.'
}

$binDirectory = Join-Path ([Environment]::GetFolderPath('UserProfile')) '.local\bin'
$launcherPath = Join-Path $binDirectory 'loop.cmd'
if ((Test-Path -LiteralPath $launcherPath) -and
    !(Get-Content -LiteralPath $launcherPath -Raw).Contains('rem PaceLoop launcher')) {
    throw "A different command already exists at $launcherPath"
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
$profileEncoding = New-Object System.Text.UTF8Encoding($true)
New-Item -ItemType Directory -Path $binDirectory -Force | Out-Null
$launcher = "@echo off`r`nrem PaceLoop launcher`r`n`"$pythonPath`" `"$entryPath`" %*`r`nexit /b %errorlevel%`r`n"
[IO.File]::WriteAllText($launcherPath, $launcher, $utf8)

$userPath = [string][Environment]::GetEnvironmentVariable('Path', 'User')
if (@($userPath -split ';').Where({ $_.TrimEnd('\') -ieq $binDirectory }).Count -eq 0) {
    $updatedPath = $userPath.TrimEnd(';') + ';' + $binDirectory
    [Environment]::SetEnvironmentVariable('Path', $updatedPath, 'User')
}

$escapedPython = $pythonPath.Replace("'", "''")
$escapedEntry = $entryPath.Replace("'", "''")
$profileBlock = @"
# >>> PaceLoop command >>>
function global:loop {
    & '$escapedPython' '$escapedEntry' @args
}
# <<< PaceLoop command <<<
"@
$documents = [Environment]::GetFolderPath('MyDocuments')
$profileFiles = @(
    (Join-Path $documents 'WindowsPowerShell\profile.ps1'),
    (Join-Path $documents 'PowerShell\profile.ps1')
)
foreach ($profileFile in $profileFiles) {
    $existing = if (Test-Path -LiteralPath $profileFile) {
        [IO.File]::ReadAllText($profileFile)
    } else { '' }
    if ($existing.Contains('# >>> PaceLoop command >>>')) {
        $updated = [regex]::Replace(
            $existing,
            '(?s)# >>> PaceLoop command >>>.*?# <<< PaceLoop command <<<',
            [System.Text.RegularExpressions.MatchEvaluator]{ param($match) $profileBlock }
        )
    } else {
        if ($existing -match '(?im)^\s*function\s+(?:global:)?loop\s*[{(]') {
            throw "A loop function already exists in $profileFile"
        }
        $updated = $existing.TrimEnd() + "`r`n`r`n" + $profileBlock + "`r`n"
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $profileFile) -Force | Out-Null
    [IO.File]::WriteAllText($profileFile, $updated, $profileEncoding)
}

. $profileFiles[0]
Write-Output 'Installed: loop twin | loop train | loop show'
Write-Output 'Open a new PowerShell terminal to use loop. Restart your terminal app for Command Prompt.'
