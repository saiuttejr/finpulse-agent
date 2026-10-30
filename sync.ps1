#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Sync this folder to github.com/saiuttejr/finpulse-agent using a PAT from .env, WITHOUT leaving
    any git repository on this device.

.DESCRIPTION
    This device is intentionally kept git-free: there is no persistent .git folder here.
    Authentication uses a fine-grained Personal Access Token stored ONLY in .env (GITHUB_PAT).

    Each run creates a THROWAWAY .git only for the moment of the transfer, then deletes it:
      1. Loads GITHUB_PAT from .env (the value is never printed).
      2. git init + attaches the remote + an env-based credential helper.
      3. Fetches origin/main and adopts it as the base WITHOUT touching your files
         (git reset --mixed - working files are never overwritten).
      4. Push mode: commits every local change (deletions included) and pushes to origin/main;
                    aborts if .env would be committed.
         Pull mode: restores files that exist on the remote but are missing locally,
                    while keeping your local edits untouched.
      5. In a finally block, DELETES the throwaway .git and clears the token, so nothing
         git-related persists on this device.

    The token is never written to git config or the remote URL; the credential helper reads
    it from the GITHUB_PAT environment variable that lives only for the run's lifetime.

.PARAMETER Message
    Commit message (push mode). Defaults to a timestamped "sync: <date time>".

.PARAMETER PullOnly
    Pull remote-only files into this folder without committing or pushing. Local edits are kept.

.PARAMETER KeepGit
    Debugging aid: do NOT delete the throwaway .git when finished (lets you inspect state).

.EXAMPLE
    ./sync.ps1
    Commit + push everything, then remove .git.

.EXAMPLE
    ./sync.ps1 "rag: add risk governance corpus"
    Push with a custom commit message.

.EXAMPLE
    ./sync.ps1 -PullOnly
    Pull the latest from origin/main (keeps your local edits), then remove .git.
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [Alias('m')]
    [string]$Message,

    [switch]$PullOnly,

    [switch]$KeepGit
)

$ErrorActionPreference = 'Stop'

$RemoteUrl = 'https://github.com/saiuttejr/finpulse-agent.git'
$Branch    = 'main'
$GitUser   = 'saiuttejr'   # basic-auth username for the PAT (NOT x-access-token)

function Invoke-Git {
    param([Parameter(ValueFromRemainingArguments = $true)] [string[]]$GitArgs)
    & git @GitArgs
    if ($LASTEXITCODE -ne 0) {
        throw "git $($GitArgs -join ' ') failed (exit $LASTEXITCODE)"
    }
}

# --- Work from the folder this script lives in ---
$root = $PSScriptRoot
Set-Location -LiteralPath $root
$gitDir = Join-Path $root '.git'

# --- Load the PAT from .env (value is never printed) ---
$envFile = Join-Path $root '.env'
if (-not (Test-Path -LiteralPath $envFile)) {
    throw ".env not found at $envFile - cannot authenticate."
}
$line = Get-Content -LiteralPath $envFile |
    Where-Object { $_ -match '^\s*GITHUB_PAT\s*=' } |
    Select-Object -First 1
if (-not $line) { throw "GITHUB_PAT not found in .env" }
$pat = ($line -replace '^\s*GITHUB_PAT\s*=\s*', '').Trim().Trim('"').Trim("'")
if ([string]::IsNullOrWhiteSpace($pat)) { throw "GITHUB_PAT is empty in .env" }

$env:GITHUB_PAT = $pat
$env:GIT_TERMINAL_PROMPT = '0'   # fail fast instead of prompting if auth is wrong

try {
    # --- Build a throwaway repo attached to the remote ---
    Invoke-Git init -b $Branch | Out-Null
    & git remote remove origin 2>$null
    Invoke-Git remote add origin $RemoteUrl
    # Use ONLY our env-based credential helper (empty value resets any inherited GCM helper)
    Invoke-Git config --local --replace-all credential.helper ''
    Invoke-Git config --local --add credential.helper "!f() { echo username=$GitUser; echo `"password=`$GITHUB_PAT`"; }; f"

    # --- Adopt the current remote state as the base, WITHOUT touching working files ---
    Invoke-Git fetch --quiet origin
    Invoke-Git reset --mixed "origin/$Branch"

    if ($PullOnly) {
        # Restore files that exist on the remote but are missing locally; keep local edits.
        $missing = @(git ls-files --deleted)
        if ($missing.Count -gt 0) {
            # Quoted '--' and a splatted array: a bare -- is swallowed and an array arg is joined into one string.
            Invoke-Git checkout '--' @missing
            Write-Host "Pulled $($missing.Count) file(s) from origin/$Branch."
        }
        $modified = @(git diff --name-only)
        if ($modified.Count -gt 0) {
            Write-Host "Kept your local edits (remote NOT pulled) for:" -ForegroundColor Yellow
            $modified | ForEach-Object { Write-Host "  $_" }
        }
        if ($missing.Count -eq 0 -and $modified.Count -eq 0) {
            Write-Host "Already up to date with origin/$Branch." -ForegroundColor Green
        }
    }
    else {
        Invoke-Git add -A
        $staged = git diff --cached --name-only
        if ($staged -match '(^|/)\.env$') {
            throw ".env is staged for commit - refusing to push (add it to .gitignore)."
        }
        if ($staged) {
            if ([string]::IsNullOrWhiteSpace($Message)) {
                $Message = "sync: $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
            }
            Invoke-Git commit -m $Message
            Invoke-Git push origin $Branch
            Write-Host "`nPushed to origin/$Branch." -ForegroundColor Green
        }
        else {
            Write-Host "No local changes - origin/$Branch already up to date." -ForegroundColor Green
        }
    }
}
finally {
    # Remove the throwaway git repo so NO git repository persists on this device.
    if (-not $KeepGit -and (Test-Path -LiteralPath $gitDir)) {
        Remove-Item -LiteralPath $gitDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    Remove-Item Env:\GITHUB_PAT -ErrorAction SilentlyContinue
}
