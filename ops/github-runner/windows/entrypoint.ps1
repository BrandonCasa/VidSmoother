$ErrorActionPreference = "Stop"

function Require-Env {
  param([Parameter(Mandatory)] [string] $Name)
  $value = [Environment]::GetEnvironmentVariable($Name)
  if ([string]::IsNullOrWhiteSpace($value)) {
    throw "Missing required environment variable: $Name"
  }
  return $value
}

function Get-RegistrationToken {
  param(
    [Parameter(Mandatory)] [string] $Repository,
    [Parameter(Mandatory)] [string] $Pat
  )

  $headers = @{
    Accept = "application/vnd.github+json"
    Authorization = "Bearer $Pat"
    "X-GitHub-Api-Version" = "2022-11-28"
    "User-Agent" = "VidSmoother-self-hosted-runner"
  }

  $uri = "https://api.github.com/repos/$Repository/actions/runners/registration-token"
  $response = Invoke-RestMethod -Method Post -Headers $headers -Uri $uri
  return $response.token
}

$repository = Require-Env "GITHUB_REPOSITORY"
$pat = Require-Env "GITHUB_PAT"
$runnerName = if ($env:RUNNER_NAME) { $env:RUNNER_NAME } else { "vidsmoother-windows-docker" }
$runnerWorkdir = if ($env:RUNNER_WORKDIR) { $env:RUNNER_WORKDIR } else { "C:\runner-work" }
$runnerLabels = if ($env:RUNNER_LABELS) { $env:RUNNER_LABELS } else { "vidsmoother-windows-docker" }
$replace = if ($env:RUNNER_REPLACE -eq "false") { $false } else { $true }
$repoUrl = "https://github.com/$repository"

New-Item -ItemType Directory -Force -Path $runnerWorkdir | Out-Null
Set-Location C:\actions-runner

$token = Get-RegistrationToken -Repository $repository -Pat $pat

$configArgs = @(
  "--unattended",
  "--url", $repoUrl,
  "--token", $token,
  "--name", $runnerName,
  "--work", $runnerWorkdir,
  "--labels", $runnerLabels
)

if ($replace) {
  $configArgs += "--replace"
}

Write-Host "Registering runner '$runnerName' for $repoUrl with labels: $runnerLabels"
& .\config.cmd @configArgs

try {
  & .\run.cmd
} finally {
  Write-Host "Removing runner '$runnerName'"
  try {
    $removeToken = Get-RegistrationToken -Repository $repository -Pat $pat
    & .\config.cmd remove --unattended --token $removeToken
  } catch {
    Write-Warning "Runner removal failed: $($_.Exception.Message)"
  }
}
