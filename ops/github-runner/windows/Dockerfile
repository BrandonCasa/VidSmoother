# escape=`
FROM mcr.microsoft.com/powershell:lts-windowsservercore-ltsc2022

SHELL ["pwsh", "-NoLogo", "-NoProfile", "-Command", "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue';"]

ARG RUNNER_VERSION=2.333.1
ENV RUNNER_VERSION=${RUNNER_VERSION}

WORKDIR C:\actions-runner

RUN Set-ExecutionPolicy Bypass -Scope Process -Force; `
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; `
    Invoke-Expression ((New-Object Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1')); `
    choco feature disable --name showDownloadProgress; `
    choco install -y git 7zip; `
    Remove-Item -Recurse -Force C:\ProgramData\chocolatey\logs -ErrorAction SilentlyContinue

RUN Invoke-WebRequest `
      -Uri "https://github.com/actions/runner/releases/download/v$env:RUNNER_VERSION/actions-runner-win-x64-$env:RUNNER_VERSION.zip" `
      -OutFile C:\actions-runner\actions-runner.zip; `
    Expand-Archive -Path C:\actions-runner\actions-runner.zip -DestinationPath C:\actions-runner -Force; `
    Remove-Item C:\actions-runner\actions-runner.zip

COPY entrypoint.ps1 C:\entrypoint.ps1

ENV RUNNER_WORKDIR=C:\runner-work `
    RUNNER_NAME=vidsmoother-windows-docker `
    RUNNER_LABELS=vidsmoother-windows-docker `
    RUNNER_REPLACE=true

ENTRYPOINT ["pwsh", "-NoLogo", "-NoProfile", "-File", "C:\\entrypoint.ps1"]
