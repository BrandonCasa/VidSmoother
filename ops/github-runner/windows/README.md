# VidSmoother Windows Docker GitHub Runner

This creates a repository-level GitHub Actions self-hosted runner inside a Windows Docker container.

The current Docker context on this machine is `desktop-linux`, so build and run these files only after Docker Desktop is switched to Windows containers.

## 1. Switch Docker Desktop

In Docker Desktop, choose **Switch to Windows containers**. Confirm:

```powershell
docker info --format '{{.OSType}}'
```

Expected output:

```text
windows
```

## 2. Create the runner credential

Create a GitHub personal access token that can create repository runner registration tokens for `BrandonCasa/VidSmoother`.

Fine-grained token:

- Repository access: `BrandonCasa/VidSmoother`
- Repository permissions: **Administration: Read and write**

Classic token:

- Private repo: `repo`
- Public repo only: `public_repo`

## 3. Start the runner

```powershell
cd ops\github-runner\windows
Copy-Item .env.example .env
notepad .env
docker compose up -d --build
docker compose logs -f github-runner
```

The container fetches a short-lived GitHub registration token when it starts, registers itself, runs jobs, and removes the runner on shutdown.

## 4. Point workflows at it

Use this runner from a workflow with:

```yaml
runs-on: [self-hosted, Windows, X64, vidsmoother-windows-docker]
```

The existing `.github/workflows/windows-release.yml` currently uses `windows-2025`, which still targets GitHub-hosted Windows runners.
