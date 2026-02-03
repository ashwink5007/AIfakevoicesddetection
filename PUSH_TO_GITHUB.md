# Push to GitHub: voiceshielddetection

Repo: **https://github.com/ashwink5007/voiceshielddetection**

## 1. Install Git (if needed)

- Download: https://git-scm.com/download/win  
- Install and **add Git to PATH**.  
- Close and reopen your terminal.

## 2. Run these commands

Open **PowerShell** or **Command Prompt**, then:

```powershell
cd "d:\MINI project"

git init
git add .
git status
git commit -m "VoiceShield: ML pipeline, web app, Backend & Front end"

git branch -M main
git remote add origin https://github.com/ashwink5007/voiceshielddetection.git

git push -u origin main
```

If the repo already has content (e.g. README), use:

```powershell
git pull origin main --allow-unrelated-histories
git push -u origin main
```

## 3. What gets pushed

- **Included:** `ml-service/` (code only), `Backend/`, `Front end/`, `README.md`, `.gitignore`, etc.  
- **Excluded by .gitignore:** `ml-service/data/` (audio files), `node_modules/`, `.env`, `__pycache__/`, `*.pkl`, `features/*.csv`.

## 4. Auth when pushing

- **HTTPS:** Git will ask for your GitHub username and a **Personal Access Token** (not your password).  
  Create a token: GitHub → Settings → Developer settings → Personal access tokens.  
- **SSH:** If you use SSH keys, set the remote to:  
  `git@github.com:ashwink5007/voiceshielddetection.git`
