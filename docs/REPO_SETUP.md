# GitHub Repo Setup Checklist

## 1) Initialize and first commit
```powershell
git init
git add .
git commit -m "Initial commit: LM lab source, scripts, and docs"
```

## 2) Create remote and push
```powershell
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

## 3) Recommended repo settings
- Enable branch protection on `main`
- Require PR reviews for direct changes
- Enable Dependabot alerts
- Enable secret scanning

## 4) Large files
Keep model binaries and full corpora out of Git. If you need to publish them:
- Use GitHub Releases assets, or
- Use external object storage and link in README

## 5) Optional CI idea
Add a workflow that compiles `full_lm.c`, `word_lm.c`, and `data_prep.c` on Windows with MinGW.
