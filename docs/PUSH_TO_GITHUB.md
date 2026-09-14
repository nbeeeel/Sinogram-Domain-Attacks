# Push this repository to GitHub

A local Git repository is included in the delivered bundle/archive history. To publish it under your own GitHub account or organization:

```bash
git remote add origin git@github.com:YOUR_ORG/projection-domain-ct-adversarial-robustness.git
git branch -M main
git push -u origin main
```

For HTTPS remotes, use the GitHub URL shown when creating an empty repository. Do not initialize the remote with a README, license, or .gitignore because those files already exist locally.

Before public release, choose the software license approved by all relevant authors/institutions and update `CITATION.cff` with final publication metadata/DOI when available.
