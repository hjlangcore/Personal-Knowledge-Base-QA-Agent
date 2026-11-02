# Contributing Guide

Thanks for your interest in contributing to this project.

## How to Contribute

- Fork the repository and create your branch from `main`.
- Keep changes focused and small.
- Add clear commit messages.
- Run local checks before opening a PR.

## Local Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Set an API key in the environment. Bash:

   ```bash
   export DASHSCOPE_API_KEY="your_key"
   ```

   PowerShell:

   ```powershell
   $env:DASHSCOPE_API_KEY="your_key"
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

4. Run tests (no paid API key required):

   ```bash
   pip install -r requirements-dev.txt
   pytest
   ```

## Coding Standards

- Follow PEP 8 style for Python code.
- Use meaningful names for functions and variables.
- Keep functions short and single-purpose.
- Add comments only where logic is non-obvious.

## Pull Request Checklist

- [ ] Code runs locally without errors
- [ ] `pytest` passes
- [ ] No hardcoded secrets or API keys
- [ ] Documentation updated (if behavior changed)
- [ ] Changes are limited to the scope of the PR

## Security

- Never commit `.env`, API keys, or local vector DB data.
- If a key is exposed, revoke and regenerate it immediately.
