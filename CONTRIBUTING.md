# Contributing

Thanks for contributing to fuzzmyagent.ai.

## Dev Setup

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Pull Requests

- Keep PRs small and focused.
- Include screenshots for UI changes.
- Add tests where practical.

## Reporting Bugs

Open an issue with:

- Steps to reproduce
- Expected vs actual behavior
- Logs / screenshots
- Your endpoint payload/response shape (redact secrets)
