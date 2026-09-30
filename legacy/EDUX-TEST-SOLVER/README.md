# EDUX Test Brute Force

This tool selects answers for the EDUX test based on answers.txt.

## Setup

```bash
cd d:\CODE\quizz-slayers\EDUX-TEST-BRUTEFORCE
python -m pip install -r requirements.txt
python -m playwright install chromium
```

## Run (headed)

```bash
pytest -s --headed --browser chromium
```

## Notes

- Credentials are stored in .env after first run.
- Update answers in answers.txt (format: "1. A").
