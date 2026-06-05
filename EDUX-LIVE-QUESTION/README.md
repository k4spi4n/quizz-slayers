# EDUX Live Question Solver

This tool answers live questions in the EDUX classroom by polling `context/active-round.txt`.

## Setup

```bash
cd EDUX-LIVE-QUESTION
python -m pip install -r requirements.txt
python -m playwright install chromium
```

## Run (headed)

```bash
pytest -s --headed --browser chromium
```
