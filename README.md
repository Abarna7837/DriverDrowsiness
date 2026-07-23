# DriverDrowsiness

## Quickstart

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the app from the project root:

```powershell
python -m app.main
```

4. Check model accuracy on the held-out test split:

```powershell
python evaluate_model.py
```

5. Add more labeled training data from the webcam:

```powershell
python collect_training_data.py
```

While the window is open, press `o` to save an awake sample and `d` to save a drowsy sample.

Notes:
- If you prefer `python app/main.py`, the entrypoint already adds the project root to `sys.path`.
- If you see `ModuleNotFoundError: No module named 'src'`, make sure you run from the repository root or set `PYTHONPATH=.`.