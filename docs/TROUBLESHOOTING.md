# Troubleshooting

Common errors and fixes. Sorted by layer.

---

## Install & environment

### `No module named 'pip'` inside `.venv`
The venv was created without pip bootstrapped.
```bash
python -m ensurepip
python -m pip install -e .
```

### `ModuleNotFoundError: No module named 'duckdb'`
The venv is not active or the project was not installed.
```bash
.venv/Scripts/activate   # Windows
# or: source .venv/bin/activate   # macOS / Linux
pip install -e .
```

### IDE hints that `duckdb` / `openai` are "not installed"
The IDE is looking at the system Python, not `.venv`. Point it at `.venv/Scripts/python.exe` (or the Unix equivalent) in the interpreter settings.

---

## Data pipeline

### `FileNotFoundError: fleet_telemetry.parquet`
You skipped the generator step.
```bash
python -m app.run_all --fleet-size 50 --days 3
# or generate from the dashboard's "Synthetic Data" tab
```

### KPI table is empty
```bash
python -c "import duckdb; c=duckdb.connect('./data/mining.duckdb'); \
  print(c.execute('SELECT COUNT(*) FROM kpi').fetchone())"
```
If it returns `(0,)`, re-run `python -m app.run_all`.

### TE values look wrong (e.g., 90+ J/TH with only 5 miners)
Cooling allocation is proportional. With 5 miners the fixed 50 kW cooling divides by a tiny fleet and distorts TE. Use &geq; 25 miners for realistic numbers. This is a PoC calibration, not a formula issue.

---

## ML layer

### `ValueError: Invalid classes inferred from unique values of y. Expected: [0 1], got [0 2]`
XGBoost requires contiguous class labels. The `FailureClassifier` handles this with an internal encoder, but you hit this if:
- You trained before the encoder fix -- re-train with the current code.
- You only have one failure class represented. Raise `FAILURE_INJECTION_RATE` in `.env` (`0.10`--`0.15` is a good starting point).

### XGBoost says "training classes [0, 2] encoded as [0, 1]"
Not an error -- this is the encoder logging that the training run only saw a subset of the four canonical classes. Predictions map back correctly.

### LSTM Autoencoder training is stuck for 20+ minutes on CPU
It trains at batch size 1. Either:
1. Skip LSTM with `python -m app.run_all --skip-training` and train IF + XGBoost separately.
2. Run on a GPU machine.
3. Patch `app/models/anomaly_detector.py` to use `DataLoader(batch_size=32)`.

### Classifier accuracy is &lt; 20 % on failure rows
Most likely you have only one failure type in the data (class imbalance). The current generator round-robins failure types across the first N miners; with only 2 failures you only get 2 types. Bump `FAILURE_INJECTION_RATE=0.10` so at least one of each type (thermal, hashboard, PSU) is present.

---

## LLM / NVIDIA NIM

### `APITimeoutError: Request timed out`
You are calling `google/gemma-4-31b-it` which is slow / intermittently available on the NIM free tier. Switch to a faster model:
```bash
# in .env
LLM_MODEL=google/gemma-3-27b-it
```
Verified: `gemma-3-27b-it` responds in ~0.3 s, `gemma-3-4b-it` in ~2.6 s.

### `400 BadRequestError: "auto" tool choice requires --enable-auto-tool-choice`
Gemma endpoints on NIM do not support `tool_choice="auto"`. The default `FleetAgent.ask()` method uses a **context-preloaded** approach that does not trigger this error. If you specifically want tool-calling, use `ask_with_tools()` and switch to a model that supports it on NIM:
```bash
LLM_MODEL=meta/llama-3.1-70b-instruct
```

### Streamlit "NVIDIA_API_KEY is not set" warning even though it is in `.env`
Streamlit started before `dotenv` was loaded. Restart Streamlit after editing `.env`:
```bash
# kill the old process (Ctrl+C in its terminal)
streamlit run app/dashboard/dashboard.py
```

### Agent returns "I hit my tool-call budget"
`max_steps` reached without a terminal answer. Either the question is too vague or the tool chain is too deep. Rephrase the question or increase `max_steps` in `FleetAgent.__init__`.

---

## Dashboard

### `use_container_width` deprecation warnings spam the logs
Already fixed in the current code (replaced with `width="stretch"`). Pull the latest commit.

### "KPI table empty" banner on every tab
Use the **Synthetic Data** tab to seed the DB, or run `python -m app.run_all` in a terminal before launching Streamlit.

### Page loads are slow
Streamlit caches queries for 60 s. First load rebuilds the cache; subsequent loads are fast. Force-refresh with `Ctrl+Shift+R`.

---

## Decision Engine

### Commands return `reason: "rate_limited"`
By design: one command per device per 5 minutes. Wait or tweak `RATE_LIMIT_SECONDS` in `DecisionEngine` for testing only (do not relax in prod).

### AI recommendation of 10 % underclock gets clipped to 5 %
By design: `CLOCK_STEP_LIMIT = 0.05`. Large changes require operator confirmation (supervised mode). This is the Safety layer doing its job.

### `temp_throttle` fires before AI can react
By design: Safety is the top of the priority stack. AI recommendations are advisory and only applied when no safety rule is triggered.

---

## Tests

### `test_features.py` / `test_generator.py` fail to import `duckdb`
These tests need the full install (`pip install -e .`). The core safety tests (`test_config`, `test_safety`, `test_decision_engine`) run without heavy deps.

### `test_data_generation_defaults` fails with `assert 0.10 == 0.05`
Your `.env` overrides `FAILURE_INJECTION_RATE`. Already fixed: the test now accepts any value in `[0, 0.5]`. Pull the latest code.

---

## Git / repo

### `.env` showed up in `git status`
Verify the gitignore pattern matches:
```bash
git check-ignore -v .env
# -> .gitignore:17:.env  .env
```
If it is already tracked, untrack without deleting:
```bash
git rm --cached .env
```

### `uv.lock` (706 KB) in the repo
If you do not use `uv`, remove it:
```bash
git rm uv.lock
```
The project also ships `requirements.txt` and `pyproject.toml`, so `uv.lock` is redundant for pip users.

---

## "It worked yesterday"

```bash
# Nuke generated state and start fresh
rm -rf data/raw data/processed data/models data/mining.duckdb
rm -rf .ruff_cache .pytest_cache
python -m app.run_all --fleet-size 50 --days 3
```

If the problem survives that, open an issue in the repo with:
- Python version (`python -V`)
- OS and architecture
- Full traceback
- Contents of `.env` **with the API key redacted**
