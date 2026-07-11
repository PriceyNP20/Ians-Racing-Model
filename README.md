# Ian Racing Model

Ian Racing Model is a read-only UK horse-racing research dashboard. It imports daily UK racecards, rejects runners that do not match the requested date or optional course filter, excludes non-runners from scoring, and ranks eligible runners with Ian Formula V3.1.

This is a research and tracking tool, not an automated betting system. It contains no bet placement functionality.

## Features

- Streamlit dashboard with date, all-course UK view, optional course filter, and race selector
- Replaceable `RacingDataProvider` interface
- Mock provider so the app runs without API credentials
- The Racing API adapter with endpoint URLs and field mapping isolated in config
- SQLite audit storage for raw provider responses
- Configurable 100-point Ian Formula V3.1 weights
- Component-level score, confidence, data-quality flag, and explanation
- Red-flag deductions for trip, class, draw, handicap mark, going, absence, jumping errors, and short prices
- Selections tracker with one winner pick and one best each-way pick per race
- Winner-pick win rate and best-EW place-rate indicators once verified results are available
- CSV export
- Separate pages for today's cards, top selections, results, model performance, and settings
- Automated tests for core data and scoring rules

## Streamlit Cloud Deploy

Use these settings:

```text
Repository: PriceyNP20/Ians-Racing-Model
Branch: main
Main file path: app/streamlit_app.py
```

For a demo deployment without API keys, add this in Streamlit secrets or leave the app on defaults:

```toml
RACING_DATA_PROVIDER = "mock"
DATABASE_URL = "sqlite:///ian_racing_model.db"
```

To use The Racing API, add:

```toml
RACING_DATA_PROVIDER = "the_racing_api"
RACING_API_BASE_URL = "https://api.theracingapi.com/v1"
RACING_API_REGION_CODES = "gb"
RACING_API_USERNAME = "your_username"
RACING_API_PASSWORD = "your_password"
DATABASE_URL = "sqlite:///ian_racing_model.db"
```

## Local Setup

Use Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

The default `.env` uses the mock provider:

```text
RACING_DATA_PROVIDER=mock
DATABASE_URL=sqlite:///ian_racing_model.db
```

## Launch Locally

```powershell
streamlit run app/streamlit_app.py
```

The mock sample is dated `2026-07-11` and includes Ascot and York runners. The sample also includes a wrong-date runner and a non-runner so validation rules are visible in tests. Course-specific validation is covered by requesting Ascot and confirming York is excluded from that filtered card.

## Run Tests

```powershell
pytest
```

## Project Structure

```text
app/
  streamlit_app.py
  pages/
sample_data/
  mock_racecard.json
src/ian_racing_model/
  config.py
  domain.py
  services.py
  ui.py
  providers/
  model/
  storage/
tests/
```

## Scoring Notes

Ian Formula V3.1 weights total 100:

| Component | Weight |
| --- | ---: |
| handicap_position | 18 |
| target_race_intent | 12 |
| pace_and_draw | 14 |
| course_suitability | 10 |
| distance_suitability | 10 |
| class_strength | 10 |
| current_performance | 10 |
| trainer_profile | 7 |
| jockey_suitability | 5 |
| market_value | 4 |

The MVP does not pretend unavailable data exists. When a component cannot be fully evidenced from the imported fields, it returns a lower-confidence neutral score with a data-quality warning and explanation.

Estimated fair odds are deliberately shown as a placeholder until verified results are imported and model calibration is possible.
