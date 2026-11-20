# IncidentIQ

An AI-powered incident intelligence platform. You give it application logs and system metrics as a CSV file. It finds the moments when something went wrong, guesses what kind of problem it was, says how serious it is, and suggests what to check first.

It runs on a laptop, needs no paid API and no internet access once installed, and works the moment you start it because it ships with synthetic demo data.

---

## 1. Problem statement

A modern app is made of many small services. Each one writes thousands of log lines and reports numbers like response time and CPU use. When something breaks at 2 a.m., an engineer has to scroll through all of it and work out what happened.

Three things make this slow:

- Fixed alert limits ("CPU over 90%") miss problems that are unusual but don't cross a limit.
- Log messages are free text, so they are hard to sort automatically.
- One real problem produces dozens of alerts, so people drown in noise.

IncidentIQ is a small answer to this. It uses machine learning to spot unusual behaviour and to read log text, then uses readable rules to group alerts into incidents, rate them, and explain them.

This field is called **AIOps** (AI for IT Operations).

## 2. What it does

1. Accepts a CSV of logs and metrics (or uses built-in demo data).
2. Finds unusual readings with **Isolation Forest**.
3. Classifies log messages as database, network, application, deployment or resource with **TF-IDF + Logistic Regression**.
4. Groups related events from the same service into one **incident**.
5. Gives each incident a **severity** (LOW, MEDIUM, HIGH, CRITICAL) using a points system you can read.
6. Suggests a **probable root cause**, shows the **evidence**, and lists **troubleshooting steps**.
7. Shows everything on a dark dashboard, with an incident detail page.

Every result is tagged **ML prediction**, **Rule-based** or **Data**, so you can always see where an answer came from. Causes are always called *probable*. The tool never claims a cause is confirmed.

## 3. Architecture and workflow

```
 CSV upload / demo data
          |
          v
  validation.py  -- checks columns, drops bad rows, writes notes
          |
          v
  ml.py: Isolation Forest  -- anomaly score + "unusual?" for every row
          |
          v
  intelligence.py: pick "events"  (unusual OR limit breached OR ERROR/CRITICAL log)
          |
          v
  group events (same service, gaps under 10 minutes)  -->  drop one-off blips
          |
          v
  ml.py: TF-IDF + Logistic Regression  -- category + confidence from log text
          |
          v
  rules: severity points, probable cause, evidence, recommended steps
          |
          v
  main.py (FastAPI)  -->  JSON  -->  dashboard (HTML + CSS + JavaScript)
```

### Folder structure

```
incidentiq/
  app/
    data_gen.py       synthetic dataset generator (seeded, reproducible)
    ml.py             trains, saves, loads and runs the two models
    intelligence.py   events -> incidents, severity, causes, evidence, steps
    validation.py     CSV checks and cleaning
    main.py           FastAPI endpoints
    static/           index.html, style.css, app.js (the dashboard)
  tests/              pytest tests
  data/sample_upload.csv   a ready-made CSV to try the upload button
  models/             saved models (created by training, not stored in git)
  Dockerfile, .dockerignore, requirements.txt, requirements-dev.txt
  render.yaml         Render deployment settings
  .github/workflows/ci.yml   GitHub Actions pipeline
```

## 4. The ML, in simple language

### Isolation Forest (anomaly detection)

Picture a crowd of dots on a page. Most dots sit in a tight cluster. A few sit far away on their own.

Isolation Forest keeps drawing random lines to split the dots apart. Dots in the crowd need many cuts before each one is alone. A lone dot far from the crowd is separated after only a few cuts. **Fewer cuts means more unusual.** That is the whole idea.

Here the "dots" are minutes of data, described by four numbers: latency, error rate, CPU and memory. We train it **only on normal minutes**, so it learns what normal looks like. Anything far from that gets a high anomaly score (roughly 0 to 1). The flag threshold learned in training is about 0.60.

Why it is a good fit: it needs no labelled examples of failures, and it looks at all four numbers together. A minute where CPU is 70% and latency is 600 ms might pass two separate limits but still be strange as a combination.

### TF-IDF (turning text into numbers)

A model cannot read words, so we convert each log line into numbers.

- **TF** (term frequency): how often a word appears in this message.
- **IDF** (inverse document frequency): how rare the word is across all messages. A word like "request" appears everywhere so it counts for little. A word like "deadlock" is rare so it counts for a lot.

We also use word pairs ("connection pool"), and we replace every number with the word `num`, because `8400ms` and `6200ms` mean the same thing to a classifier.

### Logistic Regression (classification)

It learns, from labelled examples, how much each word pushes a message towards each category. "Deadlock" pushes towards database. "DNS" pushes towards network. For a new message it outputs a probability for each of the five categories. The highest is the prediction, and its probability is the confidence.

Why it is a good fit: it is fast, small, needs no GPU, and you can inspect which words drive a decision.

### The rules (not ML)

Everything else is plain Python in `intelligence.py`:

- **Events:** a row counts if it is anomalous, crosses a limit (latency 800 ms, errors 5%, CPU 90%, memory 90%), or has an ERROR/CRITICAL log level.
- **Grouping:** events from one service less than 10 minutes apart become one incident.
- **Blip filter:** a single odd minute with no ERROR log and no limit crossed is ignored.
- **Severity points:** +1 if ML flags it, +1 if the score is far above threshold, +1 for each crossed limit, +1 or +2 for ERROR or CRITICAL logs, +1 or +2 for long duration. Total 8+ is CRITICAL, 6+ HIGH, 4+ MEDIUM, otherwise LOW.
- **Root cause:** keyword rules per category (for example "deadlock" gives "lock contention"), plus cross-checks against the metrics.
- **Low confidence:** if the classifier is under 40% sure, the incident is labelled *unclassified* and no cause is suggested.

## 5. Dataset

There is no real company data here. `app/data_gen.py` builds a synthetic dataset, and the same seed always produces the same data.

| Column | Meaning |
|---|---|
| timestamp | one reading per service per minute |
| service | api-gateway, auth-service, payment-service, order-service, inventory-db, notification-service |
| log_level | INFO, WARN, ERROR, CRITICAL |
| message | the log line |
| latency_ms | response time (normal about 120 ms) |
| error_rate | percent of failed requests (normal about 0.8%) |
| cpu_pct | CPU use (normal about 40%) |
| memory_pct | memory use (normal about 55%) |

- **Training set** (seed 42): 1,500 minutes x 6 services = 9,000 rows, with 25 injected incidents per category.
- **Hold-out test set** (seed 99): same recipe, different random numbers, used only for scoring.
- **Demo set** (seed 7): 3 hours, 1,080 rows, six planted incidents. One is still running at the end.

Normal minutes also include harmless things on purpose: occasional latency spikes and WARN messages like "Retrying request". The tool should not raise incidents for those.

CSV rules for uploads: all eight columns are required, percentages must be 0 to 100, at most 20,000 rows and 5 MB. Bad rows are dropped and reported, not hidden.

## 6. Train and run locally

You need Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python -m app.ml                   # generates data, trains both models, saves them in models/
uvicorn app.main:app --reload      # open http://localhost:8000
```

If you skip the training command, the app trains the models by itself on first start (a few seconds).

Try it: click **Load demo data**, or upload `data/sample_upload.csv`. API docs are at `http://localhost:8000/docs`.

Run the tests:

```bash
python -m pytest -q
```

### API

| Endpoint | What it does |
|---|---|
| `GET /health` | health check, returns `{"status":"ok"}` |
| `GET /api/demo` | analyses the demo dataset and returns summary, chart data and incidents |
| `POST /api/analyze` | upload a CSV (form field `file`), returns the same as above |
| `GET /api/incidents` | incidents from the latest analysis; filter with `?severity=HIGH&category=database` |
| `GET /api/model-info` | training details, hold-out scores and rule settings |

## 7. Docker

```bash
docker build -t incidentiq .
docker run --rm -p 8000:8000 incidentiq
# or on another port:
docker run --rm -e PORT=10000 -p 10000:10000 incidentiq
```

The models are trained while the image is built, so the container starts fast. It reads the `PORT` environment variable, which is what Render uses.

## 8. GitHub Actions and Render

**The pipeline** (`.github/workflows/ci.yml`):

1. `test` job: installs dependencies, trains the models, runs pytest.
2. `docker` job (only if tests pass): builds the image, starts it on port 10000, and checks that `/health` and `/api/demo` respond.
3. Render deploys only if those checks pass, because `render.yaml` sets `autoDeployTrigger: checksPass`.

**Steps to deploy**

1. Create an empty GitHub repository, then push this folder:
   ```bash
   git init
   git add .
   git commit -m "IncidentIQ first version"
   git branch -M main
   git remote add origin https://github.com/<your-username>/incidentiq.git
   git push -u origin main
   ```
2. Open the **Actions** tab on GitHub and wait for the CI run to turn green.
3. Go to [render.com](https://render.com), sign in with GitHub.
4. Click **New +**, then **Blueprint**, pick your repository, and click **Apply**. Render reads `render.yaml` and creates the service (Docker, free plan, health check `/health`, deploy after checks pass).
   - Doing it by hand instead: **New + > Web Service**, choose the repo, set **Language** to Docker, **Health Check Path** to `/health`, and **Auto-Deploy** to *After CI Checks Pass*.
5. Wait for the first build (a few minutes). When it says **Live**, open the `onrender.com` link.

No secrets or API keys are needed. On the free plan the service may sleep when idle, so open it a few minutes before your demo.

**Checklist for the deployed app**

- [ ] `https://<your-app>.onrender.com/health` shows `{"status":"ok","models_loaded":true}`
- [ ] The dashboard opens and shows "Demo data (synthetic)" and 7 incidents
- [ ] The charts move (live replay) and red dots appear during the spikes
- [ ] Clicking INC-001 opens a detail page with log message, score, cause, evidence and steps
- [ ] Uploading `data/sample_upload.csv` works
- [ ] Uploading a `.txt` file shows a clear error instead of crashing
- [ ] `/api/model-info` and `/docs` load
- [ ] The latest commit has a green check on GitHub and a "Live" deploy on Render

## 9. Example walkthrough: INC-001

The demo data contains a database problem on `inventory-db` starting at 09:35. Here is what the tool reports.

- **Detection:** nine minutes in a row had unusual readings. Peak latency was 2,409 ms (normal is about 122 ms) and error rate peaked at 14.2% (normal about 1%). The anomaly score reached 0.74 against a threshold of 0.60.
- **Original log:** `CRITICAL Connection pool exhausted: all 212 connections in use`
- **Classifier:** database, 96% confidence, averaged over nine log lines.
- **Severity: HIGH (7 points):** +1 ML flag, +1 score far above threshold, +1 latency over limit, +1 error rate over limit, +2 for a CRITICAL log, +1 for lasting at least 5 minutes.
- **Probable cause (rule):** the log text contains "connection pool", so the database connection pool is probably exhausted and new requests wait or fail.
- **Evidence:** the anomaly score, the classifier result, the keyword match, and the two metrics that crossed their limits. CPU stayed near 58%, so the machine itself was not overloaded.
- **First step suggested:** check how many connections are open and which service holds them.

Whether this is the real cause would still need a human to check. That is why the wording says "probably".

## 10. Limitations

- **The data is synthetic.** Messages come from fixed templates, so the classifier scored 100% on the hold-out set. That number is too good to apply to real logs, which are messier. Isolation Forest scored 91% precision and 99.6% recall on the same kind of data.
- The demo includes one false alarm on purpose: INC-002 is a cluster of harmless spikes. The classifier is only 37% sure, so it is labelled unclassified and rated LOW. No model is perfect.
- Root-cause rules are keyword based, so they only cover patterns someone wrote down.
- The latest result is kept in memory and shared by everyone using the server. Fine for a demo, not for many users.
- Models are not retrained when you upload data. Metrics are judged against the synthetic idea of "normal" (latency near 120 ms and so on), so a system with a different normal would need its own training data.
- No login, no database, no alert notifications.

## 11. Future improvements

- Train on real, labelled logs and measure accuracy honestly.
- Learn a separate "normal" for each service.
- Add a feedback button so engineers can mark a suggestion right or wrong, then retrain.
- Store incidents in a database so history survives restarts.
- Send alerts to email or chat.
- Try sentence embeddings for log classification and compare against TF-IDF.

---

## My notes (fill this in yourself)

Your examiner will ask about *your* experience, and only you can answer that. Write two or three honest lines for each:

- Which part was hardest to understand when you first read it, and how did you work it out?
- One thing you changed or tested yourself (for example, changing a threshold in `intelligence.py` and watching what happened to the severity).
- One thing you would do differently.

Also check your college's rules on AI assistance. If it needs to be declared, say plainly which parts were AI-assisted.
