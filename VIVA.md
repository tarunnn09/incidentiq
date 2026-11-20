# IncidentIQ: viva and demo guide

Read this out loud a few times in your own words. Don't memorise it word for word.

## The 1-minute explanation

"My project is called IncidentIQ. It helps engineers find and understand problems in IT systems.

Every app writes logs, which are text messages, and metrics, which are numbers like response time and CPU use. When something breaks, a person has to dig through all of it. My tool does the first pass automatically.

It uses two small machine-learning models. The first is Isolation Forest. It learns what normal metrics look like and gives a score to anything unusual. The second is TF-IDF with Logistic Regression. It reads a log message and guesses the type of problem: database, network, application, deployment or resource.

After that, plain Python rules take over. They group related alerts into one incident, give it a severity from LOW to CRITICAL, suggest a probable cause, show the evidence, and list the first things to check.

It is all built with Python, FastAPI and scikit-learn, packaged in Docker, tested with GitHub Actions and deployed on Render. It costs nothing to run because it does not call any paid AI service."

## 3-minute live demo

1. Open the dashboard. Point at the **System health** card: "It is Degraded because one incident is still running."
2. Point at the charts: "Red dots are readings the ML model found unusual. The dashed line is a fixed alert limit. Notice some red dots sit below the line. That is the point of ML: it catches odd combinations that a single limit misses."
3. Click **CRITICAL** in the severity box to filter, then clear it.
4. Open **INC-001**. Walk top to bottom: log message, anomaly score, predicted type, severity points, probable cause, evidence, steps. Show the tags: "Purple means the ML model predicted it. Green means a rule decided it."
5. Open **INC-002** and say: "This one is a false alarm. The classifier is only 37% sure, so the tool says unclassified and does not guess a cause. I left it in to show the limits."
6. Click **Upload CSV**, choose `data/sample_upload.csv`. Then upload a `.txt` file to show the error message.

## Questions you may get

**What is AIOps?**
Using AI and automation to monitor IT systems, detect problems and help fix them. It is mostly about cutting through the noise.

**What are logs and metrics?**
Logs are text messages a program writes. Metrics are numbers measured over time, like latency or CPU use.

**Why Isolation Forest?**
It doesn't need examples of failures, only normal data. It scores how easy a point is to separate from the rest. Easy to separate means unusual. It is fast and works on several numbers at once.

**Why train it on normal data only?**
In real systems, normal data is plentiful and failures are rare and varied. Learning "normal" lets us flag anything different, even a failure we have never seen.

**What is TF-IDF?**
A way to turn text into numbers. Words that are frequent in one message but rare overall, like "deadlock", get a high weight. Common words like "request" get a low weight.

**Why Logistic Regression for classification?**
It is simple, fast, works well on TF-IDF features and gives probabilities, which I use as confidence. I can also see which words matter.

**Why not use an LLM?**
Cost, speed and explainability. These small models run on a laptop for free, and every decision can be traced to a score or a rule. An LLM would need a paid API and can sound confident when it is wrong.

**How is severity decided?**
A points system, not ML. Points are added for ML flagging, for each crossed limit, for ERROR or CRITICAL logs and for long duration. 8 or more is CRITICAL, 6 to 7 HIGH, 4 to 5 MEDIUM, below that LOW. The page lists every point.

**How do you decide the root cause?**
Keyword rules per category plus checks against the metrics. Example: log says "connection pool" and the category is database, so the rule says the connection pool is probably exhausted. I always say "probable" because the tool cannot prove it.

**What is precision and recall?**
Precision: of the things I flagged, how many were really problems. Recall: of the real problems, how many I caught. On my synthetic test data, Isolation Forest had 91% precision and 99.6% recall.

**Your classifier got 100%. Isn't that suspicious?**
Yes, and I say so in the README. The data is synthetic and built from fixed message templates, so it is easy. Real logs would give lower accuracy. The honest number to quote is "100% on synthetic data, not proven on real data."

**What is the difference between an event and an incident?**
An event is one unusual row. An incident is a group of related events from the same service, less than 10 minutes apart. Grouping stops one problem from showing up as 20 alerts.

**How does the pipeline work?**
I push to GitHub. GitHub Actions installs dependencies, trains the models, runs the tests, builds the Docker image and checks `/health`. Render is set to deploy only after those checks pass.

**What would you improve?**
Use real labelled logs, learn a separate normal per service, add a feedback button, store incidents in a database and send alerts.

**What are the limitations?**
Synthetic data, keyword-based causes, one shared result in memory, and "normal" is fixed to the training data.

## Words to avoid

- Don't say the tool "finds the cause". Say it suggests a **probable** cause.
- Don't say it is "100% accurate". Say it scored 100% on **synthetic** hold-out data.
- Don't say it is production-ready. Say it is a college MVP.
