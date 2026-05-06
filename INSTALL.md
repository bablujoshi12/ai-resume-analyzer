# AI Resume Analyzer — Installation (Windows)

This project is a Streamlit app with MySQL database.

## Prerequisites

- Python **3.9.x** (recommended)
- MySQL Server (running)
- Git (optional, for cloning)

## 1) Get the code

### Option A: Clone from GitHub

```bash
git clone https://github.com/bablujoshi12/ai-resume-analyzer.git
cd ai-resume-analyzer
```

### Option B: ZIP download

- Download ZIP → Extract → open extracted folder in terminal.

## 2) Create database

Create DB (once):

```sql
CREATE DATABASE cv;
```

You can also run the schema file:

- `schema.sql`

## 3) Configure DB credentials (optional)

By default, app uses:
- host: `localhost`
- user: `root`
- password: empty
- db: `cv`

If your MySQL credentials differ, set environment variables:

PowerShell:

```powershell
setx DB_HOST "localhost"
setx DB_USER "root"
setx DB_PASSWORD "your_password"
setx DB_NAME "cv"
```

Open a **new terminal** after `setx`.

## 4) Create & activate virtual environment

From project root:

```bash
python -m venv venvapp
venvapp\Scripts\activate
```

## 5) Install dependencies

```bash
cd App
pip install -r requirements.txt
pip install scikit-learn joblib scipy reportlab
```

## 6) Download NLP resources (required)

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
python -m spacy download en_core_web_sm
```

## 7) Run the app

```bash
cd App
streamlit run App.py
```

Open:
- `http://localhost:8501`

## Admin & User login notes

### Admin
- First time open **Admin** section → create first admin → then login.

### User
- Open **Analysis** → Signup/Login → then run analysis.
- **My Reports** shows only your saved analyses.

## Common issues

- **NLTK LookupError**: run step (6) again.
- **MySQL connection error**: ensure MySQL running + DB credentials correct.
- **Port already used**:

```bash
streamlit run App.py --server.port 8502
```

