# MLOps Project: Iris Classification API

This project demonstrates a minimal but complete MLOps pipeline for **building, tracking, packaging, deploying, and monitoring** a machine learning model for the **Iris dataset**.

**Course Event:**  
*"Build, Track, Package, Deploy and Monitor an ML Model using MLOps Best Practices"*

---

## Technologies Used
- **Version Control:** Git & GitHub  
- **Experiment Tracking:** MLflow  
- **API Framework:** FastAPI with Pydantic  
- **Containerization:** Docker & Docker Compose  
- **CI/CD:** GitHub Actions  
- **Logging & Monitoring:** Python logging module, Prometheus  

---

## Project Structure
```
.
├── .github/workflows/      # GitHub Actions CI/CD pipeline
├── data/raw/               # Raw dataset storage
├── mlruns/                 # MLflow experiment tracking artifacts and database
├── src/                    # Source code
│   ├── api/                # FastAPI application
│   ├── data/               # Data processing scripts
│   ├── training/           # Model training scripts
│   └── utils/              # Utility modules (e.g., logging)
├── tests/                  # Pytest tests
├── .dockerignore           # Files to ignore in Docker build
├── .flake8                 # flake8 linter configuration
├── .gitignore              # Git ignore file
├── docker-compose.yml      # Docker Compose config
├── Dockerfile              # Docker image definition
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── summary.md              # Project summary
```

---

## How to Run the Project (Recommended Method)

This project uses **Docker Compose** to create a consistent and portable environment for both training and serving the model.

### 1 Prerequisites
- Install [Docker](https://docs.docker.com/get-docker/) and Docker Compose.
- Clone this repository.

```bash
git clone <repo-url>
cd <repo-directory>
```

### 2 Prepare the Initial Dataset
```bash
python src/data/make_dataset.py
```

### 3 Run the MLOps Workflow with Docker Compose

#### Step A: Train the Model
```bash
docker-compose run --rm training
```

#### Step B: Run the API Server
```bash
docker-compose up api
```
API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

#### Step C: Stop the Services
```bash
docker-compose down
```

---

<details>
<summary><strong> Alternative: Manual Local Run Without Docker Compose</strong></summary>

**Local Setup:**
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**Generate Raw Data:**
```bash
python src/data/make_dataset.py
```

**Run Experiments and Register Model:**
```bash
# Terminal 1: MLflow UI
mlflow ui --backend-store-uri ./mlruns

# Terminal 2: Training
python src/training/train.py
```
View MLflow UI: [http://127.0.0.1:5000](http://127.0.0.1:5000)

**Run the API Locally:**
```bash
uvicorn src/api/main:app --host 0.0.0.0 --port 8000
```
Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

</details>

---

## CI/CD Pipeline
**File:** `.github/workflows/ci-cd.yml`

Pipeline steps:
1. **Lint & Test:** Runs `flake8` and `pytest`.
2. **Build & Push:** Builds the Docker image and pushes to Docker Hub.

**Secrets Required in GitHub:**
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

---

## API Endpoints
| Method | Endpoint     | Description                     |
|--------|--------------|---------------------------------|
| GET    | `/`          | Health check                    |
| GET    | `/docs`      | Swagger UI                      |
| POST   | `/predict`   | Make a prediction               |
| GET    | `/metrics`   | Prometheus metrics              |

---