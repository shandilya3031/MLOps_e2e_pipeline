# MLOps Project Summary: Iris Classification

### 1. Project Overview
This project successfully implements a full MLOps lifecycle for a classification model using the Iris dataset. The primary goal was to automate the process from model training to deployment and monitoring, adhering to MLOps best practices. The final deliverable is a containerized REST API that serves predictions from the best-performing model.

### 2. Architecture Diagram
```
+----------+     +--------+     +-------------------+     +------------------+     +-------------+
|          |     |        |     |                   |     |                  |     |             |
|  GitHub  |---->| GitHub |---->|   Docker Hub      |---->| Docker Container |---->| End User    |
| (Code)   |     | Actions|     | (Image Registry)  |     | (Running API)    |     | (API Request)
|          |     | (CI/CD)|     |                   |     |                  |     |             |
+----------+     +--------+     +-------------------+     +------------------+     +-------------+
      ^              |
      |              | (Lint, Test, Build, Push)
      |              |
+-----+--------------+-----+
|                            |
|    Local Development       |
|  - VSCode                  |
|  - Data Prep (make_dataset.py)
|  - Model Training (train.py)
|  - MLflow Tracking (mlflow.db)
|                            |
+----------------------------+
```

### 3. Technology Choices
- **Dataset:** Iris (Classification)
- **Model Training:** Scikit-learn (Logistic Regression, Random Forest).
- **Experiment Tracking:** **MLflow** was used to log parameters, metrics, and model artifacts for each run. It provided a clear UI to compare models and its Model Registry was used to version and stage the production model.
- **API:** **FastAPI** was chosen for its high performance, automatic OpenAPI documentation, and native support for data validation via Pydantic, which fulfilled a bonus requirement.
- **Containerization:** **Docker** was used to package the FastAPI application, its dependencies, and the model-loading logic into a portable and isolated container.
- **CI/CD:** **GitHub Actions** automates the workflow. On every push to the `main` branch, it lints the code, runs tests, builds the Docker image, and pushes it to Docker Hub, ensuring continuous integration and delivery.
- **Monitoring:** Basic logging was implemented using Python's `logging` module to capture requests and predictions. For the bonus, a `/metrics` endpoint was exposed using `prometheus-fastapi-instrumentator` to provide application-level metrics like request latency and counts.

### 4. Workflow Walkthrough
1.  **Data Preparation:** The `src/data/make_dataset.py` script fetches and saves the raw data, making the data sourcing step reproducible.
2.  **Model Experimentation:** `src/training/train.py` trains multiple models. All experiments are tracked in MLflow. The script programmatically identifies the best model based on accuracy and registers it in the MLflow Model Registry, transitioning it to the "Production" stage.
3.  **API Service:** The FastAPI application in `src/api/main.py` loads the "Production" model from the MLflow Registry upon startup. It exposes a `/predict` endpoint that takes validated input and returns a prediction.
4.  **Deployment:** The `Dockerfile` defines the steps to create a production-ready image. The GitHub Actions pipeline automates the build and push to Docker Hub. The service can be deployed anywhere using a simple `docker run` command.

### 5. Conclusion & Future Improvements
This project provides a solid foundation for an MLOps pipeline.
- **What worked well:** The integration of MLflow, FastAPI, and Docker created a seamless and automated path from experiment to production.
- **Potential Improvements:**
    - **Continuous Deployment (CD):** The pipeline could be extended to automatically deploy the new Docker image to a cloud service like AWS ECS or a Kubernetes cluster.
    - **Advanced Monitoring:** Integrate Grafana with the Prometheus endpoint to create dashboards for visualizing model performance and API health over time.
    - **Model Retraining Trigger:** Implement a webhook or scheduled job that triggers the training pipeline on a schedule or when new data is detected.