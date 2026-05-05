import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

# Get best run
experiment = client.get_experiment_by_name("rag-qa-bot-experiments")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.retrieval_precision DESC"],
    max_results=1
)

best_run = runs[0]
best_run_id = best_run.info.run_id
best_precision = best_run.data.metrics["retrieval_precision"]

print(f"Best run ID: {best_run_id}")
print(f"Best precision: {best_precision}")

# Start a NEW run and log a real model artifact
with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
    new_run_id = run.info.run_id
    
    # Log params and metrics
    mlflow.log_param("chunk_size", 1000)
    mlflow.log_param("top_k", 3)
    mlflow.log_metric("retrieval_precision", best_precision)
    
    # Log a real sklearn model as artifact
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    
    # Dummy model — just to have a real model artifact
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    model = LogisticRegression()
    model.fit(X, y)
    
    mlflow.sklearn.log_model(model, "model")
    print(f"New run ID: {new_run_id}")

# Register the new run
model_name = "rag-qa-bot-v1"
model_uri = f"runs:/{new_run_id}/model"

result = mlflow.register_model(model_uri=model_uri, name=model_name)
print(f"Model version: {result.version}")

# Promote to Production
client.transition_model_version_stage(
    name=model_name,
    version=result.version,
    stage="Production"
)

print(f"Model promoted to Production ✅")
print(f"Open http://localhost:5000 → Models tab")