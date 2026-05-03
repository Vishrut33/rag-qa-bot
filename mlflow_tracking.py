import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("rag-qa-bot-experiments")

configs = [
    (300,  50,  3),
    (500,  100, 3),
    (500,  100, 5),
    (1000, 200, 3),
    (1000, 200, 5),
]

for chunk_size, overlap, top_k in configs:
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("chunk_overlap", overlap)
        mlflow.log_param("top_k", top_k)
        mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
        
        # Simulate metrics
        import random
        random.seed(chunk_size + top_k)
        precision = round(random.uniform(0.70, 0.95), 3)
        latency   = round(chunk_size * 0.8 + top_k * 120, 1)
        
        # Log metrics
        mlflow.log_metric("retrieval_precision", precision)
        mlflow.log_metric("latency_ms", latency)
        
        print(f"chunk={chunk_size}, k={top_k} → precision={precision}")

print("\nDone. Open http://localhost:5000")