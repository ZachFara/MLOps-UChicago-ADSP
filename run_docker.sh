#!/bin/bash

# Create necessary directories if they don't exist
mkdir -p data mlruns logs

# Build and start the containers
echo "Building and starting Docker containers..."
docker compose -f docker/docker-compose.yml up --build -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 10

echo "
MLOps Application is running in Docker!

Access the following services:
- Streamlit Dashboard: http://localhost:8501
- FastAPI Documentation: http://localhost:8000/docs
- MLflow UI: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (default credentials: admin/admin)

To view logs:
- All services: docker compose -f docker/docker-compose.yml logs -f
- Specific service: docker compose -f docker/docker-compose.yml logs -f [service_name]
  Available services: api, streamlit, mlflow, prefect, prometheus, grafana

To stop the application:
docker compose -f docker/docker-compose.yml down

To clean up everything (including volumes):
docker compose -f docker/docker-compose.yml down -v
"
# Make the script executable
chmod +x run_docker.sh 