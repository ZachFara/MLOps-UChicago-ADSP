#!/bin/bash

# Function to test if a port is responding
test_port() {
    local port=$1
    local service=$2
    local max_attempts=30
    local attempt=1

    echo "Testing $service on port $port..."
    
    while ! nc -z localhost $port; do
        if [ $attempt -eq $max_attempts ]; then
            echo "❌ $service failed to start after $max_attempts attempts"
            return 1
        fi
        echo "Attempt $attempt: Waiting for $service to start..."
        sleep 2
        ((attempt++))
    done
    
    echo "✅ $service is running on port $port"
    return 0
}

# Function to test MLflow
test_mlflow() {
    echo "Testing MLflow service..."
    docker compose -f docker/docker-compose.yml up -d mlflow
    test_port 5000 "MLflow"
    
    # Test MLflow API
    if curl -s http://localhost:5000/api/2.0/mlflow/experiments/list | grep -q "experiments"; then
        echo "✅ MLflow API is responding correctly"
    else
        echo "❌ MLflow API is not responding correctly"
    fi
}

# Function to test FastAPI
test_api() {
    echo "Testing API service..."
    docker compose -f docker/docker-compose.yml up -d api
    test_port 8000 "FastAPI"
    
    # Test FastAPI docs
    if curl -s http://localhost:8000/docs | grep -q "swagger"; then
        echo "✅ FastAPI docs are accessible"
    else
        echo "❌ FastAPI docs are not accessible"
    fi
}

# Function to test Streamlit
test_streamlit() {
    echo "Testing Streamlit service..."
    docker compose -f docker/docker-compose.yml up -d streamlit
    test_port 8501 "Streamlit"
    
    # Test Streamlit
    if curl -s http://localhost:8501 | grep -q "streamlit"; then
        echo "✅ Streamlit is accessible"
    else
        echo "❌ Streamlit is not accessible"
    fi
}

# Clean up function
cleanup() {
    echo "Cleaning up containers..."
    docker compose -f docker/docker-compose.yml down
}

# Main testing sequence
echo "Starting service tests..."

# Create necessary directories
mkdir -p data mlruns logs

# Test each service
test_mlflow
test_api
test_streamlit

# Offer to show logs
read -p "Would you like to see the logs for any service? (mlflow/api/streamlit/no): " show_logs

case $show_logs in
    mlflow)
        docker compose -f docker/docker-compose.yml logs mlflow
        ;;
    api)
        docker compose -f docker/docker-compose.yml logs api
        ;;
    streamlit)
        docker compose -f docker/docker-compose.yml logs streamlit
        ;;
    *)
        echo "Skipping logs..."
        ;;
esac

# Ask if user wants to clean up
read -p "Would you like to stop all services? (y/n): " cleanup_services

if [ "$cleanup_services" = "y" ]; then
    cleanup
else
    echo "Services are still running. Use 'docker compose -f docker/docker-compose.yml down' to stop them later."
fi 