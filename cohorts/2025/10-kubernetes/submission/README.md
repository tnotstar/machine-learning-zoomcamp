# Homework 10: Kubernetes & ONNX Model Serving

This folder contains the solution for Homework 10 of the [Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) (2025 Cohort).

## Objective

The goal of this homework is to deploy a machine learning model using Kubernetes. This involves containerizing the model, creating Kubernetes deployment and service manifests, and deploying them to a local Kubernetes cluster (e.g., using Kind).

## Files Description

- `deployment.yaml`: The Kubernetes deployment configuration file defining the replicas and container specs.
- `service.yaml`: The Kubernetes service configuration to expose the deployment.
- `01_install.sh`: Script to install prerequisites (Kind, Kubectl).
- `02_prepare.sh`: Script to build the Docker image and load it into the Kind cluster.
- `03_run.sh`: Script to apply the Kubernetes manifests.
- `04_test.sh`: Script to run the test against the deployed model.

## How to Run

### Prerequisites

Ensure you have the following installed:

- Docker
- Kubectl
- Kind (or Minikube)

### Steps

1.  **Build the Docker image:**

    ```bash
    docker build -t zoomcamp-model:3.13.10-hw10 .
    ```

2.  **Create a Kind cluster (if not already running):**

    ```bash
    kind create cluster
    ```

3.  **Load the image into Kind:**

    ```bash
    kind load docker-image zoomcamp-model:3.13.10-hw10
    ```

4.  **Apply the Kubernetes manifests:**

    ```bash
    kubectl apply -f deployment.yaml
    kubectl apply -f service.yaml
    ```

5.  **Port Forwarding:**
    To access the service from your local machine (adjust ports as necessary):

    ```bash
    kubectl port-forward service/subscription 9696:80
    ```

6.  **Test the prediction:**
    Run the test script to send a request to the model:
    ```bash
    python test.py
    ```

## Homework Answers

- **Question 1:** 0.49999999999842815
- **Question 2:** kind version 0.30.0
- **Question 3:** Pod
- **Question 4:** ClusterIP
- **Question 5:** kind load docker-image
- **Question 6:** 9696
- **Question 7:** subscription
