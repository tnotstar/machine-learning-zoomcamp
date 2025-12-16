#!/usr/bin/bash

git pull --rebase
  
kind create cluster
kubectl cluster-info --context kind-kind
kubectl get services
kind load docker-image zoomcamp-model:3.13.10-hw10

# code deployment-yaml
kubectl apply -f deployment.yaml
kubectl get pods

# code service.yaml
kubectl apply -f service.yaml
kubectl get svc

kubectl port-forward service/subscription 9696:80
