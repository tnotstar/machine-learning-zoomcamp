#!/usr/bin/bash

sudo apt update && sudo apt install -y kubectl &&
    kubectl version --client

go install sigs.k8s.io/kind@v0.30.0 &&
    kind --version
