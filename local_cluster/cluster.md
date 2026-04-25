**Author:** Frederico Rocha Boller  
**Project:** Distributed Matrix Operations using CUDA and OpenMPI for Distributed Systems course 

## Overview
This repository contains the setup and source code for a hybrid, distributed GPU computing cluster. It connects two physical Windows machines running WSL. My first efforts were towards google's cloud gpu, but since I realized that I could use my old machine, I decided to build this. I used **OpenMPI** and **CUDA C++**. 

## Hardware Architecture
The cluster leverages asymmetric GPU hardware:
* **Node 0**: NVIDIA GeForce RTX 5060 Ti
* **Node 1**: NVIDIA GeForce GTX 1050 Ti

## Network Topology
I tried to connect them via lan to have minimal latency, but since my old machine had a Windows 10, which utilizes a Hyper-V Virtual Switch that places WSL2 behind a strict Double NAT, preventing standard peer-to-peer MPI communication, I decided to create a P2P tunnel using `tailscale`, which introduced some latency (about 40ms, but it depends on the server that it connects to).

## Data and Load Balancing decisions
Due to the constraints of the Tailscale tunnel (approx. 40ms latency), sending large data structures (like 1GB matrices) over the network would bottleneck the GPUs. My strategy here is to use an integer seed to generate the float numbers (that vary between -1 and 1) of the matrices based on their position indices. Also, due to asymmetric hardware, the GTX 1050 TI does less work than the RTX 5060 TI.

## Usage
The entire compilation, deployment, and execution pipeline is automated via `make`.

The automation architecture is split across specific files to ensure the main codebase remains portable and local environment variables stay private.

* **`cluster_config.mk`:**
  This file acts as the local environment registry. It stores the specific IP addresses and usernames for the worker nodes (e.g., the `100.x.x.x` Tailscale IP for the worker machine). Extracting these variables from the main scripts means the repository can be safely shared or cloned to a different network without hardcoded IPs breaking the build.
* **`hostfile`:**
  OpenMPI uses a hostfile to map the network topology and allocate slots (processes) to specific machines. Like the config file, this is specific to your local physical setup and should not be tracked by version control. 