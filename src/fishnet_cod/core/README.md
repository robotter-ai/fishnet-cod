# Fishnet

Fishnet stands for **Financial Signal Hosting NETwork**.

It is a Compute-over-Data (CoD) system that uses the distributed Aleph.im network as a substrate for computation.
It is a decentralized, peer-to-peer, and serverless system that allows users to run statistical computations on their
timeseries data without having to upload it to a centralized server.

This python module contains a common data model, built on the
[Aleph Active Record SDK (AARS)](https://github.com/aleph-im/active-record-sdk), that is being used by the Fishnet API
and Executor VMs. The data model is used to store and query:
- Timeseries & Datasets
- Algorithms
- Permissions
- Executions
- Results

Also contains the executor code for the Fishnet Executor VM. Right now it supports Pandas, but in the future it will
support other execution environments (e.g. PyTorch, Tensorflow).

## Roadmap

- [x] Basic message model
- [x] API for communicating with Fishnet system
  - [x] Basic CRUD operations
  - [x] Permission management
  - [x] Local VM caching
  - [ ] Signature verification of requests
  - [x] Discovery of other API instances
  - [x] Dedicated API deploy function
  - [ ] Timeslice distribution across Executor nodes
- [x] Executor VM
  - [x] Listens for Aleph "Execution" messages and executes them
  - [x] Uploads results to Aleph
  - [x] Pandas support
  - [x] Dedicated Executor deploy function
  - [ ] Distributed execution & aggregation
    - [x] Discovery of other Executor instances
    - [x] Uploading executors with metadata: assigned timeslice, code version
    - [ ] Deploy multiple executors
  - [ ] Different execution environments (e.g. PyTorch, Tensorflow)
  - [ ] GPU support
- [ ] Versioning and immutable VMs
  - [x] Automatic Versioning & Deprecation
  - [x] Version Manifest & Message metadata
  - [ ] Make all deployments immutable
