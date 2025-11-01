#!/bin/bash

cd backend
source venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
uvicorn app:app --reload --host 0.0.0.0 --port 8000
