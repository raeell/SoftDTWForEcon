#/bin/bash

python3 main.py
uvicorn api.api:app --host "0.0.0.0"



