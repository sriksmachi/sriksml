# Copyright (c) Microsoft. All rights reserved.
# Gunicorn configuration file
import multiprocessing
from dotenv import load_dotenv

load_dotenv(override=True)
max_requests = 1000
max_requests_jitter = 50
timeout = 259200
log_file = "-"
bind = "0.0.0.0:8000"
worker_class = "uvicorn.workers.UvicornWorker"
workers = 1  # multiprocessing.cpu_count() * 2 + 1
