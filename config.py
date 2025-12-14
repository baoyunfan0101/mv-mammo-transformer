# config.py

import os
from dotenv import load_dotenv

# Detect runtime environment
if "COLAB_GPU" in os.environ or os.path.isdir("/content"):
    ENV = "colab"
elif os.path.exists("/home/ubuntu") or os.environ.get("AWS_EXECUTION_ENV"):
    ENV = "aws"
else:
    home = os.path.expanduser("~")
    if "yunfan" in home.lower() or "MacBook" in os.uname().nodename:
        ENV = "local_mac"
    else:
        ENV = "other"

# Use different .env files for different environments
if ENV == "colab":
    env_path = os.path.join(os.path.dirname(__file__), ".env.colab")
else:
    env_path = os.path.join(os.path.dirname(__file__), ".env")

if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    pass

# Project root: directory containing this config.py
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Raw data dir: directory containing the raw VinDr-Mammo dataset
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", os.path.join(PROJECT_ROOT, "raw_data"))

# RAW_DATA_DIR/
IMG_ROOT = os.path.join(RAW_DATA_DIR, "images")
BREAST_LEVEL_PATH = os.path.join(RAW_DATA_DIR, "breast-level_annotations.csv")
FINDING_PATH = os.path.join(RAW_DATA_DIR, "finding_annotations.csv")
METADATA_PATH = os.path.join(RAW_DATA_DIR, "metadata.csv")

# ./checkpoints
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ./configs
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")
os.makedirs(CONFIG_DIR, exist_ok=True)

# ./status.csv
if ENV == "colab":
    STATUS_PATH = os.path.join(PROJECT_ROOT, "status_colab.csv")
else:
    STATUS_PATH = os.path.join(PROJECT_ROOT, "status.csv")

# ./bbox.csv
BBOX_PATH = os.path.join(PROJECT_ROOT, "bbox.csv")

# ./data
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# ./src
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
