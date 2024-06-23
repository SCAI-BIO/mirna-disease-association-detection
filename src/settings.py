import platform
import socket
import logging
from typing import Dict

# Defaults
home_base_dir = "/home/smadan"
project_dir = "/git/mirna-disease-association-detector-submission"
LOCAL_MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_TRACKING_URI = LOCAL_MLFLOW_TRACKING_URI
LOGGING_LEVEL = logging.INFO
EXTERNAL_LOGGING_LEVEL = logging.INFO
OTPUNA_STORAGE="sqlite:///" + home_base_dir + "/" + project_dir + "/example.db"

BASE_DATA_DIR = home_base_dir + project_dir + "/data"
BC5CDR_DATA_DIR = home_base_dir + project_dir + "/data/NER/bc5cdr_data"
MIRNA_DATA_DIR = home_base_dir  + project_dir + "/data/NER/miRNA_data"
MIRTEX_DATA_DIR = home_base_dir  + project_dir + "/data/NER/miRTex_data"
NCBI_DATA_DIR = home_base_dir  + project_dir + "/data/NER/ncbi_data"
MIRNA_DISEASE_ASSOC_DATA_DIR = home_base_dir + project_dir + "/data/RE/miRNA_disease_relations"
MULTI_TASK_ASSOC_DATA_DIR = home_base_dir + project_dir + "/data/RE/mt_relations"

DATA_DIR = BC5CDR_DATA_DIR # NCBI_DATA_DIR

BASE_MODEL_DIR = home_base_dir + project_dir + "/models"
#BERT_MODEL = BASE_MODEL_DIR + "/pretrained_biobert_pubmed_pmc"
TRAINED_MODEL_DIR = home_base_dir + project_dir + "/executions/trained_model"
TMP_DIR = home_base_dir + project_dir + "/tmp"
LABEL_MAP: Dict[int, str] = {0:""}
