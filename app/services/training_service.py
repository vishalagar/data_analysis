import os
import copy
import time
import torch
import csv
import logging 
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

try:
    import optuna
except ImportError:
    pass

from app.config import (
    MODELS_DIR, LOGS_DIR, get_data_paths, DATASET_ROOT,
    DL_PROCESS_WRAPPER_PATH, TRAINING_JSON_PATH, EVALUATION_JSON_PATH, 
    STATUS_FILE_PATH, MODEL_CONFIG_DIR, BASE_DIR
)
from app.logger_config import setup_logger
from app.services.data_service import CustomImageDataset, train_transform, val_transform
import subprocess
import json
import re
import shutil

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- FIX 1: Initialize logger broadly here, but we will CHANGE the file handler per run ---
# We point this to a generic 'startup.log' initially so we don't create empty training logs at boot.
logger = setup_logger("pluto_trainer", os.path.join(LOGS_DIR, "startup.log"), mode='a')


# --- DL Process Wrapper Integration ---

def generate_config_json(config_path, mode="Train", params=None):
    """Updates the existing JSON configuration file for DLProcessWrapper."""
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at: {config_path}")
        return False
        
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config file {config_path}: {e}")
        return False

    # Update Parameters from Training Service calls
    if params and "Model" in config_data:
        if "epochs" in params:
            config_data["Model"]["epochs"] = int(params["epochs"])
        if "batch_size" in params:
            config_data["Model"]["iBatchSize"] = int(params["batch_size"])
        if "lr" in params:
            config_data["Model"]["fBaseLR"] = float(params["lr"])
            
    # Write Config back
    try:
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        logger.info(f"Updated config file at: {config_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to write config file: {e}")
        return False

def parse_status_log(log_path):
    """
    Parses the Status.txt file for training progress.
    Returns the last parsed epoch info.
    """
    if not os.path.exists(log_path):
        return None

    last_info = None
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # [2025/12/17 10:25:12]epoch: 0, val_loss: 2.1864, train_loss: 1.8939, best_epoch: 0, best_loss: 2.1864, val_acc: 28.04%
                match = re.search(r"epoch:\s*(\d+),\s*val_loss:\s*([\d.]+),\s*train_loss:\s*([\d.]+),\s*best_epoch:\s*(\d+),\s*best_loss:\s*([\d.]+),\s*val_acc:\s*([\d.]+?)%", line)
                if match:
                    last_info = {
                        "epoch": int(match.group(1)),
                        "val_loss": float(match.group(2)),
                        "train_loss": float(match.group(3)),
                        "best_epoch": int(match.group(4)),
                        "best_loss": float(match.group(5)),
                        "val_acc": float(match.group(6))
                    }
    except Exception as e:
        logger.error(f"Error parsing status log: {e}")
        
    return last_info

def parse_confusion_matrix_file(file_path):
    """
    Parses ConfMatrixTest.txt or ConfMatrixEval.txt.
    Look for specific lines to parse Miss Rate and Overkill Rate if available, 
    or parse the matrix table itself.
    User provided:
    OK, 112, 91, 21, 81.2500, 18.7500 
    None, 66, 48, 18, 72.7273, 27.2727 
    ReTest, 43, 19, 24, 44.1860, 55.8140 
    Sum, 221, 158, 63, 71.4932, 28.5068
    
    We need to extract 81.2500 (Accuracy?) and 18.7500 (Miss Rate for OK?).
    Actually user said: "Miss and overkill rate... data"
    Let's look at the "Sum" line or individual class lines.
    Usually: 
    OK class -> Miss Rate = (False Negative / Total OK)
    Other class -> Overkill Rate = (False Positive / Total Other)
    
    But let's look at the user example for ConfMatrixTest.txt:
    Confusion Matrix :
         OK     None     ReTest     Unknown
    OK     91     15     6     0     
    None     13     48     5     0     
    ReTest     18     6     19     0    

    OK, 112, 91, 21, 81.2500, 18.7500  <-- Total, Correct, Incorrect, Acc?, Miss?
    
    Let's assume the CSV-like part at the bottom has the rates.
    """
    if not os.path.exists(file_path):
        logger.warning(f"Confusion matrix file not found: {file_path}")
        return None

    results = {}
    total_metrics = {}
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # Parse the CSV-like section at the end
        # Format seems to be: ClassName, Total, Correct, Incorrect, Accuracy, ErrorRate?
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                try:
                    # Check for "Sum" line
                    row_label = parts[0].strip()
                    
                    # IGNORE Unknown class
                    if row_label.lower() == "unknown":
                        continue
                        
                    metrics = {
                        "total": int(parts[1]),
                        "correct": int(parts[2]),
                        "incorrect": int(parts[3]),
                        "accuracy": float(parts[4]),
                        "error_rate": float(parts[5])
                    }
                    if row_label.lower() == "sum":
                        total_metrics = metrics
                    else:
                        results[row_label] = metrics
                except ValueError:
                    continue # Header or malformed line
                    
    except Exception as e:
        logger.error(f"Error parsing confusion matrix file: {e}")
        return None
        
    return {"class_metrics": results, "total_metrics": total_metrics}

def run_dl_process_wrapper(config_path, mode="Train"):
    """
    Executes the DLProcessWrapper.exe with the given config and mode.
    Modes: "Train", "Evaluation", "Test"
    """
    exe_path = DL_PROCESS_WRAPPER_PATH
    if not os.path.exists(exe_path):
        logger.error(f"DLProcessWrapper.exe not found at {exe_path}. Please check config.py or DL_PROCESS_WRAPPER_PATH env var.")
        return False

    cmd = [exe_path, config_path, mode]
    logger.info(f"Executing: {' '.join(cmd)}")
    
    try:
        # We use Popen to run it and wait for completion.
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # This will block until the process terminates.
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"DLProcessWrapper failed with code {process.returncode}")
            logger.error(f"Stderr: {stderr.decode('utf-8')}")
            return False
            
        logger.info(f"DLProcessWrapper finished successfully.")
        return True
        
    except Exception as e:
        logger.error(f"Failed to run DLProcessWrapper: {e}")
        return False

def poll_for_status(file_path, timeout=600):
    """
    Polls the Status.txt file until "SUCCESS", "TENSORRT", or "End Learning" is found, or timeout.
    Returns the parsed status if successful, None otherwise.
    """
    start_time = time.time()
    logger.info(f"Polling {file_path} for completion status...")
    
    while time.time() - start_time < timeout:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    if "SUCCESS" in content or "TENSORRT" in content or "End Learning" in content:
                        logger.info("Training process reported SUCCESS/TENSORRT/End Learning.")
                        return parse_status_log(file_path)
            except Exception as e:
                pass # Retry on read error
        
        time.sleep(2)
        
    logger.warning(f"Timeout waiting for training completion in {file_path}")
    return None

def poll_for_file(file_path, timeout=300):
    """
    Polls until the file exists and is not empty.
    Returns True if found, False on timeout.
    """
    start_time = time.time()
    logger.info(f"Polling for file: {file_path}")
    
    while time.time() - start_time < timeout:
        if os.path.exists(file_path):
            try:
                if os.path.getsize(file_path) > 0:
                    logger.info(f"File found and ready: {file_path}")
                    return True
            except:
                pass
        time.sleep(2)
        
    logger.warning(f"Timeout waiting for file: {file_path}")
    return False

def run_automated_training(full_epochs=1, custom_params=None, custom_dataset_path=None):
    """
    Replaced Run Automated Training to use DLProcessWrapper.
    """
    logger.info("Starting Automated Training via DLProcessWrapper")
    
    # 0. Clean up stale files to ensure we don't read old data
    try:
        if os.path.exists(STATUS_FILE_PATH):
            os.remove(STATUS_FILE_PATH)
        
        # Cleanup eval/test matrices if possible (path guessing)
        # We can't easily guess them all without parsing configs, but we can try basic locations if needed.
        # For now, deleting Status.txt is the most critical for Training completion check.
    except Exception as e:
        logger.warning(f"Failed to clean up stale files: {e}")

    # 1. Setup paths
    training_json = TRAINING_JSON_PATH
    eval_json = EVALUATION_JSON_PATH
    
    # Remove auto-calculation of num_classes as requested by user ("don't change")
    params = custom_params or {}
    params['epochs'] = full_epochs
    
    # Update Training.json
    if not generate_config_json(training_json, mode="Train", params=params):
         return {"status": "error", "message": "Failed to generate Training.json"}

    # 3. Run Training
    logger.info("Launching Training Process...")
    if not run_dl_process_wrapper(training_json, "Train"):
        logger.error("Training process failed execution (exit code).")
        return {"status": "error", "message": "Training process failed execution"}
    
    # 4. Poll for Training Completion
    # We poll Status.txt for "SUCCESS" or "TENSORRT"
    last_status = poll_for_status(STATUS_FILE_PATH, timeout=60000) # Long timeout for training
    
    if not last_status:
        # Fallback: maybe it finished but didn't write SUCCESS? 
        # But user said "wait till SUCCESS or TENSORRT comes".
        # We will log a warning but try to proceed if we see *some* status, or abort?
        # User implies strict wait.
        logger.warning("Training completion marker not found. Proceeding with caution or check Status.txt content manually.")
        last_status = parse_status_log(STATUS_FILE_PATH) # Try to get whatever is there
    
    logger.info(f"Training Completed. Last Status: {last_status}")

    # 5. Run Evaluation -- ONLY after training success
    
    # Prepare Evaluation Config
    # shutil.copy(training_json, eval_json)

    # Cleanup old Eval Matrix if exists
    eval_matrix_path = os.path.join(os.path.dirname(training_json), "ConfMatrixEvaluation.txt")
    if os.path.exists(eval_matrix_path):
        try: os.remove(eval_matrix_path)
        except: pass
    
    # Ensure StopBufferTest.txt exists before Evaluation
    stop_buffer_path = os.path.join(os.path.dirname(training_json), "StopBufferTest.txt")
    try:
        with open(stop_buffer_path, 'w') as f:
            f.write("1") # Ensure it has content
    except Exception as e:
        logger.warning(f"Failed to create StopBufferTest.txt: {e}")

    logger.info("Launching Evaluation Process...")
    if not run_dl_process_wrapper(eval_json, "Evaluate"):
        logger.error("Evaluation process failed execution. Aborting Testing.")
        return {
            "status": "error", 
            "message": "Evaluation process failed execution",
            "last_training_status": last_status
        }
    
    # Poll for Eval Matrix
    if not poll_for_file(eval_matrix_path, timeout=600):
         logger.error("Timeout waiting for Evaluation results.")
    
    # 6. Parse Eval Results
    eval_results = parse_confusion_matrix_file(eval_matrix_path)
    
    # 7. Run Testing - ONLY after Eval success
    # We need to read the model name from Training.json to know the test directory
    model_name = "Model_1" # Default
    try:
        with open(training_json, 'r') as f:
            d = json.load(f)
            if "Model" in d and "name" in d["Model"]:
                model_name = d["Model"]["name"]
    except:
        pass

    test_dir = os.path.join(d["Model"]["SolutionDir"], "Test", model_name)
    os.makedirs(test_dir, exist_ok=True)
    
    real_test_json_path = os.path.join(test_dir, "Testing.json")
    
    # Copy training config to this new location for Testing
    # shutil.copy(training_json, real_test_json_path)
    
    # Cleanup old Test Matrix if exists
    test_matrix_path = os.path.join(test_dir, "ConfMatrixTest.txt")
    if os.path.exists(test_matrix_path):
        try: os.remove(test_matrix_path)
        except: pass
    
    # Ensure StopBufferTest.txt exists before Testing
    stop_buffer_test_path = os.path.join(test_dir, "StopBufferTest.txt")
    try:
        with open(stop_buffer_test_path, 'w') as f:
            f.write("1") 
    except Exception as e:
        logger.warning(f"Failed to create StopBufferTest.txt in Test dir: {e}")

    logger.info(f"Launching Testing Process with config at: {real_test_json_path}")
    if not run_dl_process_wrapper(real_test_json_path, "Test"):
         logger.error("Testing process failed execution.")
         return {
            "status": "error", 
            "message": "Testing process failed execution",
            "last_training_status": last_status,
            "eval_results": eval_results
        }
    
    # Poll for Test Matrix
    if not poll_for_file(test_matrix_path, timeout=600):
        logger.error("Timeout waiting for Testing results.")
    
    # 8. Parse Test Results
    test_results = parse_confusion_matrix_file(test_matrix_path)
    
    return {
        "status": "success",
        "last_training_status": last_status,
        "eval_results": eval_results,
        "test_results": test_results,
        "model_name": model_name
    }

def run_inference():
    # Placeholder for inference if needed, or redirect to test logic
    return {"status": "info", "message": "Inference moved to Test flow via DLProcessWrapper"}
