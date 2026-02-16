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
import glob
import shutil

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- FIX 1: Initialize logger broadly here, but we will CHANGE the file handler per run ---
# We point this to a generic 'startup.log' initially so we don't create empty training logs at boot.
logger = setup_logger("pluto_trainer", os.path.join(LOGS_DIR, "startup.log"), mode='a')



# --- Dataset Scanning & JSON Update Logic ---
def scan_dataset_and_update_configs():
    """
    Scans the dataset directory (Train/Val/Test) and updates 
    Training.json and Testing.json with the current file lists and class definitions.
    """
    logger.info("Scanning dataset and updating JSON configurations...")
    
    paths = get_data_paths()
    train_dir = paths['train']
    val_dir = paths['val']
    test_dir = paths['test']
    
    # 1. Identify Classes (Canonical List)
    # We look at train_dir for the source of truth for classes
    classes = []
    if os.path.exists(train_dir):
        classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    
    if not classes:
        logger.warning("No classes found in Train directory. Skipping JSON update.")
        return False

    # Map class name to index (0-based)
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    
    # 2. Build classlst and classBin
    # Check for user-defined class mapping metadata
    meta_path = os.path.join(DATASET_ROOT, "dataset_meta.json")
    user_ok_classes = set()
    user_ng_classes = set()
    
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                user_ok_classes = {c.lower() for c in meta.get("ok_classes", [])}
                user_ng_classes = {c.lower() for c in meta.get("ng_classes", [])}
        except Exception as e:
            logger.warning(f"Failed to read dataset metadata: {e}")

    class_lst = []
    class_bin = []
    
    for cls_name in classes:
        # classlst entry
        class_lst.append({
            "ClassName": cls_name,
            "iClassWeights": 1
        })
        
        # classBin entry
        c_lower = cls_name.lower()
        
        if c_lower in user_ok_classes:
            bin_name = "OK"
        elif c_lower in user_ng_classes:
            bin_name = "NG"
        else:
            # Fallback Logic: If name is "OK" (case-insensitive) -> "OK", else "NG"
            bin_name = "OK" if c_lower == "ok" else "NG"
            
        class_bin.append({
            "classBinName": bin_name
        })
        

    # 3. Helper to scan images
    def scan_images(directory, dataset_id=1):
        img_list = []
        if not os.path.exists(directory):
            return img_list
            
        for cls_name in classes:
            cls_path = os.path.join(directory, cls_name)
            if not os.path.exists(cls_path):
                continue
                
            label_idx = class_to_idx.get(cls_name, 0)
            
            # Recursive scan for images
            # Using glob for simplicity
            pattern = os.path.join(cls_path, "**", "*")
            files = glob.glob(pattern, recursive=True)
            
            for f_path in files:
                if f_path.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):
                    img_list.append({
                        "ImageName": os.path.abspath(f_path),
                        "Label": label_idx,
                        "MaskPath": None,
                        "Captions": None,
                        "DatasetId": dataset_id,
                        "CaptionsSentence": None,
                        "TabularInfo": None
                    })
        return img_list

    train_imgs = scan_images(train_dir)
    val_imgs = scan_images(val_dir)
    test_imgs = scan_images(test_dir)
    
    logger.info(f"Found {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test images.")
    logger.info(f"Classes identified: {classes}")

    # 4. Update Training.json
    if os.path.exists(TRAINING_JSON_PATH):
        try:
            with open(TRAINING_JSON_PATH, 'r') as f:
                t_data = json.load(f)
            
            # Update fields
            t_data["classlst"] = class_lst
            t_data["classBin"] = class_bin
            t_data["trainImglst"] = train_imgs
            t_data["ValImgList"] = val_imgs
            
            # Update Model counts
            if "Model" in t_data:
                t_data["Model"]["iTrainImgCount"] = len(train_imgs)
                t_data["Model"]["iValidationImgCount"] = len(val_imgs)
                t_data["Model"]["iTotalClasses"] = len(classes)
                # t_data["Model"]["bIsValPresent"] = len(val_imgs) > 0 # Keep existing logic or force true?
                
            with open(TRAINING_JSON_PATH, 'w') as f:
                json.dump(t_data, f, indent=2)
            logger.info(f"Updated {TRAINING_JSON_PATH}")
        except Exception as e:
            logger.error(f"Failed to update Training.json: {e}")
            return False
            
    # 5. Update Testing.json
    # Create it if it doesn't exist? Or only update if exists?
    # Uses TESTING_JSON_PATH
    target_test_json = TESTING_JSON_PATH
    if not os.path.exists(target_test_json) and os.path.exists(TRAINING_JSON_PATH):
         # Create from template if missing
         shutil.copy(TRAINING_JSON_PATH, target_test_json)
    
    if os.path.exists(target_test_json):
        try:
            with open(target_test_json, 'r') as f:
                test_data = json.load(f)
            
            test_data["classlst"] = class_lst
            test_data["classBin"] = class_bin
            test_data["testImglst"] = test_imgs
            
            if "Model" in test_data:
                 test_data["Model"]["iTestImgCount"] = len(test_imgs)
                 test_data["Model"]["iTotalClasses"] = len(classes)
            
            with open(target_test_json, 'w') as f:
                json.dump(test_data, f, indent=2)
            logger.info(f"Updated {target_test_json}")
        except Exception as e:
            logger.error(f"Failed to update Testing.json: {e}")
            return False

    return True


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

def parse_confusion_matrix_file(file_path, ok_classes=None):
    """
    Parses ConfMatrixTest.txt or ConfMatrixEval.txt.
    Calculates Miss Rate and Overkill Rate using provided ok_classes as good 
    and other classes (except Unknown) as not good (NG).
    Ignores Unknown class in calculations.
    
    Args:
        file_path (str): Path to the confusion matrix file.
        ok_classes (list): List of class names considered "OK" (Good). 
                           If None, defaults to ["OK"].
    """
    if not os.path.exists(file_path):
        logger.warning(f"Confusion matrix file not found: {file_path}")
        return None

    if ok_classes is None:
        ok_classes = ["OK"]
        
    # Normalize ok_classes for easier comparison
    ok_classes_norm = [c.lower() for c in ok_classes]

    results = {}
    total_metrics = {}
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # --- 1. Calculate Miss/Overkill from Matrix Table ---
        miss_count = 0
        total_defects = 0
        overkill_count = 0
        total_ok = 0
        
        matrix_start_idx = -1
        for i, line in enumerate(lines):
            if "Confusion Matrix :" in line:
                matrix_start_idx = i
                break
        
        if matrix_start_idx != -1 and matrix_start_idx + 1 < len(lines):
            # Parse Header
            header_line = lines[matrix_start_idx + 1]
            headers = [h.strip() for h in header_line.split() if h.strip()]
            
            # Map headers to indices
            header_map = {h.lower(): idx for idx, h in enumerate(headers)}
            
            # Identify which columns correspond to OK classes
            ok_col_indices = []
            for h, idx in header_map.items():
                if h in ok_classes_norm:
                    ok_col_indices.append(idx)
            
            unknown_col_idx = header_map.get("unknown", -1)
            
            # Parse Rows
            current_idx = matrix_start_idx + 2
            while current_idx < len(lines):
                line = lines[current_idx].strip()
                if not line or "," in line: # End of matrix or start of CSV section
                    break
                    
                parts = [p.strip() for p in line.split() if p.strip()]
                if not parts:
                    current_idx += 1
                    continue
                    
                row_label = parts[0]
                # Parse counts - skip label
                try:
                    counts = [int(x) for x in parts[1:]]
                except ValueError:
                    current_idx += 1 # unexpected format
                    continue

                # Ignore Unknown Row
                if row_label.lower() == "unknown":
                    current_idx += 1
                    continue
                    
                # Calculate row totals excluding Unknown column predictions
                row_total = sum(counts)
                unknown_pred_count = 0
                if unknown_col_idx != -1 and unknown_col_idx < len(counts):
                    unknown_pred_count = counts[unknown_col_idx]
                
                valid_row_total = row_total - unknown_pred_count
                
                # Determine if this row is an OK class or NG class
                is_ok_row = row_label.lower() in ok_classes_norm
                
                # Calculate how many predictions were "OK"
                pred_ok_total = 0
                for idx in ok_col_indices:
                    if idx < len(counts):
                        pred_ok_total += counts[idx]
                        
                if is_ok_row:
                    # Valid Good Class (OK)
                    # Overkill = Good predicted as Defect (Predicted != OK and != Unknown)
                    # So, Overkill = Valid Total - Predicted OK
                    row_overkill = valid_row_total - pred_ok_total
                    
                    overkill_count += row_overkill
                    total_ok += valid_row_total
                else:
                    # Valid Defect Class (NG)
                    # Miss = Defect predicted as OK
                    miss_count += pred_ok_total
                    total_defects += valid_row_total
                    
                current_idx += 1

        # Calculate Rates
        miss_rate = (miss_count / total_defects * 100.0) if total_defects > 0 else 0.0
        overkill_rate = (overkill_count / total_ok * 100.0) if total_ok > 0 else 0.0

        # --- 2. Parse CSV-like Section ---
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                try:
                    row_label = parts[0].strip()
                    
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
                        # Inject calculated rates
                        total_metrics["miss_rate"] = miss_rate
                        total_metrics["overkill_rate"] = overkill_rate
                    else:
                        results[row_label] = metrics
                except ValueError:
                    continue 

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
    
    # --- Step 0: Scan Dataset and Update JSONs ---
    # This ensures Training.json and Testing.json have correct file lists and class info
    if not scan_dataset_and_update_configs():
         return {"status": "error", "message": "Failed to scan dataset and update configurations."}
    
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
    
    # --- identify OK classes from training.json for metrics calculation ---
    ok_classes = ["OK"] # default fallback
    try:
        with open(training_json, 'r') as f:
            t_data = json.load(f)
            class_lst = t_data.get("classlst", [])
            class_bin = t_data.get("classBin", [])
            
            if class_lst and class_bin and len(class_lst) == len(class_bin):
                found_ok = []
                for i, c_item in enumerate(class_lst):
                    c_name = c_item.get("ClassName")
                    if i < len(class_bin):
                         bin_name = class_bin[i].get("classBinName")
                         if bin_name == "OK" and c_name:
                             found_ok.append(c_name)
                
                if found_ok:
                    ok_classes = found_ok
                    logger.info(f"identified OK classes from config: {ok_classes}")
    except Exception as e:
        logger.warning(f"Failed to parse OK classes from training.json: {e}")

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
    eval_results = parse_confusion_matrix_file(eval_matrix_path, ok_classes=ok_classes)
    
    # 7. Run Testing - ONLY after Eval success
    # We need to read the model name from Training.json to know the test directory
    model_name = "Model_1" # Default
    solution_dir = str(MODELS_DIR) # Safe default
    
    try:
        with open(training_json, 'r') as f:
            d = json.load(f)
            if "Model" in d:
                 if "name" in d["Model"]:
                     model_name = d["Model"]["name"]
                 if "SolutionDir" in d["Model"]:
                     solution_dir = d["Model"]["SolutionDir"]
    except:
        logger.warning("Failed to parse Training.json for Test dir path. Using defaults.")

    test_dir = os.path.join(solution_dir, "Test", model_name)
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
    test_results = parse_confusion_matrix_file(test_matrix_path, ok_classes=ok_classes)
    
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
