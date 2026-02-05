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

from app.config import MODELS_DIR, LOGS_DIR, get_data_paths, DATASET_ROOT
from app.logger_config import setup_logger
from app.services.data_service import CustomImageDataset, train_transform, val_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- FIX 1: Initialize logger broadly here, but we will CHANGE the file handler per run ---
# We point this to a generic 'startup.log' initially so we don't create empty training logs at boot.
logger = setup_logger("pluto_trainer", os.path.join(LOGS_DIR, "startup.log"), mode='a')

def create_model(num_classes):
    model = models.resnet18(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    r_loss, correct, total = 0.0, 0, 0
    for img, lbl, _ in loader:
        img, lbl = img.to(DEVICE), lbl.to(DEVICE)
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, lbl)
        loss.backward()
        optimizer.step()
        r_loss += loss.item() * img.size(0)
        _, pred = torch.max(out, 1)
        total += lbl.size(0)
        correct += (pred == lbl).sum().item()
    return (r_loss / total, correct / total) if total > 0 else (0.0, 0.0)

def validate(model, loader, criterion):
    model.eval()
    r_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for img, lbl, _ in loader:
            img, lbl = img.to(DEVICE), lbl.to(DEVICE)
            out = model(img)
            r_loss += criterion(out, lbl).item() * img.size(0)
            _, pred = torch.max(out, 1)
            total += lbl.size(0)
            correct += (pred == lbl).sum().item()
    return (r_loss / total, correct / total) if total > 0 else (0.0, 0.0)

def get_predictions_and_labels(model, loader):
    model.eval()
    all_preds, all_labels, all_paths = [], [], []
    with torch.no_grad():
        for imgs, labels, paths in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)
    return np.array(all_labels), np.array(all_preds), all_paths

def calculate_miss_overkill_rates(y_true, y_pred, class_names, ok_class="OK"):
    ok_idx = None
    for idx, name in enumerate(class_names):
        if name.upper() == ok_class.upper():
            ok_idx = idx
            break
    if ok_idx is None: return None
    
    y_true_binary = (y_true == ok_idx).astype(int)
    y_pred_binary = (y_pred == ok_idx).astype(int)
    
    miss_count = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    total_defects = np.sum(y_true_binary == 0)
    overkill_count = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    total_ok = np.sum(y_true_binary == 1)
    
    return {
        "miss_rate": (miss_count / total_defects * 100) if total_defects > 0 else 0.0,
        "miss_count": miss_count,
        "total_defects": total_defects,
        "overkill_rate": (overkill_count / total_ok * 100) if total_ok > 0 else 0.0,
        "overkill_count": overkill_count,
        "total_ok": total_ok
    }

def print_confusion_matrix(cm, class_names):
    logger.info("\n" + "="*60 + "\nCONFUSION MATRIX\n" + "="*60)
    header = f"{'':>12} |" + "".join([f" {name:>8} |" for name in class_names])
    logger.info(header + "\n" + "-"*60)
    for i, true_class in enumerate(class_names):
        row = f"{true_class:>12} |" + "".join([f" {cm[i, j]:>8} |" for j in range(len(class_names))])
        logger.info(row)
    logger.info("="*60)

# --- FIX 2: Accept timestamp argument so CSV matches the log file ---
def generate_misclassification_csv(y_true, y_pred, paths, class_names, mode="val", timestamp=None):
    ts = timestamp if timestamp else time.strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{mode}_misclassifications_{ts}.csv"
    csv_path = os.path.join(LOGS_DIR, csv_filename)
    try:
        with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Original", "Predicted", "File Path"])
            for i in range(len(y_true)):
                if y_true[i] != y_pred[i]:
                    writer.writerow([class_names[y_true[i]], class_names[y_pred[i]], paths[i]])
        return csv_path
    except Exception as e:
        logger.error(f"Failed to generate CSV: {e}")
        return None

# --- FIX 3: Accept timestamp argument ---
def evaluate_and_log_metrics(model, val_loader, class_names, ok_class="OK", mode="val", timestamp=None):
    y_true, y_pred, paths = get_predictions_and_labels(model, val_loader)
    
    # Pass timestamp to CSV generator
    csv_path = generate_misclassification_csv(y_true, y_pred, paths, class_names, mode=mode, timestamp=timestamp)
    
    cm = confusion_matrix(y_true, y_pred)
    print_confusion_matrix(cm, class_names)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    logger.info(f"\n{mode.upper()} METRICS\nAccuracy: {accuracy:.2%}")
    for i, cls in enumerate(class_names):
        logger.info(f"  {cls:>10}: Precision={precision[i]:.2%}, Recall={recall[i]:.2%}, F1={f1[i]:.2%}")
    
    rates = calculate_miss_overkill_rates(y_true, y_pred, class_names, ok_class)
    if rates:
        logger.info(f"\nMiss Rate: {rates['miss_rate']:.2f}% | Overkill Rate: {rates['overkill_rate']:.2f}%")
    
    return {
        "accuracy": accuracy, 
        "confusion_matrix": cm.tolist(), 
        "miss_rate": rates['miss_rate'] if rates else None, 
        "overkill_rate": rates['overkill_rate'] if rates else None, 
        "csv_path": csv_path
    }

# --- FIX 4: Accept log_file_path argument to write direct dumps to the correct file ---
def train_core(params, ds_train, ds_val, log_file_path, epochs=300, min_epochs=30, patience=15):
    # Note: log_file_path is kept in args for compatibility, but we use the global 'logger' 
    # which is already configured to write to this file.
    
    num_classes = len(ds_train.classes) if hasattr(ds_train, 'classes') else len(ds_train.dataset.classes)
    model = create_model(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params.get("lr", 0.001))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
    
    train_loader = DataLoader(ds_train, batch_size=params.get("batch_size", 32), shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=params.get("batch_size", 32), shuffle=False)
    
    best_loss = float('inf')
    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    triggers = 0
    
    # Write Header using logger
    logger.info(f"\n{'='*20} New Training Session {'='*20}")

    for epoch in range(epochs):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc = validate(model, val_loader, criterion)
        scheduler.step(v_loss)
        
        if v_loss < best_loss:
            best_loss, best_acc, best_wts = v_loss, v_acc, copy.deepcopy(model.state_dict())
            triggers = 0
        else:
            triggers += 1
            
        log_entry = (
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | "
            f"Train Acc: {t_acc:.2%} | Val Acc: {v_acc:.2%} | "
            f"Patience: {triggers}/{patience}"
        )
        
        # FIX: Use logger for everything. This ensures it goes to the file defined in the handler.
        # This prevents the "overwrite" issue caused by mixing manual file writes with logger streams.
        logger.info(log_entry)

        if triggers >= patience and (epoch + 1) >= min_epochs:
            stop_msg = f"Early stopping triggered at epoch {epoch+1}"
            logger.info(stop_msg)
            break
            
    model.load_state_dict(best_wts)
    return model, best_acc

def run_automated_training(full_epochs=1, custom_params=None, custom_dataset_path=None):
    # --- FIX: Generate timestamp and switch log file cleanly ---
    current_timestamp = time.strftime("%Y%m%d_%H%M%S")
    new_log_filename = f"training_log_{current_timestamp}.txt"
    new_log_path = os.path.join(LOGS_DIR, new_log_filename)
    
    # 1. Clear existing FileHandlers to stop writing to startup.log or old logs
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close() # Good practice to close the old stream
            
    # 2. Add the NEW FileHandler
    # mode='a' is safer than 'w' generally, but 'w' is fine here since the filename is unique.
    file_handler = logging.FileHandler(new_log_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    
    # 3. Force the console handler to be present if it's missing (optional safety)
    has_console = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    if not has_console:
        console_handler = logging.StreamHandler()
        logger.addHandler(console_handler)

    logger.info(f"Starting Training Session: {current_timestamp}")
    logger.info(f"Logging to: {new_log_path}")

    # Resolve Dataset Path
    if custom_dataset_path:
        if not os.path.exists(custom_dataset_path):
             return {"status": "error", "message": f"Custom dataset path not found: {custom_dataset_path}"}
        train_dir = os.path.join(custom_dataset_path, "train")
        val_dir = os.path.join(custom_dataset_path, "val")
        logger.info(f"Using custom dataset path: {custom_dataset_path}")
    else:
        paths = get_data_paths()
        train_dir = paths["train"]
        val_dir = paths["val"]
    
    if not os.path.exists(train_dir): return {"status": "error", "message": "No train dir"}
    
    ds_train = CustomImageDataset(train_dir, transform=train_transform)
    ds_val = CustomImageDataset(val_dir, transform=val_transform)
    
    # Resolve Hyperparameters
    final_params = {"lr": 0.001, "batch_size": 32}
    
    if custom_params:
        logger.info(f"Using custom hyperparameters: {custom_params}")
        final_params.update(custom_params)
    else:
        try:
            def objective(trial):
                lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
                bs = trial.suggest_categorical("batch_size", [16, 32])
                subset = torch.utils.data.Subset(ds_train, range(min(len(ds_train), 1500)))
                
                # Note: We still pass new_log_path, but train_core now ignores it for writing
                # and uses the global logger instead.
                _, acc = train_core({"lr": lr, "batch_size": bs}, subset, ds_val, new_log_path, epochs=1, min_epochs=1, patience=1)
                return acc

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=1)
            final_params = study.best_params
            logger.info(f"Best Optuna Params: {final_params}")
        except Exception as e:
            logger.warning(f"Optuna search failed: {e}. Using defaults.")

    # Final Training
    epochs = int(final_params.get("epochs", full_epochs))
    
    model, val_acc = train_core(final_params, ds_train, ds_val, new_log_path, epochs=epochs, min_epochs=1, patience=1)

    # Evaluation
    val_loader = DataLoader(ds_val, batch_size=int(final_params.get("batch_size", 32)), shuffle=False)
    
    ok_class = "OK"
    for cls in ds_val.classes:
        if cls.upper() == "OK": ok_class = cls; break

    eval_results = evaluate_and_log_metrics(
        model, 
        val_loader, 
        ds_val.classes, 
        ok_class=ok_class, 
        mode="training",
        timestamp=current_timestamp
    )
    
    save_path = os.path.join(MODELS_DIR, "best_model.pth")
    torch.save(model.state_dict(), save_path)
    
    # Ensure all file buffers are flushed at the end
    for handler in logger.handlers:
        handler.flush()

    return {
        "accuracy": val_acc, 
        "params": final_params,
        "class_names": ds_train.classes,
        "confusion_matrix": eval_results.get("confusion_matrix"),
        "miss_rate": eval_results.get("miss_rate"),
        "overkill_rate": eval_results.get("overkill_rate"),
        "log_path": new_log_path,
        "csv_path": eval_results.get("csv_path"),
        "train_dir" : train_dir,
        "val_dir" : val_dir     
        }

def run_inference():
    """Runs inference using the best saved model and returns metrics."""
    paths = get_data_paths()
    # You can change this to paths["test"] if you prefer testing on the test set
    data_dir = paths["test"] 
    
    if not os.path.exists(data_dir):
        return {"status": "error", "message": f"Directory not found: {data_dir}"}

    # Load Dataset
    dataset = CustomImageDataset(data_dir, transform=val_transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Load Model
    model_path = os.path.join(MODELS_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        return {"status": "error", "message": "No model found. Please train first."}
    
    model = create_model(len(dataset.classes)).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    # Detect OK class for Inference rates
    ok_class = "OK" 
    if dataset.classes:
        for cls in dataset.classes:
            if cls.upper() == "OK":
                ok_class = cls
                break

    # Pass mode="inference" to distinguish the CSV
    eval_results = evaluate_and_log_metrics(
        model, 
        loader, 
        dataset.classes, 
        ok_class=ok_class, 
        mode="inference"
    )
    
    return {
        "status": "success",
        "confusion_matrix": eval_results.get("confusion_matrix"),
        "accuracy": eval_results.get("accuracy"),
        "csv_path": eval_results.get("csv_path"),
        "class_names": dataset.classes
    }