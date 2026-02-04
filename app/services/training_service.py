import os
import copy
import time
import torch
import csv
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

# 1. Generate a unique timestamp for this training run
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")

# 2. Set dynamic Log Path with timestamp
LOG_FILENAME = f"training_log_{TIMESTAMP}.txt"
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILENAME)

# Initialize logger with the dynamic path
logger = setup_logger("pluto_trainer", LOG_FILE_PATH, mode='w')

def create_model(num_classes):
    # Standard ResNet18 for this project
    model = models.resnet18(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    r_loss, correct, total = 0.0, 0, 0
    # Note: Accepting 3 items from loader (img, lbl, path)
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
    """Get all predictions, true labels, and file paths from a data loader."""
    model.eval()
    all_preds = []
    all_labels = []
    all_paths = []
    
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
    """
    Calculate Miss Rate and Overkill Rate.
    Miss Rate: Percentage of actual defects (non-OK) that were classified as OK
    Overkill Rate: Percentage of actual OK items that were classified as defects (non-OK)
    """
    # Find OK class index (case-insensitive)
    ok_idx = None
    for idx, name in enumerate(class_names):
        if name.upper() == ok_class.upper():
            ok_idx = idx
            break
    
    if ok_idx is None:
        return None
    
    # Create binary classification: OK vs NOT-OK
    y_true_binary = (y_true == ok_idx).astype(int)  # 1 = OK, 0 = NOT-OK
    y_pred_binary = (y_pred == ok_idx).astype(int)  # 1 = OK, 0 = NOT-OK
    
    # Calculate counts
    miss_count = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    total_defects = np.sum(y_true_binary == 0)
    
    overkill_count = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    total_ok = np.sum(y_true_binary == 1)
    
    # Calculate rates
    miss_rate = (miss_count / total_defects * 100) if total_defects > 0 else 0.0
    overkill_rate = (overkill_count / total_ok * 100) if total_ok > 0 else 0.0
    
    return {
        "miss_rate": miss_rate,
        "miss_count": miss_count,
        "total_defects": total_defects,
        "overkill_rate": overkill_rate,
        "overkill_count": overkill_count,
        "total_ok": total_ok
    }

def print_confusion_matrix(cm, class_names):
    """Print confusion matrix in a formatted table."""
    logger.info("\n" + "="*60)
    logger.info("CONFUSION MATRIX")
    logger.info("="*60)
    
    # Header
    header = f"{'':>12} |"
    for name in class_names:
        header += f" {name:>8} |"
    logger.info(header)
    logger.info("-"*60)
    
    # Rows
    for i, true_class in enumerate(class_names):
        row = f"{true_class:>12} |"
        for j in range(len(class_names)):
            row += f" {cm[i, j]:>8} |"
        logger.info(row)
    logger.info("="*60)

def generate_misclassification_csv(y_true, y_pred, paths, class_names, mode="val"):
    """
    Generates a CSV file for wrongly detected images.
    Uses 'mode' to distinguish between training and inference files.
    """
    # 3. Use the mode prefix so files don't overwrite each other
    csv_filename = f"{mode}_misclassifications_{TIMESTAMP}.csv"
    csv_path = os.path.join(LOGS_DIR, csv_filename)
    
    try:
        with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Original", "Predicted", "File Path"])
            
            count = 0
            for i in range(len(y_true)):
                if y_true[i] != y_pred[i]:
                    writer.writerow([
                        class_names[y_true[i]], 
                        class_names[y_pred[i]], 
                        paths[i]
                    ])
                    count += 1
                    
        logger.info(f"Misclassification CSV generated: {csv_path} ({count} entries)")
        return csv_path
    except Exception as e:
        logger.error(f"Failed to generate CSV: {e}")
        return None

def evaluate_and_log_metrics(model, val_loader, class_names, ok_class="OK", mode="val"):
    """
    Evaluate model and log confusion matrix, metrics, and miss/overkill rates.
    Accepts 'mode' to pass down to CSV generator.
    """
    
    # Get predictions and paths
    y_true, y_pred, paths = get_predictions_and_labels(model, val_loader)
    
    # Generate Misclassification CSV with specific mode
    csv_path = generate_misclassification_csv(y_true, y_pred, paths, class_names, mode=mode)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Print confusion matrix
    print_confusion_matrix(cm, class_names)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    logger.info("\n" + "="*60)
    logger.info(f"{mode.upper()} METRICS")
    logger.info("="*60)
    logger.info(f"Accuracy: {accuracy:.2%}")
    
    # Per-class metrics
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    logger.info("\nPer-class Metrics:")
    for i, cls in enumerate(class_names):
        logger.info(f"  {cls:>10}: Precision={precision[i]:.2%}, Recall={recall[i]:.2%}, F1={f1[i]:.2%}")
    
    # Calculate and log miss/overkill rates
    rates = calculate_miss_overkill_rates(y_true, y_pred, class_names, ok_class)
    
    if rates:
        logger.info("\n" + "="*60)
        logger.info("MISS AND OVERKILL RATES")
        logger.info("="*60)
        logger.info(f"Miss Rate:     {rates['miss_rate']:.2f}% ({rates['miss_count']}/{rates['total_defects']})")
        logger.info(f"  -> {rates['miss_count']} defects incorrectly classified as {ok_class}")
        logger.info(f"Overkill Rate: {rates['overkill_rate']:.2f}% ({rates['overkill_count']}/{rates['total_ok']})")
        logger.info(f"  -> {rates['overkill_count']} {ok_class} items incorrectly classified as defects")
        logger.info("="*60)
    else:
        logger.info(f"\n[INFO] OK class '{ok_class}' not found. Skipping miss/overkill rate calculation.")
        logger.info(f"       Available classes: {class_names}")
    
    return {
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "miss_rate": rates['miss_rate'] if rates else None,
        "overkill_rate": rates['overkill_rate'] if rates else None,
        "csv_path": csv_path
    }

def train_core(params, ds_train, ds_val, epochs=300, min_epochs=30, patience=15):
    """Core training loop with Early Stopping and Direct File Logging."""
    # Handle both full datasets and Subset objects
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
    
    # Write a header to the log file to separate runs or phases
    try:
        with open(LOG_FILE_PATH, "a") as f:
            f.write(f"\n{'='*20} New Training Session {'='*20}\n")
    except Exception as e:
        print(f"Warning: Could not write header to log file: {e}")

    for epoch in range(epochs):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc = validate(model, val_loader, criterion)
        scheduler.step(v_loss)
        
        # Track best model
        if v_loss < best_loss:
            best_loss = v_loss
            best_acc = v_acc
            best_wts = copy.deepcopy(model.state_dict())
            triggers = 0
        else:
            triggers += 1
            
        # ---------------------------------------------------------
        # DIRECT LOG DUMP (Forces write to disk every epoch)
        # ---------------------------------------------------------
        log_entry = (
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | "
            f"Train Acc: {t_acc:.2%} | Val Acc: {v_acc:.2%} | "
            f"Patience: {triggers}/{patience}\n"
        )
        
        try:
            with open(LOG_FILE_PATH, "a") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Error writing to log file: {e}")
        # ---------------------------------------------------------

        # Standard logger (keep this for console output if needed)
        if (epoch + 1) % 5 == 0:
            logger.info(log_entry.strip())

        # Early Stopping
        if triggers >= patience and (epoch + 1) >= min_epochs:
            stop_msg = f"Early stopping triggered at epoch {epoch+1}\n"
            logger.info(stop_msg.strip())
            # Dump stop message to file too
            with open(LOG_FILE_PATH, "a") as f:
                f.write(stop_msg)
            break
            
    model.load_state_dict(best_wts)
    return model, best_acc

def run_automated_training(full_epochs=300):
    logger.info("Starting Auto-ML Training...")
    paths = get_data_paths()
    train_dir = paths["train"]
    val_dir = paths["val"]
    
    if not os.path.exists(train_dir): return {"status": "error", "message": "No train dir"}
    
    ds_train = CustomImageDataset(train_dir, transform=train_transform)
    ds_val = CustomImageDataset(val_dir, transform=val_transform)
    
    # 1. Hyperparameter Search (Optuna)
    best_params = {"lr": 0.001, "batch_size": 32}
    
    try:
        def objective(trial):
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            bs = trial.suggest_categorical("batch_size", [16, 32])
            
            # Use larger subset for difficult datasets
            subset_size = min(len(ds_train), 1500)
            subset = torch.utils.data.Subset(ds_train, range(subset_size))
            
            _, acc = train_core({"lr": lr, "batch_size": bs}, subset, ds_val, epochs=20, min_epochs=10, patience=5)
            return acc

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1)
        best_params = study.best_params
        logger.info(f"Best Params Found: {best_params}")
    except Exception as e:
        logger.warning(f"Optuna search failed/skipped: {e}. Using defaults.")

    # 2. Final Training
    model, val_acc = train_core(best_params, ds_train, ds_val, epochs=full_epochs, min_epochs=30, patience=20)

    # 3. Evaluate and log comprehensive metrics
    logger.info("\n" + "="*60)
    logger.info("FINAL MODEL EVALUATION")
    logger.info("="*60)
    
    # Create validation loader for evaluation
    val_loader = DataLoader(ds_val, batch_size=best_params.get("batch_size", 32), shuffle=False)
    
    # Detect OK class
    ok_class = "OK" 
    if ds_val.classes:
        for cls in ds_val.classes:
            if cls.upper() == "OK":
                ok_class = cls
                break
    
    # Pass mode="training" to distinguish the CSV
    eval_results = evaluate_and_log_metrics(
        model, 
        val_loader, 
        ds_val.classes, 
        ok_class=ok_class, 
        mode="training"
    )
    
    # Save model
    save_path = os.path.join(MODELS_DIR, "best_model.pth")
    tmp_path = save_path + ".tmp"
    torch.save(model.state_dict(), tmp_path)
    os.replace(tmp_path, save_path)
    
    logger.info(f"\nModel saved to: {save_path}")
    
    return {
        "status": "completed", 
        "accuracy": val_acc, 
        "params": best_params,
        "class_names": ds_train.classes,
        "confusion_matrix": eval_results.get("confusion_matrix"),
        "miss_rate": eval_results.get("miss_rate"),
        "overkill_rate": eval_results.get("overkill_rate"),
        "log_path": LOG_FILE_PATH,
        "csv_path": eval_results.get("csv_path"),
        "data_path": DATASET_ROOT
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