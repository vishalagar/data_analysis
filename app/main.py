import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import shutil
import zipfile
import io
import os
import threading

# Minimal Imports
from app.config import BASE_DIR, MODELS_DIR, get_data_paths, DATASET_ROOT
from app.services.agent_service import analyze_situation_and_decide
from app.services.data_service import apply_fix, get_dataset_stats
from app.services.training_service import run_automated_training, run_inference

app = FastAPI(title="AutoML Agent (n8n)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Dataset
paths = get_data_paths()
app.mount("/dataset", StaticFiles(directory=paths["dataset_root"]), name="dataset")

# --- State ---
training_state = {"status": "idle", "result": None, "error": None}
state_lock = threading.Lock()

# --- Models ---
class FixRequest(BaseModel):
    file_path: str
    action: str
    new_label: Optional[str] = None

class FixItem(BaseModel):
    file_path: str
    issue_type: str
    suggested_label: Optional[str] = None
    split: Optional[str] = None

class BatchFixRequest(BaseModel):
    status: Optional[str] = None
    total_issues_found: Optional[int] = None
    all_issues: List[FixItem]

class UploadPathRequest(BaseModel):
    file_path: str

# NEW: Model for Training Request
class TrainingRequest(BaseModel):
    dataset_path: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None

# --- Routes ---

@app.get("/api/status")
def get_status():
    """Polled by n8n to check system state."""
    return {"dataset": get_dataset_stats(), "training": training_state}

@app.get("/api/analyze")
def analyze():
    """Step 1 in n8n flow: Analyze dataset."""
    return analyze_situation_and_decide()

@app.post("/api/batch_fix")
def batch_fix(req: List[BatchFixRequest]):
    """Step 2 in n8n flow: Apply AI suggestions."""
    results = []
    for batch_item in req:
        for issue in batch_item.all_issues:
            s, m = apply_fix(issue.file_path, issue.suggested_label)
            results.append({
                "path": issue.file_path, 
                "success": s, 
                "message": m
            })
    
    success_count = sum(1 for r in results if r["success"])
    return {"status": "completed", "success": success_count, "results": results}


@app.post("/api/train")
def start_train(req: Optional[TrainingRequest] = None):
    """
    Step 3: Train model.
    Optional JSON Body:
    {
      "dataset_path": "/path/to/custom/dataset",
      "hyperparameters": { "lr": 0.005, "batch_size": 16, "epochs": 50 }
    }
    """
    # 1. Check if already running
    if training_state["status"] == "running": 
        raise HTTPException(400, "Already running")
    
    # Extract params if provided
    dataset_path = req.dataset_path if req else None
    custom_params = req.hyperparameters if req else None

    # 2. Update state to running
    with state_lock: 
        training_state.update({"status": "running", "result": None, "error": None})

    try:
        # 3. RUN TRAINING
        res = run_automated_training(
            custom_params=custom_params,
            custom_dataset_path=dataset_path
        )
        
        # 4. Success
        with state_lock: 
            training_state.update({"status": "completed", "result": res})
        
        return {"status": "completed", "result": res}

    except Exception as e:
        # 5. Failure
        with state_lock: 
            training_state.update({"status": "failed", "error": str(e)})
        raise HTTPException(500, detail=f"Training failed: {str(e)}")

@app.post("/api/inference")
def inference():
    """Runs inference on the validation/test set."""
    result = run_inference()
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    return result

@app.post("/api/upload")
def upload(req: UploadPathRequest):
    """Load dataset from a local file path (Zip only)."""
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")
    
    if not req.file_path.endswith(".zip"):
        raise HTTPException(status_code=400, detail="File must be a .zip archive")

    try:
        if os.path.exists(DATASET_ROOT):
            shutil.rmtree(DATASET_ROOT)
        os.makedirs(DATASET_ROOT)

        with zipfile.ZipFile(req.file_path, 'r') as z:
            z.extractall(DATASET_ROOT)
            
        return {
            "status": "success", 
            "message": f"Dataset extracted from {req.file_path}",
            "files_extracted": len(z.namelist())
        }

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="The provided file is not a valid zip archive.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@app.get("/api/download_csv")
def download_csv(file_path: str):
    """
    Downloads a CSV file given its file path.
    Useful for retrieving the 'misclassifications.csv' generated during training/inference.
    
    Usage: GET /api/download_csv?file_path=/absolute/path/to/logs/file.csv
    """
    # 1. Basic validation
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    # 2. Ensure it is a CSV (Security best practice)
    if not file_path.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files can be downloaded via this endpoint.")
        
    # 3. Return the file as a downloadable attachment
    return FileResponse(
        path=file_path, 
        filename=os.path.basename(file_path), 
        media_type='text/csv'
    )
    

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)