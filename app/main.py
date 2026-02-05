import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import shutil
import zipfile
import io
import os
import threading

# Minimal Imports
from app.config import BASE_DIR, MODELS_DIR, get_data_paths, DATASET_ROOT
from app.services.agent_service import analyze_situation_and_decide
from app.services.data_service import apply_fix, get_dataset_stats
from app.services.training_service import run_automated_training,run_inference

app = FastAPI(title="AutoML Agent (n8n)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Dataset (Dynamic)
# We wrap this in a middleware or check on request? 
# For StaticFiles, it needs a directory at startup.
# We will point it to the ROOT dataset folder, so we can access /dataset/{dataset_name}/...
paths = get_data_paths() # Initial load
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
    issue_type: str          # CHANGED: Matches "issue_type" in your JSON
    suggested_label: Optional[str] = None
    split: Optional[str] = None

class BatchFixRequest(BaseModel):
    status: Optional[str] = None           # Added to handle JSON extra field
    total_issues_found: Optional[int] = None # Added to handle JSON extra field
    all_issues: List[FixItem]
class UploadPathRequest(BaseModel):
    file_path: str
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
def batch_fix(req: List[BatchFixRequest]):  # CHANGED: Accepts a List
    """Step 2 in n8n flow: Apply AI suggestions."""
    results = []
    
    # Loop through the list (usually contains 1 item from n8n)
    for batch_item in req:
        for issue in batch_item.all_issues:
            # Call the fix function
            s, m = apply_fix(issue.file_path, issue.suggested_label)
            results.append({
                "path": issue.file_path, 
                "success": s, 
                "message": m
            })
    
    success_count = sum(1 for r in results if r["success"])
    return {"status": "completed", "success": success_count, "results": results}


@app.post("/api/train")
def start_train():
    """Step 3: Train model and wait for result."""
    # 1. Check if already running
    if training_state["status"] == "running": 
        raise HTTPException(400, "Already running")
    
    # 2. Update state to running
    with state_lock: 
        training_state.update({"status": "running", "result": None, "error": None})

    try:
        # 3. RUN TRAINING DIRECTLY (No Threading)
        # The code stops here and waits until this function finishes.
        res = run_automated_training()
        
        # 4. Success: Update state and return result
        with state_lock: 
            training_state.update({"status": "completed", "result": res})
        
        return {"status": "completed", "result": res}

    except Exception as e:
        # 5. Failure: Update state and raise error
        with state_lock: 
            training_state.update({"status": "failed", "error": str(e)})
        raise HTTPException(500, detail=f"Training failed: {str(e)}")

@app.post("/api/inference")
def inference():
    """
    Runs inference on the validation/test set
    """
    result = run_inference()
    
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    return result

@app.post("/api/upload")
def upload(req: UploadPathRequest):
    """
    Load dataset from a local file path (Zip only).
    The file must exist on the server/container filesystem.
    """
    # 1. Validate file path
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")
    
    if not req.file_path.endswith(".zip"):
        raise HTTPException(status_code=400, detail="File must be a .zip archive")

    try:
        # 2. Clear existing dataset directory
        if os.path.exists(DATASET_ROOT):
            shutil.rmtree(DATASET_ROOT)
        os.makedirs(DATASET_ROOT)

        # 3. Extract the zip file from the path
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

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
