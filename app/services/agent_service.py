# app/services/agent_service.py

from app.services.data_service import get_dataset_stats, detect_issues

def analyze_situation_and_decide():
    """
    Gather dataset statistics and potential issues.
    Returns raw data so the orchestration layer (n8n) can decide what to do.
    """
    # 1. Get High-level Stats (Count of images per split)
    stats = get_dataset_stats()
    
    # 2. Run Deep Analysis (Find label errors, outliers, etc.)
    # This might take a few seconds depending on dataset size
    detection_results = detect_issues()
    issues = detection_results.get("issues", [])
    
    
    # 3. Return Context for n8n
    return {
        "status": "analysis_completed",
        "dataset_summary": stats,
        "total_issues_found": len(issues),
        # "issues_preview": issues[:5],  # Send first 5 issues for context if needed
        "all_issues": issues           # Full list required if n8n wants to fix them immediately
    }

def diagnose_after_exploration(results):
    """
    Analyzes training results to see if we need to loop back.
    """
    best = results.get("best_result", {})
    val_acc = best.get("val_acc", 0.0)
    
    # Simple logic to pass back to n8n
    status = "success" if val_acc > 0.85 else "needs_improvement"
    
    return {
        "diagnosis": status, 
        "best_accuracy": val_acc,
        "details": "Model performed well." if status == "success" else "Model underperforming."
    }