import os
import shutil
import json
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock keys in config to avoid import errors if any
sys.modules['app.config'] = type('config', (object,), {
    'MODELS_DIR': os.path.abspath("temp_models"),
    'LOGS_DIR': os.path.abspath("temp_logs"),
    'BASE_DIR': os.path.abspath("."),
    'DATASET_ROOT': os.path.abspath("temp_dataset"),
    'DL_PROCESS_WRAPPER_PATH': "mock_dl_wrapper.exe",
    'get_data_paths': lambda: {
        'train': os.path.abspath("temp_dataset/train"), 
        'val': os.path.abspath("temp_dataset/val"), 
        'test': os.path.abspath("temp_dataset/test")
    }
})

# Mock 3rd party libs
sys.modules['torch'] = type('torch', (object,), {'device': lambda x: 'cpu', 'cuda': type('cuda', (object,), {'is_available': lambda: False})})
sys.modules['torch.nn'] = type('nn', (object,), {})
sys.modules['torch.optim'] = type('optim', (object,), {})
sys.modules['torch.utils'] = type('utils', (object,), {})
sys.modules['torch.utils.data'] = type('data', (object,), {'DataLoader': None})
sys.modules['torchvision'] = type('torchvision', (object,), {'models': type('models', (object,), {})})
sys.modules['torchvision.models'] = type('models', (object,), {})
sys.modules['numpy'] = type('numpy', (object,), {})
sys.modules['sklearn'] = type('sklearn', (object,), {})
sys.modules['sklearn.metrics'] = type('metrics', (object,), {
    'confusion_matrix': lambda: None,
    'accuracy_score': lambda: None,
    'precision_score': lambda: None,
    'recall_score': lambda: None,
    'f1_score': lambda: None
})
sys.modules['optuna'] = type('optuna', (object,), {})

# Mock app modules
sys.modules['app.logger_config'] = type('logger_config', (object,), {'setup_logger': lambda *args, **kwargs: logging.getLogger("test")})
sys.modules['app.services.data_service'] = type('data_service', (object,), {'CustomImageDataset': None, 'train_transform': None, 'val_transform': None})

# Import services
# We need to hack the import because app.config is mocked above, 
# but training_service imports it. 
# Actually, since we are running this as a script, we need to ensure app directory is in path.
sys.path.append(os.path.abspath("."))
from app.services import training_service

def setup_env():
    # Cleanups
    if os.path.exists("temp_models"): shutil.rmtree("temp_models")
    if os.path.exists("temp_dataset"): shutil.rmtree("temp_dataset")
    if os.path.exists("temp_logs"): shutil.rmtree("temp_logs")
    
    os.makedirs("temp_models")
    os.makedirs("temp_dataset/train/classA")
    os.makedirs("temp_dataset/val/classA")
    os.makedirs("temp_dataset/test/classA")
    os.makedirs("temp_logs")
    
    # Create dummy images
    with open("temp_dataset/train/classA/img1.jpg", "w") as f: f.write("img")
    
    # Create training.json template
    with open("training.json", "w") as f:
        json.dump({"Model": {"name": "TemplateModel"}}, f)

def verify_get_latest():
    logger.info("Verifying get_latest_model_dir...")
    
    # Case 1: Empty -> should create Model_1
    name, path = training_service.get_latest_model_dir()
    assert name == "Model_1"
    assert "Model_1" in path
    assert os.path.exists(path)
    logger.info("Case 1 Passed: Model_1 created.")
    
    # Case 2: Model_1 exists -> should return Model_1
    name, path = training_service.get_latest_model_dir()
    assert name == "Model_1"
    logger.info("Case 2 Passed: Model_1 returned.")

def verify_create_new_version():
    logger.info("Verifying create_new_model_version...")
    
    # Create Model_1 Training.json to test copying
    m1_path = os.path.join("temp_models", "Model_1")
    with open(os.path.join(m1_path, "Training.json"), "w") as f:
        json.dump({"Model": {"name": "Model_1", "some_setting": 123}}, f)
        
    # Create initial Testing.json in a nested path to simulate real structure?
    # Or just flat in Model_1 as per current logic? Current logic checks flat first.
    # Let's put it in Model_1/Testing.json
    with open(os.path.join(m1_path, "Testing.json"), "w") as f:
        json.dump({"Model": {"name": "Model_1_test"}}, f)
        
    # Seed Test/Model_1/Testing.json
    test_m1_path = os.path.join("temp_models", "Test", "Model_1")
    os.makedirs(test_m1_path, exist_ok=True)
    with open(os.path.join(test_m1_path, "Testing.json"), "w") as f:
         json.dump({"Model": {"name": "Model_1_test"}}, f)

    # Trigger creation
    new_name, new_path = training_service.create_new_model_version()
    
    assert new_name == "Model_2"
    assert os.path.exists(new_path)
    
    # Check if files copied
    t_json_path = os.path.join(new_path, "Training.json")
    assert os.path.exists(t_json_path)
    
    with open(t_json_path, "r") as f:
        data = json.load(f)
        assert data["Model"]["name"] == "Model_2"
        assert data["Model"]["some_setting"] == 123
        
    logger.info("Case 3 Passed: Model_2 created and config updated.")
    
    # Verify Testing.json in Test/Model_2 folder
    test_sub_dir = os.path.join("temp_models", "Test", "Model_2")
    # Our logic created it in get_latest... wait no, create_new_model_version creates it.
    assert os.path.exists(test_sub_dir)
    assert os.path.exists(os.path.join(test_sub_dir, "Testing.json"))
    
    # Verify Testing.json name update
    with open(os.path.join(test_sub_dir, "Testing.json"), "r") as f:
        data = json.load(f)
        assert data["Model"]["name"] == "Model_2"
        
    logger.info("Case 4 Passed: Testing folder created.")

def run_verification():
    setup_env()
    try:
        verify_get_latest()
        verify_create_new_version()
        print("VERIFICATION SUCCESSFUL")
    except Exception as e:
        print(f"VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        # shutil.rmtree("temp_models")
        # shutil.rmtree("temp_dataset")
        pass

if __name__ == "__main__":
    run_verification()
