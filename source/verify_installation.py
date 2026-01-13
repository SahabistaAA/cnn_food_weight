"""
Verification script for Food Classification Models.
Runs a quick training cycle for each model on dummy/subset data.
"""
import sys
from pathlib import Path
from loguru import logger

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

from source.main import get_model, FoodPipeline
from source.config import config

def verify_models():
    logger.info("Starting verification...")
    
    # Mock data if needed or use real data loader if fast enough
    # For verification, we try to load real data but limit it
    pipeline = FoodPipeline()
    try:
        pipeline.load_data()
        # Limit data for speed
        pipeline.data['train'] = pipeline.data['train'].head(20)
        pipeline.data['val'] = pipeline.data['val'].head(10)
        pipeline.data['test'] = pipeline.data['test'].head(10)
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return

    models = ['SVM', 'RF', 'DT', 'KNN', 'CNN']
    
    for model_name in models:
        try:
            logger.info(f"Verifying {model_name}...")
            pipeline.run_classification(model_name, optimize=False)
            logger.info(f"{model_name} passed.")
        except Exception as e:
            logger.exception(f"{model_name} failed: {e}")

if __name__ == "__main__":
    verify_models()
