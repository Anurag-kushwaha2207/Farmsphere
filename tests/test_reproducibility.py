import os
import hashlib
import pytest
from datasets.generate_datasets import generate_datasets
from ml_models.preprocessing.data_cleaner import run_cleaning
from ml_models.preprocessing.dataset_integrator import run_integration

def compute_file_hash(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def test_pipeline_reproducibility_identical_checksums():
    """
    Validates that executing the dataset generation and integration pipeline
    twice produces 100% bitwise identical processed CSV checksums.
    """
    # Run 1
    generate_datasets()
    run_cleaning()
    run_integration()
    
    yield_csv = 'datasets/processed/yield_prediction_clean.csv'
    soil_clean = 'datasets/processed/soil_cleaned.csv'
    
    hash_yield_1 = compute_file_hash(yield_csv)
    hash_soil_1 = compute_file_hash(soil_clean)
    
    # Run 2
    generate_datasets()
    run_cleaning()
    run_integration()
    
    hash_yield_2 = compute_file_hash(yield_csv)
    hash_soil_2 = compute_file_hash(soil_clean)
    
    assert hash_yield_1 == hash_yield_2, "yield_prediction_clean.csv checksum differs between consecutive runs!"
    assert hash_soil_1 == hash_soil_2, "soil_cleaned.csv checksum differs between consecutive runs!"
