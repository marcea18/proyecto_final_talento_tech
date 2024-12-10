"""
This module defines the configurations and file paths used in the project

Variables:
    DATA_FOLDER (str): Name of the folder containing the data
    CONCRETE_FILE (str): Name of the CSV file containing the cement
    composition and strength
    CONCRETE_PATH (Path): Full path to the CSV file of cement composition and
    strength
"""
from pathlib import Path

cwd = Path.cwd()

DATA_FOLDER = "data"
MODEL_FOLDER = "Model"
CONCRETE_FILE = "CPN50_cement_composition_strength.csv"

CONCRETE_PATH = cwd / DATA_FOLDER / CONCRETE_FILE

R1_MODELS = "r1_models"
R2_MODELS = "r2_models"
R7_MODELS = "r7_models"
R28_MODELS = "r28_models"

R1_BEST_MODEL_RMSE = "r1_best_model_rmse.joblib"
R1_BEST_MODEL_MAE = "r1_best_model_mae.joblib"
R1_BEST_MODEL_R2 = "r1_best_model_r2.joblib"

R2_BEST_MODEL_RMSE = "r2_best_model_rmse.joblib"
R2_BEST_MODEL_MAE = "r2_best_model_mae.joblib"
R2_BEST_MODEL_R2 = "r2_best_model_r2.joblib"

R7_BEST_MODEL_RMSE = "r7_best_model_rmse.joblib"
R7_BEST_MODEL_MAE = "r7_best_model_mae.joblib"
R7_BEST_MODEL_R2 = "r7_best_model_r2.joblib"

R28_BEST_MODEL_RMSE = "r28_best_model_rmse.joblib"
R28_BEST_MODEL_MAE = "r28_best_model_mae.joblib"
R28_BEST_MODEL_R2 = "r28_best_model_r2.joblib"

R1_BEST_MODEL_RMSE_PATH = cwd / MODEL_FOLDER / R1_MODELS / R1_BEST_MODEL_RMSE
R1_BEST_MODEL_MAE_PATH = cwd / MODEL_FOLDER / R1_MODELS / R1_BEST_MODEL_MAE
R1_BEST_MODEL_R2_PATH = cwd / MODEL_FOLDER / R1_MODELS / R1_BEST_MODEL_R2

R2_BEST_MODEL_RMSE_PATH = cwd / MODEL_FOLDER / R2_MODELS / R2_BEST_MODEL_RMSE
R2_BEST_MODEL_MAE_PATH = cwd / MODEL_FOLDER / R2_MODELS / R2_BEST_MODEL_MAE
R2_BEST_MODEL_R2_PATH = cwd / MODEL_FOLDER / R2_MODELS / R2_BEST_MODEL_R2

R7_BEST_MODEL_RMSE_PATH = cwd / MODEL_FOLDER / R7_MODELS / R7_BEST_MODEL_RMSE
R7_BEST_MODEL_MAE_PATH = cwd / MODEL_FOLDER / R7_MODELS / R7_BEST_MODEL_MAE
R7_BEST_MODEL_R2_PATH = cwd / MODEL_FOLDER / R7_MODELS / R7_BEST_MODEL_R2

R28_BEST_MODEL_RMSE_PATH = cwd / MODEL_FOLDER / R28_MODELS / R28_BEST_MODEL_RMSE
R28_BEST_MODEL_MAE_PATH = cwd / MODEL_FOLDER / R28_MODELS / R28_BEST_MODEL_MAE
R28_BEST_MODEL_R2_PATH = cwd / MODEL_FOLDER / R28_MODELS / R28_BEST_MODEL_R2
