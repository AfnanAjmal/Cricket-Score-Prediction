from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import json
import os
from typing import List, Dict, Any, Union
import pycaret.regression as pyr
from config import MODELS_DIR, validate_models_directory

app = FastAPI(title="ML Models Dashboard API", version="1.0.0")

# Add CORS middleware for Streamlit integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Validate models directory on startup
validate_models_directory()

# Load model info list
def load_model_info_list():
    model_info_path = os.path.join(MODELS_DIR, "cricket_model_info_list.pkl")
    if not os.path.exists(model_info_path):
        raise FileNotFoundError(f"Model info file not found at: {model_info_path}")
    with open(model_info_path, "rb") as f:
        return pickle.load(f)

# Pydantic models for API
class PredictionInput(BaseModel):
    features: Dict[str, Union[str, int, float]]
    model_name: str

class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    model_name: str

@app.get("/")
async def root():
    return {"message": "ML Models Dashboard API", "version": "1.0.0"}

@app.get("/models")
async def get_models():
    """Returns list of model names with summary metrics"""
    try:
        model_info_list = load_model_info_list()
        models_summary = []
        
        for model_info in model_info_list:
            # Extract metrics from the nested structure
            metrics = model_info["metrics"]
            summary = {
                "name": model_info["name"],
                "filename": model_info["filename"],
                "r2": metrics["R2"][0],
                "mae": metrics["MAE"][0],
                "rmse": metrics["RMSE"][0],
                "mse": metrics["MSE"][0],
                "rmsle": metrics["RMSLE"][0],
                "mape": metrics["MAPE"][0]
            }
            models_summary.append(summary)
        
        return {"models": models_summary, "total_models": len(models_summary)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")

@app.get("/models/best")
async def get_best_model():
    """Returns best model by highest R²"""
    try:
        model_info_list = load_model_info_list()
        best_model = None
        best_r2 = -1
        
        for model_info in model_info_list:
            r2_score = model_info["metrics"]["R2"][0]
            if r2_score > best_r2:
                best_r2 = r2_score
                best_model = {
                    "name": model_info["name"],
                    "filename": model_info["filename"],
                    "r2": r2_score,
                    "mae": model_info["metrics"]["MAE"][0],
                    "rmse": model_info["metrics"]["RMSE"][0],
                    "mse": model_info["metrics"]["MSE"][0],
                    "rmsle": model_info["metrics"]["RMSLE"][0],
                    "mape": model_info["metrics"]["MAPE"][0]
                }
        
        return {"best_model": best_model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding best model: {str(e)}")

@app.get("/models/metrics")
async def get_models_metrics():
    """Returns full comparison table of all models"""
    try:
        model_info_list = load_model_info_list()
        comparison_data = []
        
        for model_info in model_info_list:
            metrics = model_info["metrics"]
            model_data = {
                "Model": model_info["name"],
                "R2": metrics["R2"][0],
                "MAE": metrics["MAE"][0],
                "RMSE": metrics["RMSE"][0],
                "MSE": metrics["MSE"][0],
                "RMSLE": metrics["RMSLE"][0],
                "MAPE": metrics["MAPE"][0]
            }
            comparison_data.append(model_data)
        
        # Calculate averages
        avg_r2 = sum([model["R2"] for model in comparison_data]) / len(comparison_data)
        avg_mae = sum([model["MAE"] for model in comparison_data]) / len(comparison_data)
        avg_rmse = sum([model["RMSE"] for model in comparison_data]) / len(comparison_data)
        
        return {
            "models_comparison": comparison_data,
            "averages": {
                "avg_r2": avg_r2,
                "avg_mae": avg_mae,
                "avg_rmse": avg_rmse
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {str(e)}")

@app.get("/model/{model_name}")
async def get_model_details(model_name: str):
    """Returns detailed metrics of selected model"""
    try:
        model_info_list = load_model_info_list()
        
        for model_info in model_info_list:
            if model_info["name"] == model_name:
                metrics = model_info["metrics"]
                return {
                    "name": model_info["name"],
                    "filename": model_info["filename"],
                    "detailed_metrics": {
                        "R2": metrics["R2"][0],
                        "MAE": metrics["MAE"][0],
                        "RMSE": metrics["RMSE"][0],
                        "MSE": metrics["MSE"][0],
                        "RMSLE": metrics["RMSLE"][0],
                        "MAPE": metrics["MAPE"][0]
                    }
                }
        
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model details: {str(e)}")

@app.post("/predict/single")
async def predict_single(prediction_input: PredictionInput):
    """Accepts JSON input + model name, returns predicted value"""
    try:
        model_info_list = load_model_info_list()
        model_filename = None
        
        # Find the model filename
        for model_info in model_info_list:
            if model_info["name"] == prediction_input.model_name:
                model_filename = model_info["filename"]
                break
        
        if not model_filename:
            raise HTTPException(status_code=404, detail=f"Model {prediction_input.model_name} not found")
        
        # Load the model using PyCaret
        model_path = os.path.join(MODELS_DIR, model_filename.replace('.pkl', ''))
        if not os.path.exists(f"{model_path}.pkl"):
            raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}.pkl")
        loaded_model = pyr.load_model(model_path)
        
        # Create DataFrame from input features
        input_df = pd.DataFrame([prediction_input.features])
        
        # Make prediction
        predictions = pyr.predict_model(loaded_model, data=input_df)
        
        # Extract prediction value (usually in 'prediction_label' column)
        if 'prediction_label' in predictions.columns:
            prediction_value = predictions['prediction_label'].iloc[0]
        else:
            # Fallback to last column if prediction_label not found
            prediction_value = predictions.iloc[0, -1]
        
        return {
            "prediction": float(prediction_value),
            "model_name": prediction_input.model_name,
            "input_features": prediction_input.features
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...), model_name: str = None):
    """Accepts CSV file + model name, returns predictions as JSON"""
    try:
        if not model_name:
            raise HTTPException(status_code=400, detail="model_name parameter is required")
        
        model_info_list = load_model_info_list()
        model_filename = None
        
        # Find the model filename
        for model_info in model_info_list:
            if model_info["name"] == model_name:
                model_filename = model_info["filename"]
                break
        
        if not model_filename:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(pd.io.common.StringIO(contents.decode('utf-8')))
        
        # Load the model using PyCaret
        model_path = os.path.join(MODELS_DIR, model_filename.replace('.pkl', ''))
        if not os.path.exists(f"{model_path}.pkl"):
            raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}.pkl")
        loaded_model = pyr.load_model(model_path)
        
        # Make predictions
        predictions = pyr.predict_model(loaded_model, data=df)
        
        # Extract prediction values
        if 'prediction_label' in predictions.columns:
            prediction_values = predictions['prediction_label'].tolist()
        else:
            # Fallback to last column if prediction_label not found
            prediction_values = predictions.iloc[:, -1].tolist()
        
        return {
            "predictions": prediction_values,
            "model_name": model_name,
            "num_predictions": len(prediction_values)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making batch predictions: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 