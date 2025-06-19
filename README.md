# 🤖 ML Models Dashboard

A modern full-stack machine learning dashboard built with **FastAPI** (backend) and **Streamlit** (frontend) to visualize and interact with multiple regression models trained using PyCaret.

## 🌟 Features

### 📊 Dashboard Page
- **Total number of models** with summary statistics
- **Average R² accuracy** across all models
- **Best model identification** (by highest R²)
- **Interactive charts** comparing models by R², MAE, and RMSE
- **Color-coded metrics** with modern styling

### 🔬 Models Page
- **Model cards** displaying individual model information
- **Detailed metrics** for each model (R², MAE, RMSE, MSE, RMSLE, MAPE)
- **Performance visualizations** for each model
- **Expandable sections** for better organization

### 🔮 Predictions Page
- **Single prediction** with dynamic input form
- **Batch prediction** via CSV file upload
- **Model selection** dropdown
- **Results visualization** and download functionality
- **Real-time predictions** through API calls

## 🏗️ Architecture

```
┌─────────────────┐    HTTP API    ┌─────────────────┐
│   Streamlit     │◄──────────────►│    FastAPI      │
│   Frontend      │    (Port 8501) │    Backend      │
│                 │                │   (Port 8000)   │
└─────────────────┘                └─────────────────┘
                                            │
                                            ▼
                                   ┌─────────────────┐
                                   │     Models      │
                                   │   Directory     │
                                   │   (PyCaret)     │
                                   └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Virtual environment activated
- Trained PyCaret models in `models/` directory

### Installation
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the dashboard:**
   ```bash
   python run_dashboard.py
   ```

3. **Access the applications:**
   - **Frontend Dashboard:** http://localhost:8501
   - **Backend API:** http://localhost:8000
   - **API Documentation:** http://localhost:8000/docs

## 📁 Project Structure

```
Implementation/
├── models/                              # Trained models directory
│   ├── cricket_model_info_list.pkl     # Model metadata
│   ├── XGBRegressor_model_3.pkl        # Individual model files
│   ├── RandomForestRegressor_model_2.pkl
│   ├── ExtraTreesRegressor_model_0.pkl
│   ├── KNeighborsRegressor_model_1.pkl
│   ├── DecisionTreeRegressor_model_4.pkl
│   └── blended_model.pkl               # Ensemble model
├── backend.py                          # FastAPI backend
├── streamlit_app.py                    # Streamlit frontend
├── run_dashboard.py                    # Startup script
├── requirements.txt                    # Dependencies
└── README.md                          # This file
```

## 🔧 API Endpoints

### Backend (FastAPI)
- `GET /models` - List all models with summary metrics
- `GET /models/best` - Get best performing model
- `GET /models/metrics` - Full comparison table
- `GET /model/{model_name}` - Detailed model metrics
- `POST /predict/single` - Single prediction
- `POST /predict/batch` - Batch prediction from CSV

### Example API Usage

**Single Prediction:**
```python
import requests

payload = {
    "features": {
        "overs": 20.0,
        "wickets": 3,
        "runs_last_5": 40,
        "wickets_last_5": 1,
        "striker_sr": 130.0,
        "non_striker_sr": 120.0
    },
    "model_name": "XGBRegressor"
}

response = requests.post("http://localhost:8000/predict/single", json=payload)
print(response.json())
```

## 📊 Model Information Structure

The dashboard expects models to be stored with the following structure:

```python
{
    "name": "XGBRegressor",
    "filename": "XGBRegressor_model_3.pkl",
    "metrics": {
        "R2": {"0": 0.97},
        "MAE": {"0": 2.31},
        "RMSE": {"0": 3.45},
        "MSE": {"0": 11.90},
        "RMSLE": {"0": 0.02},
        "MAPE": {"0": 0.015}
    }
}
```

## 🎨 Features in Detail

### Modern UI Design
- **Clean layout** with sidebar navigation
- **Color-coded metrics** (green for best performance)
- **Interactive charts** using Plotly
- **Responsive design** that works on different screen sizes
- **Loading spinners** for better UX

### Real-time Model Interaction
- **Dynamic form generation** for single predictions
- **CSV upload and processing** for batch predictions
- **Model comparison visualizations**
- **Downloadable results** in CSV format

### Performance Monitoring
- **Comprehensive metrics** (R², MAE, RMSE, MSE, RMSLE, MAPE)
- **Average performance** calculations
- **Best model highlighting**
- **Individual model analysis**

## 🔧 Customization

### Adding New Features
1. **New feature inputs** in the prediction form (modify `streamlit_app.py`)
2. **Additional metrics** in the API responses (modify `backend.py`)
3. **Custom visualizations** using Plotly (add to dashboard page)

### Styling
- Modify the CSS in `streamlit_app.py` for custom themes
- Add new color schemes for different model types
- Customize chart colors and layouts

## 🐛 Troubleshooting

### Common Issues

1. **Backend connection error:**
   - Ensure FastAPI server is running on port 8000
   - Check if virtual environment is activated
   - Verify all dependencies are installed

2. **Model loading errors:**
   - Check if `models/` directory exists
   - Verify PyCaret models are compatible
   - Ensure `cricket_model_info_list.pkl` exists

3. **Prediction errors:**
   - Verify feature names match model expectations
   - Check data types in CSV files
   - Ensure no missing values in input data

### Debug Mode
Run components separately for debugging:

```bash
# Backend only
uvicorn backend:app --reload --port 8000

# Frontend only
streamlit run streamlit_app.py --server.port 8501
```

## 📝 Dependencies

- **fastapi** - Backend API framework
- **uvicorn** - ASGI server
- **streamlit** - Frontend framework
- **pycaret** - ML model loading and prediction
- **plotly** - Interactive visualizations
- **pandas** - Data manipulation
- **requests** - HTTP client for API calls

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

---

**Built with ❤️ using FastAPI and Streamlit** 