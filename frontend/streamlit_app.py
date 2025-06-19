import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="ML Models Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .metric-card h3 {
        color: #ffffff !important;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card h2 {
        color: #ffffff !important;
        font-size: 2.2rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .best-model-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .best-model-card h3 {
        color: #ffffff !important;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .best-model-card h2 {
        color: #ffffff !important;
        font-size: 1.6rem;
        font-weight: bold;
        margin: 0.5rem 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .best-model-card p {
        color: #ffffff !important;
        font-size: 1.1rem;
        font-weight: 500;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .model-card h4 {
        color: #ffffff !important;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .model-card p {
        color: #ffffff !important;
        font-size: 1.1rem;
        font-weight: 500;
        margin: 0.5rem 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .model-card strong {
        color: #ffffff !important;
        font-weight: 700;
    }
    .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Fix for Streamlit's default styling overrides */
    .stMarkdown div[data-testid="metric-container"] {
        background: transparent !important;
        border: none !important;
    }
    
    /* Enhanced styling for expandable sections */
    .streamlit-expanderHeader {
        background-color: #f8f9fa !important;
        border-radius: 10px !important;
    }
    
    /* Make sure text in cards is always visible */
    .metric-card *, .best-model-card *, .model-card * {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# API base URL - Use environment variable or fallback to localhost
import os
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Helper functions
@st.cache_data(ttl=600)  # Cache for 10 minutes
def fetch_models():
    """Fetch all models data"""
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching models: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend API. Please make sure the FastAPI server is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

@st.cache_data(ttl=600)  # Cache for 10 minutes
def fetch_best_model():
    """Fetch best model data"""
    try:
        response = requests.get(f"{API_BASE_URL}/models/best")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching best model: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

@st.cache_data(ttl=600)  # Cache for 10 minutes
def fetch_metrics():
    """Fetch comparison metrics"""
    try:
        response = requests.get(f"{API_BASE_URL}/models/metrics")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching metrics: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

@st.cache_data(ttl=300)  # Cache model details for 5 minutes
def fetch_model_details_cached(model_name):
    """Fetch detailed model information with caching"""
    try:
        response = requests.get(f"{API_BASE_URL}/model/{model_name}")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        return None

def fetch_model_details(model_name):
    """Fetch detailed model information"""
    try:
        response = requests.get(f"{API_BASE_URL}/model/{model_name}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching model details: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def make_single_prediction(features, model_name):
    """Make single prediction"""
    try:
        payload = {
            "features": features,
            "model_name": model_name
        }
        response = requests.post(f"{API_BASE_URL}/predict/single", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error making prediction: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def make_batch_prediction(csv_file, model_name):
    """Make batch prediction"""
    try:
        files = {"file": csv_file}
        # Pass model_name as query parameter
        params = {"model_name": model_name}
        response = requests.post(f"{API_BASE_URL}/predict/batch", files=files, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error making batch prediction: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def validate_csv_columns(df, required_columns):
    """Validate CSV has all required columns"""
    missing_columns = [col for col in required_columns if col not in df.columns]
    return missing_columns

# Initialize session state for caching
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'last_selected_model' not in st.session_state:
    st.session_state.last_selected_model = None

# Sidebar navigation
st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
st.sidebar.title("🤖 ML Dashboard")

# Add refresh button
if st.sidebar.button("🔄 Refresh Data", help="Clear cache and reload all data"):
    st.cache_data.clear()
    st.session_state.models_loaded = False
    st.rerun()

# Force clear cache for this session to ensure updated models are loaded
if 'force_refresh_done' not in st.session_state:
    st.cache_data.clear()
    st.session_state.force_refresh_done = True

page = st.sidebar.selectbox(
    "Navigate to:",
    ["📊 Dashboard", "🔬 Models", "🔮 Predictions"]
)

# Show API status in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("**🔗 API Status**")
    
    # Add refresh button
    if st.button("🔄 Refresh Status", use_container_width=True):
        # Clear all caches
        st.cache_data.clear()
        st.rerun()
    
    try:
        # Use a fresh session to avoid any caching issues
        session = requests.Session()
        response = session.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            st.success("✅ Backend Connected")
            # Show API info
            try:
                api_info = response.json()
                st.caption(f"Version: {api_info.get('version', 'Unknown')}")
            except:
                pass
        else:
            st.error(f"❌ Backend Error (Status: {response.status_code})")
    except requests.exceptions.ConnectionError:
        st.error("❌ Backend Offline - Connection Refused")
    except requests.exceptions.Timeout:
        st.error("❌ Backend Timeout - Server too slow")
    except Exception as e:
        st.error(f"❌ Backend Error: {str(e)}")
        st.caption("Check if backend is running on localhost:8000")

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Main content based on selected page
if page == "📊 Dashboard":
    st.markdown('<h1 class="main-header">🤖 ML Models Dashboard</h1>', unsafe_allow_html=True)
    
    # Fetch data with loading indicators
    with st.spinner("🔄 Loading dashboard data..."):
        models_data = fetch_models()
        
    if models_data:
        # Load other data in parallel with spinner
        with st.spinner("📊 Loading metrics and best model..."):
            col1, col2 = st.columns(2)
            with col1:
                best_model_data = fetch_best_model()
            with col2:
                metrics_data = fetch_metrics()
    
    if models_data and best_model_data and metrics_data:
        # Summary metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>📈 Total Models</h3>
                    <h2>{models_data['total_models']}</h2>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col2:
            avg_r2 = metrics_data['averages']['avg_r2']
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>🎯 Average R²</h3>
                    <h2>{avg_r2:.4f}</h2>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col3:
            avg_mae = metrics_data['averages']['avg_mae']
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>📊 Average MAE</h3>
                    <h2>{avg_mae:.4f}</h2>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col4:
            best_model = best_model_data['best_model']
            st.markdown(
                f"""
                <div class="best-model-card">
                    <h3>🏆 Best Model</h3>
                    <h2>{best_model['name']}</h2>
                    <p>R²: {best_model['r2']:.4f}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        # Charts section
        st.subheader("📊 Model Performance Comparison")
        
        # Prepare data for charts
        df_models = pd.DataFrame(models_data['models'])
        
        # Create comprehensive comparison tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Overall Performance", 
            "📈 Individual Metrics", 
            "🔄 Radar Comparison",
            "🏆 Performance Ranking", 
            "📊 Metric Distribution",
            "⚖️ Model Trade-offs"
        ])
        
        with tab1:
            st.markdown("### 🎯 Overall Model Performance Dashboard")
            
            # Create subplots for comprehensive overview
            col1, col2 = st.columns(2)
            
            with col1:
                # R² vs MAE Scatter Plot
                fig_scatter = px.scatter(
                    df_models, 
                    x='mae', 
                    y='r2',
                    size='rmse',
                    color='name',
                    title="📊 R² vs MAE (Bubble size = RMSE)",
                    labels={'mae': 'Mean Absolute Error', 'r2': 'R² Score'},
                    hover_data=['rmse', 'mse']
                )
                fig_scatter.update_layout(
                    xaxis_title="Mean Absolute Error (Lower is Better)",
                    yaxis_title="R² Score (Higher is Better)",
                    showlegend=True
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col2:
                # Performance Score Calculation (normalized)
                df_models_norm = df_models.copy()
                # Normalize metrics (R² higher is better, others lower is better)
                df_models_norm['r2_norm'] = df_models_norm['r2'] / df_models_norm['r2'].max()
                df_models_norm['mae_norm'] = df_models_norm['mae'].min() / df_models_norm['mae']
                df_models_norm['rmse_norm'] = df_models_norm['rmse'].min() / df_models_norm['rmse']
                df_models_norm['performance_score'] = (df_models_norm['r2_norm'] + df_models_norm['mae_norm'] + df_models_norm['rmse_norm']) / 3
                
                # Overall Performance Score
                fig_performance = px.bar(
                    df_models_norm.sort_values('performance_score', ascending=True), 
                    x='performance_score',
                    y='name',
                    orientation='h',
                    title="🏆 Overall Performance Score",
                    color='performance_score',
                    color_continuous_scale='Viridis',
                    text='performance_score'
                )
                fig_performance.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig_performance.update_layout(
                    xaxis_title="Performance Score (Higher is Better)",
                    yaxis_title="Models",
                    showlegend=False
                )
                st.plotly_chart(fig_performance, use_container_width=True)
        
        with tab2:
            st.markdown("### 📈 Individual Metric Comparisons")
            
            # Create 2x2 grid for different metrics
            col1, col2 = st.columns(2)
            
            with col1:
                # R² Comparison (Higher is better)
                fig_r2 = px.bar(
                    df_models.sort_values('r2', ascending=True), 
                    x='r2',
                    y='name',
                    orientation='h',
                    title="🎯 R² Score Comparison",
                    color='r2',
                    color_continuous_scale='Greens',
                    text='r2'
                )
                fig_r2.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                fig_r2.update_layout(xaxis_title="R² Score", yaxis_title="Models")
                st.plotly_chart(fig_r2, use_container_width=True)
                
                # RMSE Comparison (Lower is better)
                fig_rmse = px.bar(
                    df_models.sort_values('rmse', ascending=False), 
                    x='rmse',
                    y='name',
                    orientation='h',
                    title="📉 RMSE Comparison",
                    color='rmse',
                    color_continuous_scale='Reds',
                    text='rmse'
                )
                fig_rmse.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig_rmse.update_layout(xaxis_title="RMSE (Lower is Better)", yaxis_title="Models")
                st.plotly_chart(fig_rmse, use_container_width=True)
            
            with col2:
                # MAE Comparison (Lower is better)
                fig_mae = px.bar(
                    df_models.sort_values('mae', ascending=False), 
                    x='mae',
                    y='name',
                    orientation='h',
                    title="📊 MAE Comparison",
                    color='mae',
                    color_continuous_scale='Oranges',
                    text='mae'
                )
                fig_mae.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig_mae.update_layout(xaxis_title="MAE (Lower is Better)", yaxis_title="Models")
                st.plotly_chart(fig_mae, use_container_width=True)
                
                # MSE Comparison (Lower is better)
                fig_mse = px.bar(
                    df_models.sort_values('mse', ascending=False), 
                    x='mse',
                    y='name',
                    orientation='h',
                    title="📈 MSE Comparison",
                    color='mse',
                    color_continuous_scale='Blues',
                    text='mse'
                )
                fig_mse.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig_mse.update_layout(xaxis_title="MSE (Lower is Better)", yaxis_title="Models")
                st.plotly_chart(fig_mse, use_container_width=True)
        
        with tab3:
            st.markdown("### 🔄 Radar Chart Comparison")
            
            # Prepare data for radar chart (normalize all metrics to 0-1 scale)
            radar_data = df_models.copy()
            
            # Normalize metrics (invert error metrics so higher = better for visualization)
            radar_data['R² Score'] = radar_data['r2']
            radar_data['MAE (inv)'] = 1 - (radar_data['mae'] - radar_data['mae'].min()) / (radar_data['mae'].max() - radar_data['mae'].min())
            radar_data['RMSE (inv)'] = 1 - (radar_data['rmse'] - radar_data['rmse'].min()) / (radar_data['rmse'].max() - radar_data['rmse'].min())
            radar_data['MSE (inv)'] = 1 - (radar_data['mse'] - radar_data['mse'].min()) / (radar_data['mse'].max() - radar_data['mse'].min())
            radar_data['MAPE (inv)'] = 1 - (radar_data['mape'] - radar_data['mape'].min()) / (radar_data['mape'].max() - radar_data['mape'].min())
            
            # Create radar chart
            categories = ['R² Score', 'MAE (inv)', 'RMSE (inv)', 'MSE (inv)', 'MAPE (inv)']
            
            fig_radar = go.Figure()
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            for i, model in enumerate(radar_data['name']):
                values = [
                    radar_data.iloc[i]['R² Score'],
                    radar_data.iloc[i]['MAE (inv)'],
                    radar_data.iloc[i]['RMSE (inv)'],
                    radar_data.iloc[i]['MSE (inv)'],
                    radar_data.iloc[i]['MAPE (inv)']
                ]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=model,
                    line_color=colors[i % len(colors)]
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="🔄 Multi-Metric Radar Comparison<br><sub>(All metrics normalized: Higher = Better)</sub>"
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
            st.info("💡 **Interpretation**: Models closer to the outer edge perform better. All error metrics are inverted so higher values indicate better performance.")
        
        with tab4:
            st.markdown("### 🏆 Performance Ranking Analysis")
            
            # Create ranking table
            ranking_data = df_models.copy()
            
            # Calculate rankings (1 = best)
            ranking_data['R² Rank'] = ranking_data['r2'].rank(ascending=False, method='min').astype(int)
            ranking_data['MAE Rank'] = ranking_data['mae'].rank(ascending=True, method='min').astype(int)
            ranking_data['RMSE Rank'] = ranking_data['rmse'].rank(ascending=True, method='min').astype(int)
            ranking_data['MSE Rank'] = ranking_data['mse'].rank(ascending=True, method='min').astype(int)
            ranking_data['MAPE Rank'] = ranking_data['mape'].rank(ascending=True, method='min').astype(int)
            
            # Average rank
            ranking_data['Average Rank'] = (
                ranking_data['R² Rank'] + ranking_data['MAE Rank'] + 
                ranking_data['RMSE Rank'] + ranking_data['MSE Rank'] + ranking_data['MAPE Rank']
            ) / 5
            
            # Display ranking table
            display_ranking = ranking_data[['name', 'R² Rank', 'MAE Rank', 'RMSE Rank', 'MSE Rank', 'MAPE Rank', 'Average Rank']].copy()
            display_ranking.columns = ['Model', 'R² Rank', 'MAE Rank', 'RMSE Rank', 'MSE Rank', 'MAPE Rank', 'Avg Rank']
            display_ranking = display_ranking.sort_values('Avg Rank')
            
            st.dataframe(
                display_ranking,
                use_container_width=True,
                column_config={
                    "Avg Rank": st.column_config.NumberColumn(
                        "Average Rank",
                        help="Lower is better - average of all metric rankings",
                        format="%.2f"
                    )
                }
            )
            
            # Ranking visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Average ranking bar chart
                fig_rank = px.bar(
                    display_ranking,
                    x='Avg Rank',
                    y='Model',
                    orientation='h',
                    title="📊 Average Ranking (Lower = Better)",
                    color='Avg Rank',
                    color_continuous_scale='RdYlGn_r',
                    text='Avg Rank'
                )
                fig_rank.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                st.plotly_chart(fig_rank, use_container_width=True)
            
            with col2:
                # Ranking heatmap
                heatmap_data = display_ranking.set_index('Model')[['R² Rank', 'MAE Rank', 'RMSE Rank', 'MSE Rank', 'MAPE Rank']]
                
                fig_heatmap = px.imshow(
                    heatmap_data.values,
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    color_continuous_scale='RdYlGn_r',
                    title="🔥 Ranking Heatmap",
                    text_auto=True
                )
                fig_heatmap.update_layout(
                    xaxis_title="Metrics",
                    yaxis_title="Models"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with tab5:
            st.markdown("### 📊 Metric Distribution Analysis")
            
            # Prepare data for distribution analysis
            metrics_long = []
            for _, row in df_models.iterrows():
                metrics_long.extend([
                    {'Model': row['name'], 'Metric': 'R²', 'Value': row['r2'], 'Type': 'Accuracy'},
                    {'Model': row['name'], 'Metric': 'MAE', 'Value': row['mae'], 'Type': 'Error'},
                    {'Model': row['name'], 'Metric': 'RMSE', 'Value': row['rmse'], 'Type': 'Error'},
                    {'Model': row['name'], 'Metric': 'MAPE', 'Value': row['mape'], 'Type': 'Error'}
                ])
            
            df_metrics_long = pd.DataFrame(metrics_long)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Box plot for metric distributions
                fig_box = px.box(
                    df_metrics_long,
                    x='Metric',
                    y='Value',
                    color='Type',
                    title="📦 Metric Distribution Across Models",
                    hover_data=['Model']
                )
                fig_box.update_layout(
                    xaxis_title="Metrics",
                    yaxis_title="Values"
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            with col2:
                # Violin plot for detailed distribution
                fig_violin = px.violin(
                    df_metrics_long,
                    x='Metric',
                    y='Value',
                    color='Type',
                    title="🎻 Detailed Metric Distribution",
                    box=True,
                    hover_data=['Model']
                )
                st.plotly_chart(fig_violin, use_container_width=True)
        
        with tab6:
            st.markdown("### ⚖️ Model Trade-offs Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy vs Speed proxy (model complexity)
                model_complexity = {
                    'ExtraTreesRegressor': 0.9,
                    'RandomForestRegressor': 0.8,
                    'KNeighborsRegressor': 0.3,
                    'DecisionTreeRegressor': 0.5
                }
                
                df_tradeoff = df_models.copy()
                df_tradeoff['Complexity'] = df_tradeoff['name'].map(model_complexity)
                
                fig_tradeoff = px.scatter(
                    df_tradeoff,
                    x='Complexity',
                    y='r2',
                    size='mae',
                    color='name',
                    title="🎯 Accuracy vs Model Complexity",
                    labels={'Complexity': 'Model Complexity (0-1)', 'r2': 'R² Score'},
                    hover_data=['mae', 'rmse']
                )
                fig_tradeoff.update_layout(
                    xaxis_title="Model Complexity (Higher = More Complex)",
                    yaxis_title="R² Score (Higher = Better)"
                )
                st.plotly_chart(fig_tradeoff, use_container_width=True)
            
            with col2:
                # Error comparison (MAE vs RMSE)
                fig_error = px.scatter(
                    df_models,
                    x='mae',
                    y='rmse',
                    size='r2',
                    color='name',
                    title="📊 Error Trade-off (MAE vs RMSE)",
                    labels={'mae': 'Mean Absolute Error', 'rmse': 'Root Mean Square Error'},
                    hover_data=['r2', 'mse']
                )
                fig_error.update_layout(
                    xaxis_title="MAE (Lower = Better)",
                    yaxis_title="RMSE (Lower = Better)"
                )
                st.plotly_chart(fig_error, use_container_width=True)
            
            # Summary insights
            st.markdown("#### 💡 Key Insights:")
            best_model = df_models.loc[df_models['r2'].idxmax()]
            lowest_mae = df_models.loc[df_models['mae'].idxmin()]
            lowest_rmse = df_models.loc[df_models['rmse'].idxmin()]
            
            insight_col1, insight_col2, insight_col3 = st.columns(3)
            
            with insight_col1:
                st.info(f"🏆 **Best Overall**: {best_model['name']}\n\nHighest R² = {best_model['r2']:.4f}")
            
            with insight_col2:
                st.info(f"🎯 **Most Precise**: {lowest_mae['name']}\n\nLowest MAE = {lowest_mae['mae']:.4f}")
            
            with insight_col3:
                st.info(f"📊 **Most Consistent**: {lowest_rmse['name']}\n\nLowest RMSE = {lowest_rmse['rmse']:.4f}")

elif page == "🔬 Models":
    st.markdown('<h1 class="main-header">🔬 Model Explorer</h1>', unsafe_allow_html=True)
    
    models_data = fetch_models()
    
    if models_data:
        # Overview section
        st.markdown("### 📊 Models Overview")
        
        # Create summary cards for all models
        cols = st.columns(len(models_data['models']))
        for i, model in enumerate(models_data['models']):
            with cols[i]:
                # Performance ranking
                rank = sorted(models_data['models'], key=lambda x: x['r2'], reverse=True).index(model) + 1
                rank_emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "🏅"
                
                st.markdown(
                    f"""
                    <div class="model-card" style="min-height: 200px;">
                        <h4>{rank_emoji} {model['name']}</h4>
                        <p><strong>Rank:</strong> #{rank}</p>
                        <p><strong>R²:</strong> {model['r2']:.4f}</p>
                        <p><strong>MAE:</strong> {model['mae']:.4f}</p>
                        <p><strong>RMSE:</strong> {model['rmse']:.4f}</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        st.markdown("---")
        
        # Detailed model exploration
        st.subheader("🔍 Detailed Model Analysis")
        
        # Create enhanced model cards
        for i, model in enumerate(models_data['models']):
            # Calculate additional insights
            rank = sorted(models_data['models'], key=lambda x: x['r2'], reverse=True).index(model) + 1
            rank_emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "🏅"
            
            # Performance category
            if model['r2'] >= 0.99:
                performance = "🌟 Excellent"
            elif model['r2'] >= 0.95:
                performance = "⭐ Very Good"
            elif model['r2'] >= 0.90:
                performance = "✨ Good"
            else:
                performance = "💫 Fair"
            
            with st.expander(f"{rank_emoji} {model['name']} - {performance} (R² = {model['r2']:.4f})", expanded=False):
                
                # Create tabs for different aspects
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📊 Performance Metrics", 
                    "📈 Visualizations", 
                    "🎯 Strengths & Insights",
                    "⚖️ Trade-offs",
                    "🔬 Technical Details"
                ])
                
                with tab1:
                    st.markdown("#### 📊 Comprehensive Performance Metrics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(
                            f"""
                            <div class="model-card">
                                <h4>🎯 Accuracy Metrics</h4>
                                <p><strong>R² Score:</strong> {model['r2']:.6f}</p>
                                <p><strong>Adjusted R²:</strong> {model['r2'] * 0.98:.6f}</p>
                                <p><strong>Accuracy:</strong> {model['r2'] * 100:.2f}%</p>
                                <p><strong>Ranking:</strong> #{rank} of {len(models_data['models'])}</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        st.markdown(
                            f"""
                            <div class="model-card">
                                <h4>📉 Error Metrics</h4>
                                <p><strong>MAE:</strong> {model['mae']:.6f}</p>
                                <p><strong>RMSE:</strong> {model['rmse']:.6f}</p>
                                <p><strong>MSE:</strong> {model['mse']:.6f}</p>
                                <p><strong>MAPE:</strong> {model['mape']:.6f}</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    
                    with col3:
                        # Calculate relative performance
                        best_r2 = max([m['r2'] for m in models_data['models']])
                        relative_performance = (model['r2'] / best_r2) * 100
                        
                        # Error comparison
                        all_maes = [m['mae'] for m in models_data['models']]
                        mae_percentile = (sorted(all_maes).index(model['mae']) + 1) / len(all_maes) * 100
                        
                        st.markdown(
                            f"""
                            <div class="model-card">
                                <h4>📊 Relative Performance</h4>
                                <p><strong>vs Best Model:</strong> {relative_performance:.1f}%</p>
                                <p><strong>MAE Percentile:</strong> {mae_percentile:.0f}%</p>
                                <p><strong>Performance Level:</strong> {performance}</p>
                                <p><strong>Recommended:</strong> {"✅ Yes" if rank <= 2 else "⚠️ Consider"}</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                
                with tab2:
                    st.markdown("#### 📈 Model Visualizations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Metrics comparison radar
                        if len(models_data['models']) > 1:
                            # Normalize metrics for radar chart
                            all_r2 = [m['r2'] for m in models_data['models']]
                            all_mae = [m['mae'] for m in models_data['models']]
                            all_rmse = [m['rmse'] for m in models_data['models']]
                            all_mape = [m['mape'] for m in models_data['models']]
                            
                            # Normalize (R² higher is better, others lower is better)
                            r2_norm = model['r2'] / max(all_r2)
                            mae_norm = min(all_mae) / model['mae']
                            rmse_norm = min(all_rmse) / model['rmse']
                            mape_norm = min(all_mape) / model['mape']
                            
                            categories = ['R² Score', 'MAE (inv)', 'RMSE (inv)', 'MAPE (inv)']
                            values = [r2_norm, mae_norm, rmse_norm, mape_norm]
                            
                            fig_radar = go.Figure()
                            fig_radar.add_trace(go.Scatterpolar(
                                r=values,
                                theta=categories,
                                fill='toself',
                                name=model['name'],
                                line_color='#667eea'
                            ))
                            
                            fig_radar.update_layout(
                                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                showlegend=False,
                                title=f"📊 {model['name']} Performance Profile",
                                height=400
                            )
                            st.plotly_chart(fig_radar, use_container_width=True)
                        
                    with col2:
                        # Individual metrics bar chart with context
                        metrics_data = {
                            'Metric': ['R² Score', 'MAE', 'RMSE', 'MSE', 'MAPE'],
                            'Value': [model['r2'], model['mae'], model['rmse'], model['mse'], model['mape']],
                            'Type': ['Accuracy', 'Error', 'Error', 'Error', 'Error']
                        }
                        
                        fig_metrics = px.bar(
                            x=metrics_data['Metric'],
                            y=metrics_data['Value'],
                            color=metrics_data['Type'],
                            title=f"📊 {model['name']} Metrics Breakdown",
                            color_discrete_map={'Accuracy': '#2ca02c', 'Error': '#d62728'}
                        )
                        fig_metrics.update_layout(height=400)
                        st.plotly_chart(fig_metrics, use_container_width=True)
                    
                    # Comparison with other models
                    st.markdown("#### 🔄 Comparison with Other Models")
                    
                    # Create comparison data
                    comparison_data = []
                    for other_model in models_data['models']:
                        comparison_data.append({
                            'Model': other_model['name'],
                            'R² Score': other_model['r2'],
                            'MAE': other_model['mae'],
                            'RMSE': other_model['rmse'],
                            'Current Model': other_model['name'] == model['name']
                        })
                    
                    df_comparison = pd.DataFrame(comparison_data)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # R² comparison
                        fig_r2_comp = px.bar(
                            df_comparison.sort_values('R² Score'),
                            x='R² Score',
                            y='Model',
                            orientation='h',
                            color='Current Model',
                            title="🎯 R² Score Comparison",
                            color_discrete_map={True: '#667eea', False: '#cccccc'}
                        )
                        fig_r2_comp.update_layout(showlegend=False, height=300)
                        st.plotly_chart(fig_r2_comp, use_container_width=True)
                    
                    with col2:
                        # MAE comparison
                        fig_mae_comp = px.bar(
                            df_comparison.sort_values('MAE', ascending=False),
                            x='MAE',
                            y='Model',
                            orientation='h',
                            color='Current Model',
                            title="📉 MAE Comparison (Lower = Better)",
                            color_discrete_map={True: '#667eea', False: '#cccccc'}
                        )
                        fig_mae_comp.update_layout(showlegend=False, height=300)
                        st.plotly_chart(fig_mae_comp, use_container_width=True)
                
                with tab3:
                    st.markdown("#### 🎯 Model Strengths & Key Insights")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Strengths analysis
                        strengths = []
                        if model['r2'] >= 0.99:
                            strengths.append("🌟 Exceptional accuracy (R² ≥ 0.99)")
                        elif model['r2'] >= 0.95:
                            strengths.append("⭐ Very high accuracy (R² ≥ 0.95)")
                        
                        if model['mae'] < 2.0:
                            strengths.append("🎯 Low prediction error (MAE < 2.0)")
                        
                        if rank == 1:
                            strengths.append("🥇 Best performing model overall")
                        elif rank == 2:
                            strengths.append("🥈 Second best performance")
                        elif rank == 3:
                            strengths.append("🥉 Third best performance")
                        
                        # Add model-specific strengths
                        if 'ExtraTrees' in model['name']:
                            strengths.extend([
                                "🌳 Handles overfitting well with random features",
                                "⚡ Good balance of bias and variance",
                                "🔄 Robust to outliers"
                            ])
                        elif 'RandomForest' in model['name']:
                            strengths.extend([
                                "🌲 Ensemble method for stability",
                                "📊 Good feature importance insights", 
                                "🛡️ Reduces overfitting risk"
                            ])
                        elif 'KNeighbors' in model['name']:
                            strengths.extend([
                                "🎯 Simple and interpretable",
                                "📍 Good for local patterns",
                                "🔄 Non-parametric approach"
                            ])
                        elif 'DecisionTree' in model['name']:
                            strengths.extend([
                                "🌿 Highly interpretable",
                                "⚡ Fast predictions",
                                "🔍 Clear decision rules"
                            ])
                        
                        st.markdown("**🎯 Key Strengths:**")
                        for strength in strengths:
                            st.markdown(f"- {strength}")
                    
                    with col2:
                        # Use case recommendations
                        st.markdown("**💡 Recommended Use Cases:**")
                        
                        if model['r2'] >= 0.99:
                            st.markdown("- 🏆 **Production deployment** - Highest accuracy")
                            st.markdown("- 📊 **Critical predictions** - Most reliable")
                            st.markdown("- 🎯 **Benchmark model** - Best performance reference")
                        elif model['r2'] >= 0.95:
                            st.markdown("- ✅ **General use** - Very good accuracy")
                            st.markdown("- 🔄 **Backup model** - Reliable alternative")
                            st.markdown("- 📈 **Comparative analysis** - Good baseline")
                        else:
                            st.markdown("- 🧪 **Experimental use** - For testing")
                            st.markdown("- 📚 **Educational purposes** - Learning different approaches")
                            st.markdown("- 🔍 **Ensemble component** - Part of larger system")
                        
                        # Performance insights
                        st.markdown("**📈 Performance Insights:**")
                        error_level = "Low" if model['mae'] < 2.0 else "Medium" if model['mae'] < 5.0 else "High"
                        consistency = "High" if model['rmse'] < 5.0 else "Medium" if model['rmse'] < 10.0 else "Low"
                        
                        st.markdown(f"- 📊 **Error Level:** {error_level}")
                        st.markdown(f"- 🎯 **Consistency:** {consistency}")
                        st.markdown(f"- 📈 **Relative Accuracy:** {relative_performance:.1f}% of best model")
                
                with tab4:
                    st.markdown("#### ⚖️ Model Trade-offs Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Complexity vs Performance
                        complexity_map = {
                            'ExtraTreesRegressor': {'complexity': 0.9, 'speed': 0.6, 'interpretability': 0.3},
                            'RandomForestRegressor': {'complexity': 0.8, 'speed': 0.7, 'interpretability': 0.4},
                            'KNeighborsRegressor': {'complexity': 0.3, 'speed': 0.8, 'interpretability': 0.7},
                            'DecisionTreeRegressor': {'complexity': 0.5, 'speed': 0.9, 'interpretability': 0.9}
                        }
                        
                        model_props = complexity_map.get(model['name'], {'complexity': 0.5, 'speed': 0.5, 'interpretability': 0.5})
                        
                        st.markdown("**🔄 Model Characteristics:**")
                        
                        # Complexity bar
                        complexity_pct = model_props['complexity'] * 100
                        st.markdown(f"**Model Complexity:** {complexity_pct:.0f}%")
                        st.progress(model_props['complexity'])
                        
                        # Speed bar  
                        speed_pct = model_props['speed'] * 100
                        st.markdown(f"**Prediction Speed:** {speed_pct:.0f}%")
                        st.progress(model_props['speed'])
                        
                        # Interpretability bar
                        interp_pct = model_props['interpretability'] * 100
                        st.markdown(f"**Interpretability:** {interp_pct:.0f}%")
                        st.progress(model_props['interpretability'])
                    
                    with col2:
                        # Trade-off recommendations
                        st.markdown("**💡 Trade-off Analysis:**")
                        
                        if model_props['complexity'] > 0.7:
                            st.info("🔬 **High Complexity Model**\n- Better accuracy\n- Longer training time\n- More memory usage")
                        elif model_props['complexity'] > 0.4:
                            st.info("⚖️ **Medium Complexity Model**\n- Balanced performance\n- Moderate resources\n- Good trade-off")
                        else:
                            st.info("⚡ **Low Complexity Model**\n- Fast predictions\n- Lower memory usage\n- Easy to understand")
                        
                        if model_props['interpretability'] > 0.7:
                            st.success("🔍 **Highly Interpretable**\n- Easy to explain\n- Good for compliance\n- Clear decision logic")
                        elif model_props['interpretability'] > 0.4:
                            st.warning("📊 **Moderately Interpretable**\n- Some explainability\n- Feature importance available\n- Partial transparency")
                        else:
                            st.error("🔒 **Black Box Model**\n- High accuracy\n- Limited explainability\n- Complex internal logic")
                
                with tab5:
                    st.markdown("#### 🔬 Technical Details & Specifications")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📋 Model Specifications:**")
                        st.code(f"""
Model Name: {model['name']}
Algorithm Type: {model['name'].replace('Regressor', '')}
Filename: {model['filename']}
Performance Rank: #{rank} of {len(models_data['models'])}
                        """)
                        
                        st.markdown("**📊 Detailed Metrics:**")
                        st.code(f"""
R² Score:    {model['r2']:.8f}
MAE:         {model['mae']:.8f}
RMSE:        {model['rmse']:.8f}
MSE:         {model['mse']:.8f}
RMSLE:       {model['rmsle']:.8f}
MAPE:        {model['mape']:.8f}
                        """)
                    
                    with col2:
                        st.markdown("**🎯 Performance Statistics:**")
                        
                        # Calculate percentiles
                        all_models = models_data['models']
                        r2_values = sorted([m['r2'] for m in all_models], reverse=True)
                        mae_values = sorted([m['mae'] for m in all_models])
                        
                        r2_percentile = ((len(r2_values) - r2_values.index(model['r2'])) / len(r2_values)) * 100
                        mae_percentile = ((mae_values.index(model['mae']) + 1) / len(mae_values)) * 100
                        
                        st.markdown(f"**R² Percentile:** {r2_percentile:.0f}% (Higher is better)")
                        st.markdown(f"**MAE Percentile:** {mae_percentile:.0f}% (Lower is better)")
                        st.markdown(f"**Overall Rank:** #{rank} of {len(all_models)}")
                        
                        # Model family info
                        if 'Tree' in model['name']:
                            st.info("🌳 **Tree-based Model Family**\n- Uses decision trees\n- Good for non-linear patterns\n- Handles feature interactions well")
                        elif 'Neighbors' in model['name']:
                            st.info("📍 **Instance-based Learning**\n- Uses similarity metrics\n- Non-parametric approach\n- Good for local patterns")
                        
                        # Quick action buttons
                        st.markdown("**⚡ Quick Actions:**")
                        if st.button(f"🔮 Go to Predictions with {model['name']}", key=f"test_{i}", use_container_width=True):
                            # Set the selected model in session state
                            st.session_state.selected_model = model['name']
                            st.session_state.navigate_to_predictions = True
                            st.info(f"✅ {model['name']} selected! Navigate to '🔮 Predictions' tab to test this model.")
                        
                        if rank <= 2:
                            st.success(f"✅ **Recommended for production use**")
                        else:
                            st.info(f"💡 **Good for experimental use**")

elif page == "🔮 Predictions":
    st.markdown('<h1 class="main-header">🔮 Model Predictions</h1>', unsafe_allow_html=True)
    
    # Load models with caching
    with st.spinner("🔄 Loading available models..."):
        models_data = fetch_models()
    
    if models_data:
        # Cache model information
        model_names = [model['name'] for model in models_data['models']]
        model_info_dict = {model['name']: model for model in models_data['models']}
        
        st.markdown("---")
        
        # Prediction tabs
        tab1, tab2 = st.tabs(["🎯 Single Prediction", "📊 Batch Prediction"])
        
        with tab1:
            st.subheader("Single Prediction")
            st.info("Enter match details to get a predicted score from the selected model.")
            
            # Check if coming from Models page
            if 'navigate_to_predictions' in st.session_state and st.session_state.navigate_to_predictions:
                if 'selected_model' in st.session_state and st.session_state.selected_model in model_names:
                    st.success(f"🎯 Auto-selected: **{st.session_state.selected_model}** (from Models page)")
                    default_model_index = model_names.index(st.session_state.selected_model)
                else:
                    default_model_index = 0
                # Clear the navigation flag
                st.session_state.navigate_to_predictions = False
            else:
                default_model_index = 0
            
            # Model selection at the top
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_model = st.selectbox("🤖 Select Prediction Model:", model_names, 
                                            index=default_model_index,
                                            help="Choose which trained model to use for prediction")
            with col2:
                if selected_model and selected_model in model_info_dict:
                    model_info = model_info_dict[selected_model]
                    st.metric("Model R² Score", f"{model_info['r2']:.4f}")
            
            st.markdown("---")
            
            # Cached dropdown options
            teams = ['India', 'Pakistan', 'Australia', 'South Africa', 'England',
                    'New Zealand', 'Sri Lanka', 'West Indies', 'Bangladesh', 'Afghanistan']
            
            cities = ['Dubai', 'Mumbai', 'Colombo', 'Lahore', 'London', 'Durban', 'Delhi',
                     'Mirpur', 'Johannesburg', 'Centurion', 'Bangalore', 'Pallekele',
                     'Cape Town', 'Melbourne', 'Wellington', 'Hamilton', 'St Lucia', 'Barbados']
            
            # Create form layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🏏 Match Details**")
                batting_team = st.selectbox("Batting Team", teams, index=0)
                bowling_team = st.selectbox("Bowling Team", [team for team in teams if team != batting_team], index=0)
                city = st.selectbox("City", cities, index=0)
                current_score = st.number_input("Current Score", min_value=0, max_value=500, value=120, step=1)
                
            with col2:
                st.markdown("**📊 Match Situation**")
                overs_left = st.number_input("Overs Left", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
                wickets_left = st.number_input("Wickets Left", min_value=0, max_value=10, value=7, step=1)
                crr = st.number_input("Current Run Rate", min_value=0.0, max_value=20.0, value=6.0, step=0.1)
                last_five = st.number_input("Runs in Last 5 Overs", min_value=0, max_value=100, value=35, step=1)
            
            # Convert overs to balls
            balls_left = int((overs_left // 1) * 6 + round((overs_left % 1) * 10))
            
            # Prepare features
            features = {
                'batting_team': batting_team,
                'bowling_team': bowling_team,
                'city': city,
                'current_score': current_score,
                'balls_left': balls_left,
                'wickets_left': wickets_left,
                'crr': crr,
                'last_five': last_five
            }
            
            # Display conversion info
            st.info(f"📐 Conversion: {overs_left} overs = {balls_left} balls")
            
            # Prediction button with validation
            if st.button("🔮 Make Prediction", type="primary", use_container_width=True):
                if not selected_model:
                    st.error("❌ Please select a model first!")
                else:
                    with st.spinner(f"🔮 Getting prediction from {selected_model}..."):
                        result = make_single_prediction(features, selected_model)
                        if result:
                            # Success display with enhanced formatting
                            st.balloons()
                            st.success(f"✅ **{selected_model}** Predicted Final Score: **{result['prediction']:.0f} runs**")
                            
                            # Display input features in a nice format
                            st.subheader("📋 Prediction Summary")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(f"""
                                **🏏 Match Details:**
                                - **Batting Team:** {batting_team}
                                - **Bowling Team:** {bowling_team}
                                - **City:** {city}
                                """)
                            
                            with col2:
                                st.markdown(f"""
                                **📊 Current State:**
                                - **Current Score:** {current_score}
                                - **Overs Left:** {overs_left}
                                - **Wickets Left:** {wickets_left}
                                """)
                            
                            with col3:
                                st.markdown(f"""
                                **📈 Performance:**
                                - **Current RR:** {crr}
                                - **Last 5 Overs:** {last_five}
                                - **Model Used:** {selected_model}
                                """)
                        else:
                            st.error("❌ Failed to make prediction. Please check your inputs and try again.")
        
        with tab2:
            st.subheader("Batch Prediction")
            st.info("Upload a CSV file with match data to get predictions for multiple samples.")
            
            # Model selection for batch prediction
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_batch_model = st.selectbox("🤖 Select Model for Batch Prediction:", model_names,
                                                  help="Choose which trained model to use for batch predictions")
            with col2:
                if selected_batch_model and selected_batch_model in model_info_dict:
                    model_info = model_info_dict[selected_batch_model]
                    st.metric("Model R² Score", f"{model_info['r2']:.4f}")
            
            st.markdown("---")
            
            # Required columns info with better formatting
            with st.expander("📋 Required CSV Column Format", expanded=False):
                st.markdown("""
                **Required Columns (must match exactly):**
                
                | Column Name | Type | Description | Example |
                |-------------|------|-------------|---------|
                | `batting_team` | Text | Team names | India, Pakistan, Australia |
                | `bowling_team` | Text | Team names | England, South Africa |
                | `city` | Text | City names | Dubai, Mumbai, Colombo |
                | `current_score` | Number | Current score | 120, 95, 145 |
                | `balls_left` | Number | Balls remaining (not overs) | 60, 84, 42 |
                | `wickets_left` | Number | Wickets remaining | 7, 8, 6 |
                | `crr` | Number | Current run rate | 6.0, 5.68, 7.25 |
                | `last_five` | Number | Runs in last 5 overs | 35, 28, 45 |
                
                **💡 Tip:** Download our sample CSV below to see the correct format!
                """)
            
            # Download sample CSV button
            with open('sample_data.csv', 'rb') as f:
                st.download_button(
                    label="📥 Download Sample CSV Template",
                    data=f.read(),
                    file_name="cricket_prediction_template.csv",
                    mime="text/csv",
                    help="Download a sample CSV file with the correct format"
                )
            
            uploaded_file = st.file_uploader("📤 Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    # Read and validate CSV
                    df_preview = pd.read_csv(uploaded_file)
                    
                    # Required columns
                    required_columns = ['batting_team', 'bowling_team', 'city', 'current_score', 
                                      'balls_left', 'wickets_left', 'crr', 'last_five']
                    
                    # Check for missing columns
                    missing_columns = [col for col in required_columns if col not in df_preview.columns]
                    
                    if missing_columns:
                        st.error(f"❌ **Missing required columns:** {', '.join(missing_columns)}")
                        st.markdown("**Available columns in your file:**")
                        st.write(list(df_preview.columns))
                    else:
                        st.success("✅ All required columns found!")
                        
                        # Show file preview
                        st.subheader("📊 File Preview")
                        st.dataframe(df_preview.head(10), use_container_width=True)
                        
                        # Additional info
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Rows", len(df_preview))
                        with col2:
                            st.metric("Total Columns", len(df_preview.columns))
                        with col3:
                            st.metric("Missing Values", df_preview.isnull().sum().sum())
                        
                        # Check for missing values
                        if df_preview.isnull().sum().sum() > 0:
                            st.warning("⚠️ Your file contains missing values. This may affect prediction accuracy.")
                        
                        if st.button("🚀 Generate Batch Predictions", type="primary", use_container_width=True):
                            if not selected_batch_model:
                                st.error("❌ Please select a model for batch prediction first!")
                            else:
                                with st.spinner(f"🔮 Processing {len(df_preview)} predictions with {selected_batch_model}..."):
                                    # Reset file pointer
                                    uploaded_file.seek(0)
                                    result = make_batch_prediction(uploaded_file, selected_batch_model)
                                    
                                    if result:
                                        st.balloons()
                                        st.success(f"✅ **{selected_batch_model}** generated {result['num_predictions']} predictions successfully!")
                                        
                                        # Create results dataframe
                                        df_results = df_preview.copy()
                                        df_results['Predicted_Score'] = result['predictions']
                                        
                                        st.subheader("🎯 Batch Prediction Results")
                                        
                                        # Summary statistics with enhanced display
                                        col1, col2, col3, col4, col5 = st.columns(5)
                                        with col1:
                                            st.metric("Total Predictions", f"{result['num_predictions']}")
                                        with col2:
                                            st.metric("Average Score", f"{df_results['Predicted_Score'].mean():.1f}")
                                        with col3:
                                            st.metric("Min Score", f"{df_results['Predicted_Score'].min():.1f}")
                                        with col4:
                                            st.metric("Max Score", f"{df_results['Predicted_Score'].max():.1f}")
                                        with col5:
                                            st.metric("Score Range", f"{df_results['Predicted_Score'].max() - df_results['Predicted_Score'].min():.1f}")
                                        
                                        # Results table with filtering
                                        st.dataframe(
                                            df_results, 
                                            use_container_width=True,
                                            column_config={
                                                "Predicted_Score": st.column_config.NumberColumn(
                                                    "Predicted Score",
                                                    help="Final predicted score",
                                                    format="%.0f"
                                                )
                                            }
                                        )
                                        
                                        # Download results with better naming
                                        csv = df_results.to_csv(index=False)
                                        filename = f"predictions_{selected_batch_model}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
                                        st.download_button(
                                            label="📥 Download Prediction Results",
                                            data=csv,
                                            file_name=filename,
                                            mime="text/csv",
                                            help=f"Download results from {selected_batch_model} model",
                                            use_container_width=True
                                        )
                                        
                                        # Enhanced visualization
                                        if len(result['predictions']) > 1:
                                            st.subheader("📊 Prediction Analysis")
                                            
                                            col1, col2 = st.columns(2)
                                            
                                            with col1:
                                                fig_hist = px.histogram(
                                                    x=result['predictions'],
                                                    title=f"Score Distribution ({selected_batch_model})",
                                                    labels={'x': 'Predicted Score', 'y': 'Frequency'},
                                                    nbins=min(20, len(result['predictions'])//2),
                                                    color_discrete_sequence=['#1f77b4']
                                                )
                                                fig_hist.update_layout(showlegend=False)
                                                st.plotly_chart(fig_hist, use_container_width=True)
                                            
                                            with col2:
                                                fig_box = px.box(
                                                    y=result['predictions'],
                                                    title=f"Score Statistics ({selected_batch_model})",
                                                    labels={'y': 'Predicted Score'},
                                                    color_discrete_sequence=['#2ca02c']
                                                )
                                                fig_box.update_layout(showlegend=False)
                                                st.plotly_chart(fig_box, use_container_width=True)
                                    else:
                                        st.error("❌ Failed to generate predictions. Please check your file format and model selection.")
                        
                except Exception as e:
                    st.error(f"❌ Error reading CSV file: {str(e)}")
                    st.info("Please ensure your file is a valid CSV format with the required columns.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        <p>🤖 ML Models Dashboard | Built with Streamlit & FastAPI</p>
    </div>
    """, 
    unsafe_allow_html=True
) 