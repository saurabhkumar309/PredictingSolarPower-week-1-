import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import time
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Solar Power Predictor Pro",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #FF6B35;
    text-align: center;
    padding: 1rem 0;
    background: linear-gradient(90deg, #FF6B35, #F7931E);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: bold;
}

.metric-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}

.feature-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #FF6B35;
    margin: 0.5rem 0;
}

.sidebar-header {
    background: linear-gradient(135deg, #FF6B35, #F7931E);
    color: white;
    padding: 0.5rem;
    border-radius: 5px;
    text-align: center;
    font-weight: bold;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

def load_data(path):
    """Load dataset with error handling"""
    try:
        df = pd.read_csv(path)
        return df, None
    except Exception as e:
        return None, str(e)

def get_data_insights(df):
    """Generate comprehensive data insights"""
    insights = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
    }
    return insights

def preprocess_data(df, target_col, scale_features=False):
    """Enhanced preprocessing with scaling option"""
    cols = df.columns.tolist()
    if target_col not in cols:
        target = cols[-1]
        st.warning(f"⚠️ '{target_col}' not found; using '{target}' instead")
    else:
        target = target_col
    
    X = df.drop(columns=[target])
    y = df[target]
    
    if scale_features:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X), 
            columns=X.columns, 
            index=X.index
        )
        return X_scaled, y, scaler
    
    return X, y, None

def train_model(X_train, y_train, model_type='linear'):
    """Train different types of models"""
    models = {
        'linear': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=1.0),
        'decision_tree': DecisionTreeRegressor(random_state=42),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    model = models[model_type]
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    return model, training_time

def evaluate_model(model, X_test, y_test):
    """Comprehensive model evaluation"""
    predictions = model.predict(X_test)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
        'mae': mean_absolute_error(y_test, predictions),
        'r2': r2_score(y_test, predictions),
        'mse': mean_squared_error(y_test, predictions)
    }
    
    return predictions, metrics

def create_correlation_heatmap(df):
    """Create interactive correlation heatmap"""
    corr_matrix = df.corr()
    fig = px.imshow(
        corr_matrix, 
        text_auto=True, 
        aspect="auto",
        title="Feature Correlation Matrix"
    )
    fig.update_layout(width=800, height=600)
    return fig

def create_actual_vs_predicted_plot(actual, predicted):
    """Create interactive actual vs predicted plot"""
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=actual, y=predicted,
        mode='markers',
        name='Predictions',
        marker=dict(
            color='rgba(55, 128, 191, 0.7)',
            size=8,
            line=dict(width=1, color='white')
        ),
        hovertemplate='Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>'
    ))
    
    # Add perfect prediction line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    fig.update_layout(
        title="Actual vs Predicted Values",
        xaxis_title="Actual Generated Power (kW)",
        yaxis_title="Predicted Generated Power (kW)",
        width=600, height=500
    )
    
    return fig

def create_residuals_plot(actual, predicted):
    """Create residuals distribution plot"""
    residuals = actual - predicted
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=30,
        name='Residuals',
        marker_color='steelblue',
        opacity=0.7
    ))
    
    fig.update_layout(
        title="Residuals Distribution",
        xaxis_title="Residual (Actual - Predicted)",
        yaxis_title="Frequency",
        width=600, height=500
    )
    
    return fig

def create_feature_importance_plot(model, feature_names):
    """Create feature importance plot for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(
            importance_df, 
            x='importance', 
            y='feature',
            orientation='h',
            title="Feature Importance"
        )
        fig.update_layout(width=600, height=400)
        return fig
    return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🌞 Solar Power Predictor Pro</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.markdown('<div class="sidebar-header">⚙️ Configuration Panel</div>', unsafe_allow_html=True)
    
    # File upload or path input
    upload_option = st.sidebar.radio("Data Input Method:", ["File Upload", "File Path"])
    
    if upload_option == "File Upload":
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file:
            df, error = load_data(uploaded_file)
        else:
            df, error = None, "Please upload a CSV file"
    else:
        data_path = st.sidebar.text_input("Dataset CSV path:", "dataset.csv")
        df, error = load_data(data_path)
    
    if error:
        st.error(f"❌ Error loading data: {error}")
        return
    
    if df is None:
        st.info("👆 Please provide a dataset to begin analysis")
        return
    
    # Configuration options
    target_col = st.sidebar.selectbox("Target Column:", df.columns.tolist(), 
                                     index=len(df.columns)-1)
    
    model_type = st.sidebar.selectbox("Model Type:", 
                                     ['linear', 'ridge', 'lasso', 'decision_tree', 'random_forest'])
    
    test_size = st.sidebar.slider("Test Set Size:", 0.1, 0.5, 0.2, 0.05)
    random_state = st.sidebar.number_input("Random Seed:", 0, 1000, 42)
    scale_features = st.sidebar.checkbox("Scale Features", value=False)
    
    # Action buttons
    col1, col2 = st.sidebar.columns(2)
    retrain = col1.button("🔄 Train Model", use_container_width=True)
    reset = col2.button("🏠 Reset", use_container_width=True)
    
    if reset:
        st.session_state.clear()
        st.experimental_rerun()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", "🔍 EDA", "🤖 Model Training", "📈 Results", "💾 Export"
    ])
    
    with tab1:
        st.subheader("Dataset Overview")
        
        # Data insights
        insights = get_data_insights(df)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <h3>{insights['shape'][0]:,}</h3>
                <p>Rows</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <h3>{insights['shape'][1]}</h3>
                <p>Columns</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <h3>{insights['missing_values']}</h3>
                <p>Missing Values</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <h3>{insights['memory_usage']:.1f} MB</h3>
                <p>Memory Usage</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Basic statistics
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
    
    with tab2:
        st.subheader("Exploratory Data Analysis")
        
        # Target distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x=target_col, nbins=30, 
                             title=f"Distribution of {target_col}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, y=target_col, 
                        title=f"Box Plot of {target_col}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_fig = create_correlation_heatmap(df[numeric_cols])
            st.plotly_chart(corr_fig, use_container_width=True)
    
    with tab3:
        st.subheader("Model Training")
        
        if retrain or "model_results" not in st.session_state:
            with st.spinner("🔄 Training model... Please wait"):
                # Preprocess data
                X, y, scaler = preprocess_data(df, target_col, scale_features)
                
                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                
                # Train model
                model, training_time = train_model(X_train, y_train, model_type)
                
                # Evaluate model
                predictions, metrics = evaluate_model(model, X_test, y_test)
                
                # Store results
                st.session_state.model_results = {
                    'model': model,
                    'scaler': scaler,
                    'predictions': predictions,
                    'y_test': y_test,
                    'metrics': metrics,
                    'training_time': training_time,
                    'feature_names': X.columns.tolist(),
                    'model_type': model_type
                }
            
            st.success(f"✅ Model trained successfully in {training_time:.2f} seconds!")
        
        if "model_results" in st.session_state:
            results = st.session_state.model_results
            
            st.markdown(f"""
            <div class="feature-card">
                <h4>🤖 Model Information</h4>
                <p><strong>Type:</strong> {results['model_type'].replace('_', ' ').title()}</p>
                <p><strong>Training Time:</strong> {results['training_time']:.3f} seconds</p>
                <p><strong>Features Used:</strong> {len(results['feature_names'])}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.subheader("Model Performance")
        
        if "model_results" in st.session_state:
            results = st.session_state.model_results
            metrics = results['metrics']
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <h3>{metrics['r2']:.3f}</h3>
                    <p>R² Score</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <h3>{metrics['rmse']:.2f}</h3>
                    <p>RMSE</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-container">
                    <h3>{metrics['mae']:.2f}</h3>
                    <p>MAE</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-container">
                    <h3>{metrics['mse']:.2f}</h3>
                    <p>MSE</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualization plots
            col1, col2 = st.columns(2)
            
            with col1:
                actual_vs_pred_fig = create_actual_vs_predicted_plot(
                    results['y_test'].values, results['predictions']
                )
                st.plotly_chart(actual_vs_pred_fig, use_container_width=True)
            
            with col2:
                residuals_fig = create_residuals_plot(
                    results['y_test'].values, results['predictions']
                )
                st.plotly_chart(residuals_fig, use_container_width=True)
            
            # Feature importance for tree-based models
            feature_imp_fig = create_feature_importance_plot(
                results['model'], results['feature_names']
            )
            if feature_imp_fig:
                st.subheader("Feature Importance")
                st.plotly_chart(feature_imp_fig, use_container_width=True)
        else:
            st.info("👆 Please train a model first to see results")
    
    with tab5:
        st.subheader("Export & Download")
        
        if "model_results" in st.session_state:
            results = st.session_state.model_results
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Model download
                model_data = joblib.dump(results['model'], 'trained_model.joblib')
                with open('trained_model.joblib', 'rb') as f:
                    st.download_button(
                        label="📥 Download Trained Model",
                        data=f.read(),
                        file_name=f"solar_power_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib",
                        mime="application/octet-stream"
                    )
            
            with col2:
                # Results CSV
                results_df = pd.DataFrame({
                    'Actual': results['y_test'].values,
                    'Predicted': results['predictions'],
                    'Residual': results['y_test'].values - results['predictions']
                })
                
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Results CSV",
                    data=csv_data,
                    file_name=f"prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            # Model summary report
            st.subheader("Model Summary Report")
            st.markdown(f"""
            ### Solar Power Prediction Model Report
            **Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            **Model Configuration:**
            - Model Type: {results['model_type'].replace('_', ' ').title()}
            - Features Scaling: {scale_features}
            - Test Set Size: {test_size:.0%}
            - Random Seed: {random_state}
            
            **Performance Metrics:**
            - R² Score: {metrics['r2']:.4f}
            - RMSE: {metrics['rmse']:.4f}
            - MAE: {metrics['mae']:.4f}
            - MSE: {metrics['mse']:.4f}
            
            **Training Details:**
            - Training Time: {results['training_time']:.3f} seconds
            - Number of Features: {len(results['feature_names'])}
            - Training Samples: {len(results['y_test']) * (1-test_size) / test_size:.0f}
            - Test Samples: {len(results['y_test'])}
            """)
        else:
            st.info("👆 Please train a model first to export results")

if __name__ == "__main__":
    main()
