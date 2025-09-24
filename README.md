# 🌞 Solar Power Predictor Pro

An advanced Streamlit application for predicting solar power generation using multiple machine learning models. Interactive visualizations, model comparison, and export functionality make it ideal for data scientists, engineers, and renewable energy analysts.

---

## 🚀 Features

- **Data Input**  
  - File upload or local CSV path  
  - Automatic header detection  

- **Data Overview**  
  - Dataset size, missing values, and memory usage metrics  
  - Data preview and descriptive statistics  

- **Exploratory Data Analysis (EDA)**  
  - Interactive histogram and box plot for target distribution  
  - Correlation heatmap of numeric features  

- **Model Training**  
  - Choose from Linear Regression, Ridge, Lasso, Decision Tree, or Random Forest  
  - Toggle feature scaling (StandardScaler)  
  - Custom test split and random seed  
  - Real-time training progress  

- **Model Evaluation**  
  - Metrics: R², RMSE, MAE, MSE  
  - Interactive Plotly charts:
    - Actual vs. Predicted scatter  
    - Residuals distribution histogram  
  - Feature importance bar chart for tree-based models  

- **Export & Download**  
  - Download trained model (.joblib) with timestamp  
  - Download prediction results (Actual, Predicted, Residual) as CSV  
  - Auto-generated model summary report  

---

## 🎨 Installation


---

## 🛠️ Usage Guide

1. **Configuration Panel**  
   - **Data Input**: Upload CSV or provide file path  
   - **Target Column**: Select the column to predict  
   - **Model Type**: Choose algorithm  
   - **Test Size & Seed**: Adjust test split and randomness  
   - **Scale Features**: Enable for normalization  
   - **Train** / **Reset** buttons  

2. **Data Overview Tab**  
   - View dataset metrics (rows, columns, missing, memory)  
   - Preview first 10 rows and summary statistics  

3. **EDA Tab**  
   - Histogram & box plot of target  
   - Correlation heatmap of features  

4. **Model Training Tab**  
   - Click **Train Model** to begin  
   - View model type, training time, and feature count  

5. **Results Tab**  
   - Inspect R², RMSE, MAE, MSE metrics  
   - Interactive Actual vs. Predicted and Residuals charts  
   - Feature importance for tree models  

6. **Export Tab**  
   - Download `.joblib` model file  
   - Download CSV of predictions and residuals  
   - View comprehensive model summary with timestamp  

---



---

## ⚙️ Configuration Options

| Option         | Description                             | Default           |
| -------------- | --------------------------------------- | ----------------- |
| **Data Input** | File upload or CSV path                 | `dataset.csv`     |
| **Target Col** | Column name to predict                  | Last column       |
| **Model Type** | `linear`, `ridge`, `lasso`, `decision_tree`, `random_forest` | `linear` |
| **Test Size**  | Fraction of data for testing            | `0.2`             |
| **Random Seed**| Seed for reproducibility                | `42`              |
| **Scale**      | Apply StandardScaler to features        | `False`           |

---

## 🤝 Contributing

1. Fork this repo  
2. Create a feature branch (`git checkout -b feature/my-feature`)  
3. Commit your changes (`git commit -m "Add my feature"`)  
4. Push to branch (`git push origin feature/my-feature`)  
5. Open a Pull Request  

---

## 🐛 Troubleshooting

- **Data load error**: Verify file path or upload valid CSV  
- **Missing target**: Ensure target column exists in header  
- **Installation issues**: Run `pip install -r requirements.txt`  

---

