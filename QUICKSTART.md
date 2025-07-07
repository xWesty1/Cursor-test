# 🚀 Quick Start Guide - CLV Prediction Project

Welcome to the Customer Lifetime Value Prediction project! This guide will help you get up and running quickly.

## 📋 Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)
- 8GB+ RAM recommended
- 2GB free disk space

## ⚡ Quick Setup (5 minutes)

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd clv-prediction-platform
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Complete Pipeline
```bash
python run_pipeline.py
```

That's it! The pipeline will:
- ✅ Generate synthetic e-commerce data (10,000 customers)
- ✅ Engineer 50+ advanced features
- ✅ Train multiple ML models with optimization
- ✅ Generate comprehensive reports
- ✅ Launch interactive dashboard

## 🎯 What You'll Get

### Data Generated
- **Customers**: 10,000 realistic customer profiles
- **Transactions**: 50,000+ purchase records
- **Campaigns**: Marketing campaign interactions
- **Features**: 50+ engineered features for ML

### Models Trained
- Linear Regression variants (Ridge, Lasso, Elastic Net)
- Tree-based models (Random Forest, Gradient Boosting)
- Advanced models (XGBoost, LightGBM) with hyperparameter optimization
- Ensemble predictions for best performance

### Outputs
- **Dashboard**: Interactive Streamlit app at `http://localhost:8501`
- **Models**: Trained models saved in `models/` directory
- **Reports**: Business insights and technical documentation
- **Visualizations**: Charts and analysis in Jupyter notebooks

## 📊 Dashboard Features

Once the pipeline completes, you'll have access to:

- **Customer Overview**: Demographics and segment analysis
- **CLV Analysis**: Distribution and predictions
- **Behavioral Insights**: Purchase patterns and trends
- **Model Performance**: Evaluation metrics and comparisons
- **Business Insights**: Actionable recommendations
- **ROI Calculator**: Marketing campaign optimization

## 🛠️ Manual Steps (Optional)

If you prefer to run components individually:

### Generate Data
```bash
python src/data_engineering/data_generator.py
```

### Feature Engineering
```bash
python src/feature_engineering/feature_pipeline.py
```

### Train Models
```bash
python src/modeling/train_models.py
```

### Launch Dashboard
```bash
streamlit run src/dashboard/app.py
```

### Explore Analysis
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

## 📁 Project Structure

```
├── data/
│   ├── raw/          # Generated customer, transaction data
│   └── processed/    # Feature-engineered datasets
├── src/
│   ├── data_engineering/     # Data generation and ETL
│   ├── feature_engineering/  # Feature creation pipelines
│   ├── modeling/            # ML model training
│   └── dashboard/           # Streamlit business app
├── models/           # Trained models and results
├── notebooks/        # Jupyter analysis notebooks
├── config/          # Configuration files
└── tests/           # Unit tests
```

## 🎓 Skills Demonstrated

This project showcases:

### Technical Skills
- **Data Engineering**: ETL pipelines, data validation
- **Feature Engineering**: RFM analysis, behavioral features
- **Machine Learning**: Ensemble methods, hyperparameter tuning
- **MLOps**: Model versioning, evaluation, monitoring
- **Visualization**: Interactive dashboards, business insights

### Business Skills
- **Customer Analytics**: Segmentation, lifetime value
- **Marketing Optimization**: Campaign ROI, targeting
- **Revenue Analytics**: Prediction, forecasting
- **Strategic Insights**: Data-driven recommendations

## 🔧 Troubleshooting

### Common Issues

**ModuleNotFoundError**: Install missing packages
```bash
pip install -r requirements.txt
```

**Memory Error**: Reduce dataset size in `data_generator.py`
```python
# Change n_customers from 10000 to 5000
customers_df, transactions_df = generator.generate_complete_dataset(n_customers=5000)
```

**Port Already in Use**: Use different port for dashboard
```bash
streamlit run src/dashboard/app.py --server.port 8502
```

**Slow Training**: Reduce optimization trials
```python
# In train_models.py, change n_trials
xgb_results = self.optimize_xgboost(X_train, y_train, X_val, y_val, n_trials=20)
```

## 📞 Support

### File Structure Check
```bash
ls -la  # Should show README.md, run_pipeline.py, src/, etc.
```

### Dependency Check
```bash
python -c "import pandas, numpy, sklearn, xgboost, lightgbm, streamlit; print('All dependencies installed!')"
```

### Data Verification
```bash
ls data/raw/  # Should show customers.csv, transactions.csv, campaigns.csv
```

## 🎯 Next Steps

After running the project:

1. **Explore the Dashboard**: Navigate through different sections
2. **Review Model Results**: Check `models/evaluation_results.json`
3. **Analyze Business Insights**: Read generated reports
4. **Customize Parameters**: Modify `config/model_config.yaml`
5. **Extend Features**: Add new features in feature engineering
6. **Deploy to Cloud**: Set up cloud deployment for production

## 🏆 For Internship Applications

This project demonstrates:
- End-to-end data science workflow
- Production-ready code quality
- Business impact focus
- Advanced ML techniques
- Interactive visualization skills

**Portfolio highlights**: Include screenshots of the dashboard, model performance metrics, and business insights in your applications.

## 📈 Performance Benchmarks

Expected results on standard hardware:
- **Data Generation**: 2-3 minutes
- **Feature Engineering**: 3-5 minutes  
- **Model Training**: 10-15 minutes
- **Total Pipeline**: 15-25 minutes

Model performance targets:
- **RMSE**: < $150 (varies with data)
- **R²**: > 0.75
- **Business Impact**: 15%+ improvement over baseline

---

**Ready to impress recruiters!** 🎓 This project showcases industry-level data science skills perfect for internship applications.