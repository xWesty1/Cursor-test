"""
Advanced MLOps Pipeline for CLV Prediction

This module implements multiple machine learning models with hyperparameter optimization,
cross-validation, and comprehensive evaluation for CLV prediction.
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import xgboost as xgb
import lightgbm as lgb
import optuna
import shap

warnings.filterwarnings('ignore')


class CLVModelTrainer:
    """Comprehensive CLV model training and evaluation pipeline."""
    
    def __init__(self, random_state: int = 42):
        """Initialize the model trainer."""
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.evaluation_results = {}
        self.feature_importance = {}
        self.best_model = None
        self.best_model_name = None
        
    def prepare_data(self, feature_matrix_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load and prepare data for training."""
        
        print("Loading feature matrix...")
        df = pd.read_csv(feature_matrix_path)
        
        # Separate features and target
        X = df.drop(['customer_id', 'clv_target'], axis=1)
        y = df['clv_target']
        
        # Get feature names
        feature_names = X.columns.tolist()
        
        print(f"Data shape: {X.shape}")
        print(f"Target statistics:")
        print(f"  Mean: ${y.mean():.2f}")
        print(f"  Std: ${y.std():.2f}")
        print(f"  Min: ${y.min():.2f}")
        print(f"  Max: ${y.max():.2f}")
        
        return X.values, y.values, feature_names
    
    def train_linear_models(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train linear regression models."""
        
        linear_models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0, random_state=self.random_state),
            'lasso': Lasso(alpha=1.0, random_state=self.random_state),
            'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=self.random_state)
        }
        
        results = {}
        
        for name, model in linear_models.items():
            print(f"Training {name}...")
            
            # Scale features for linear models
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            train_pred = model.predict(X_train_scaled)
            val_pred = model.predict(X_val_scaled)
            
            # Evaluation
            results[name] = {
                'model': model,
                'scaler': scaler,
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                'train_mae': mean_absolute_error(y_train, train_pred),
                'val_mae': mean_absolute_error(y_val, val_pred),
                'train_r2': r2_score(y_train, train_pred),
                'val_r2': r2_score(y_val, val_pred)
            }
            
            self.models[name] = model
            self.scalers[name] = scaler
        
        return results
    
    def optimize_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray, n_trials: int = 100) -> Dict[str, Any]:
        """Optimize XGBoost with Optuna."""
        
        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': self.random_state
            }
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                     early_stopping_rounds=50, verbose=False)
            
            val_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            return rmse
        
        print("Optimizing XGBoost...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_params['random_state'] = self.random_state
        
        # Train final model with best parameters
        best_model = xgb.XGBRegressor(**best_params)
        best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                      early_stopping_rounds=50, verbose=False)
        
        # Predictions
        train_pred = best_model.predict(X_train)
        val_pred = best_model.predict(X_val)
        
        results = {
            'model': best_model,
            'best_params': best_params,
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'val_mae': mean_absolute_error(y_val, val_pred),
            'train_r2': r2_score(y_train, train_pred),
            'val_r2': r2_score(y_val, val_pred)
        }
        
        self.models['xgboost'] = best_model
        return results
    
    def optimize_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray, n_trials: int = 100) -> Dict[str, Any]:
        """Optimize LightGBM with Optuna."""
        
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': self.random_state,
                'verbose': -1
            }
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            
            val_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            return rmse
        
        print("Optimizing LightGBM...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_params['random_state'] = self.random_state
        best_params['verbose'] = -1
        
        # Train final model with best parameters
        best_model = lgb.LGBMRegressor(**best_params)
        best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        
        # Predictions
        train_pred = best_model.predict(X_train)
        val_pred = best_model.predict(X_val)
        
        results = {
            'model': best_model,
            'best_params': best_params,
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'val_mae': mean_absolute_error(y_val, val_pred),
            'train_r2': r2_score(y_train, train_pred),
            'val_r2': r2_score(y_val, val_pred)
        }
        
        self.models['lightgbm'] = best_model
        return results
    
    def train_ensemble_models(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train ensemble models."""
        
        ensemble_models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200, max_depth=10, random_state=self.random_state
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1, random_state=self.random_state
            )
        }
        
        results = {}
        
        for name, model in ensemble_models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            # Evaluation
            results[name] = {
                'model': model,
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                'train_mae': mean_absolute_error(y_train, train_pred),
                'val_mae': mean_absolute_error(y_val, val_pred),
                'train_r2': r2_score(y_train, train_pred),
                'val_r2': r2_score(y_val, val_pred)
            }
            
            self.models[name] = model
        
        return results
    
    def create_ensemble_predictions(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Create ensemble predictions from multiple models."""
        
        print("Creating ensemble predictions...")
        
        # Get predictions from all models
        predictions = {}
        
        for name, model in self.models.items():
            if name in ['linear_regression', 'ridge', 'lasso', 'elastic_net']:
                # Use scaled features for linear models
                scaler = self.scalers[name]
                X_val_scaled = scaler.transform(X_val)
                pred = model.predict(X_val_scaled)
            else:
                pred = model.predict(X_val)
            predictions[name] = pred
        
        # Simple average ensemble
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        
        # Weighted ensemble (higher weight for better models)
        weights = []
        for name in predictions.keys():
            if name in self.evaluation_results:
                # Use inverse of validation RMSE as weight
                weight = 1 / (self.evaluation_results[name]['val_rmse'] + 1e-6)
                weights.append(weight)
            else:
                weights.append(1.0)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        weighted_ensemble_pred = np.average(list(predictions.values()), axis=0, weights=weights)
        
        # Evaluation
        ensemble_results = {
            'simple_ensemble': {
                'predictions': ensemble_pred,
                'rmse': np.sqrt(mean_squared_error(y_val, ensemble_pred)),
                'mae': mean_absolute_error(y_val, ensemble_pred),
                'r2': r2_score(y_val, ensemble_pred)
            },
            'weighted_ensemble': {
                'predictions': weighted_ensemble_pred,
                'rmse': np.sqrt(mean_squared_error(y_val, weighted_ensemble_pred)),
                'mae': mean_absolute_error(y_val, weighted_ensemble_pred),
                'r2': r2_score(y_val, weighted_ensemble_pred),
                'weights': dict(zip(predictions.keys(), weights))
            }
        }
        
        return ensemble_results
    
    def analyze_feature_importance(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Analyze feature importance using SHAP."""
        
        print("Analyzing feature importance...")
        
        feature_importance = {}
        
        # Tree-based models have built-in feature importance
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_importance[name] = dict(zip(feature_names, importance))
        
        # SHAP analysis for best model
        if self.best_model and hasattr(self.best_model, 'predict'):
            try:
                # Sample data for SHAP (to speed up computation)
                sample_size = min(1000, X.shape[0])
                sample_idx = np.random.choice(X.shape[0], sample_size, replace=False)
                X_sample = X[sample_idx]
                
                if hasattr(self.best_model, 'feature_importances_'):
                    # Tree-based model
                    explainer = shap.TreeExplainer(self.best_model)
                    shap_values = explainer.shap_values(X_sample)
                else:
                    # Linear model
                    if self.best_model_name in self.scalers:
                        X_sample = self.scalers[self.best_model_name].transform(X_sample)
                    explainer = shap.LinearExplainer(self.best_model, X_sample)
                    shap_values = explainer.shap_values(X_sample)
                
                # Calculate mean absolute SHAP values
                mean_shap = np.mean(np.abs(shap_values), axis=0)
                feature_importance['shap'] = dict(zip(feature_names, mean_shap))
                
            except Exception as e:
                print(f"SHAP analysis failed: {e}")
        
        self.feature_importance = feature_importance
        return feature_importance
    
    def cross_validate_best_model(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict[str, float]:
        """Perform cross-validation on the best model."""
        
        if not self.best_model:
            print("No best model found. Please train models first.")
            return {}
        
        print(f"Cross-validating {self.best_model_name}...")
        
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        if self.best_model_name in self.scalers:
            # Scale features for linear models
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            scores = cross_val_score(self.best_model, X_scaled, y, cv=kfold, 
                                   scoring='neg_mean_squared_error')
        else:
            scores = cross_val_score(self.best_model, X, y, cv=kfold, 
                                   scoring='neg_mean_squared_error')
        
        rmse_scores = np.sqrt(-scores)
        
        cv_results = {
            'mean_rmse': rmse_scores.mean(),
            'std_rmse': rmse_scores.std(),
            'individual_scores': rmse_scores.tolist()
        }
        
        return cv_results
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray, feature_names: List[str],
                        test_size: float = 0.2) -> Dict[str, Any]:
        """Train all models and return comprehensive results."""
        
        print("Splitting data...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        all_results = {}
        
        # Train linear models
        linear_results = self.train_linear_models(X_train, y_train, X_val, y_val)
        all_results.update(linear_results)
        
        # Train ensemble models
        ensemble_results = self.train_ensemble_models(X_train, y_train, X_val, y_val)
        all_results.update(ensemble_results)
        
        # Optimize XGBoost and LightGBM
        xgb_results = self.optimize_xgboost(X_train, y_train, X_val, y_val, n_trials=50)
        lgb_results = self.optimize_lightgbm(X_train, y_train, X_val, y_val, n_trials=50)
        
        all_results['xgboost'] = xgb_results
        all_results['lightgbm'] = lgb_results
        
        self.evaluation_results = all_results
        
        # Find best model based on validation RMSE
        best_rmse = float('inf')
        for name, results in all_results.items():
            val_rmse = results['val_rmse']
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                self.best_model_name = name
                self.best_model = results['model']
        
        print(f"\nBest model: {self.best_model_name} (Validation RMSE: {best_rmse:.2f})")
        
        # Create ensemble predictions
        ensemble_results = self.create_ensemble_predictions(X_val, y_val)
        all_results.update(ensemble_results)
        
        # Feature importance analysis
        self.analyze_feature_importance(X, feature_names)
        
        # Cross-validation
        cv_results = self.cross_validate_best_model(X, y)
        all_results['cross_validation'] = cv_results
        
        return all_results
    
    def save_models(self, save_dir: str = 'models/'):
        """Save trained models and results."""
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, f'{save_dir}/{name}_model.pkl')
        
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f'{save_dir}/{name}_scaler.pkl')
        
        # Save best model separately
        if self.best_model:
            joblib.dump(self.best_model, f'{save_dir}/best_model.pkl')
            
            # Save model metadata
            metadata = {
                'best_model_name': self.best_model_name,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(f'{save_dir}/model_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Save evaluation results
        # Convert numpy arrays to lists for JSON serialization
        results_for_json = {}
        for model_name, results in self.evaluation_results.items():
            if isinstance(results, dict):
                clean_results = {}
                for key, value in results.items():
                    if isinstance(value, np.ndarray):
                        clean_results[key] = value.tolist()
                    elif isinstance(value, (np.int64, np.float64)):
                        clean_results[key] = float(value)
                    elif key != 'model':  # Skip model objects
                        clean_results[key] = value
                results_for_json[model_name] = clean_results
        
        with open(f'{save_dir}/evaluation_results.json', 'w') as f:
            json.dump(results_for_json, f, indent=2)
        
        # Save feature importance
        with open(f'{save_dir}/feature_importance.json', 'w') as f:
            json.dump(self.feature_importance, f, indent=2)
        
        print(f"Models and results saved to {save_dir}")


def main():
    """Main training pipeline."""
    
    print("=== CLV Model Training Pipeline ===\n")
    
    # Initialize trainer
    trainer = CLVModelTrainer(random_state=42)
    
    # Load data
    X, y, feature_names = trainer.prepare_data('data/processed/feature_matrix.csv')
    
    # Train all models
    results = trainer.train_all_models(X, y, feature_names)
    
    # Print results summary
    print("\n=== Model Performance Summary ===")
    print(f"{'Model':<20} {'Val RMSE':<10} {'Val MAE':<10} {'Val R²':<10}")
    print("-" * 50)
    
    for name, result in results.items():
        if isinstance(result, dict) and 'val_rmse' in result:
            rmse = result['val_rmse']
            mae = result.get('val_mae', 0)
            r2 = result.get('val_r2', 0)
            print(f"{name:<20} {rmse:<10.2f} {mae:<10.2f} {r2:<10.3f}")
    
    # Save models
    trainer.save_models()
    
    print(f"\nTraining complete! Best model: {trainer.best_model_name}")


if __name__ == "__main__":
    main()