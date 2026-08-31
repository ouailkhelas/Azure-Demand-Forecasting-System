#!/usr/bin/env python3
"""
Model Training Script for Demand Forecasting
Trains multiple ML models and selects best performer
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime
import json
import os

class ModelTraining:
    def __init__(self, train_file, test_file):
        print("📚 Loading data...")
        self.train_data = pd.read_csv(train_file)
        self.test_data = pd.read_csv(test_file)
        self.models = {}
        self.results = {}
        
        print(f"   Train shape: {self.train_data.shape}")
        print(f"   Test shape: {self.test_data.shape}")
    
    def prepare_features(self):
        """Prepare features and targets"""
        print("\n🔧 Preparing features...")
        
        # Define feature columns (exclude date, product_id, sales)
        self.feature_cols = [col for col in self.train_data.columns 
                            if col not in ['date', 'product_id', 'sales']]
        
        print(f"   Features: {len(self.feature_cols)}")
        print(f"   {self.feature_cols}")
        
        # Prepare train data
        self.X_train = self.train_data[self.feature_cols]
        self.y_train = self.train_data['sales']
        
        # Prepare test data
        self.X_test = self.test_data[self.feature_cols]
        self.y_test = self.test_data['sales']
        
        print(f"✅ Features prepared")
        return self
    
    def train_linear_regression(self):
        """Train Linear Regression model"""
        print("\n🤖 Training Linear Regression...")
        
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        
        # Predict
        y_pred = model.predict(self.X_test)
        
        # Evaluate
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        self.models['linear_regression'] = model
        self.results['linear_regression'] = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mse': mse
        }
        
        print(f"   R² Score: {r2:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        
        return self
    
    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n🤖 Training Random Forest...")
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)
        
        # Predict
        y_pred = model.predict(self.X_test)
        
        # Evaluate
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        self.models['random_forest'] = model
        self.results['random_forest'] = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mse': mse
        }
        
        print(f"   R² Score: {r2:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        
        # Feature importance
        importances = model.feature_importances_
        top_features = sorted(zip(self.feature_cols, importances), 
                             key=lambda x: x[1], reverse=True)[:5]
        print(f"   Top 5 features:")
        for feat, imp in top_features:
            print(f"      {feat}: {imp:.4f}")
        
        return self
    
    def train_gradient_boosting(self):
        """Train Gradient Boosting model"""
        print("\n🤖 Training Gradient Boosting...")
        
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model.fit(self.X_train, self.y_train)
        
        # Predict
        y_pred = model.predict(self.X_test)
        
        # Evaluate
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        self.models['gradient_boosting'] = model
        self.results['gradient_boosting'] = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mse': mse
        }
        
        print(f"   R² Score: {r2:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        
        return self
    
    def select_best_model(self):
        """Select best performing model"""
        print("\n🏆 Selecting best model...")
        
        # Find model with highest R² score
        best_model_name = max(self.results, 
                             key=lambda x: self.results[x]['r2'])
        best_score = self.results[best_model_name]['r2']
        
        print(f"\n📊 Model Performance Comparison:")
        print("-" * 50)
        print(f"{'Model':<20} {'R² Score':<15} {'RMSE':<15}")
        print("-" * 50)
        
        for model_name, metrics in self.results.items():
            marker = "✅" if model_name == best_model_name else "  "
            print(f"{marker} {model_name:<17} {metrics['r2']:<14.4f} {metrics['rmse']:<14.4f}")
        
        print("-" * 50)
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   R² Score: {best_score:.4f}")
        
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        return self
    
    def save_model(self, output_file='models/demand_forecast_model.pkl'):
        """Save trained model"""
        print(f"\n💾 Saving model to {output_file}...")
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'feature_columns': self.feature_cols,
            'metrics': self.results[self.best_model_name],
            'training_date': datetime.now().isoformat()
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved successfully")
        
        # Save metrics as JSON
        metrics_file = output_file.replace('.pkl', '_metrics.json')
        with open(metrics_file, 'w') as f:
            # Convert numpy types to float
            metrics_clean = {
                k: {mk: float(mv) for mk, mv in v.items()} 
                for k, v in self.results.items()
            }
            json.dump(metrics_clean, f, indent=2)
        
        print(f"✅ Metrics saved to {metrics_file}")
        
        return self

def main():
    """Main execution"""
    print("=" * 60)
    print("   DEMAND FORECASTING - MODEL TRAINING")
    print("=" * 60)
    
    try:
        # Check if data files exist
        if not os.path.exists('data/train_data.csv'):
            print("❌ Error: train_data.csv not found")
            print("   Run data preparation first:")
            print("   python scripts/data_preparation.py")
            return 1
        
        # Initialize training
        trainer = ModelTraining('data/train_data.csv', 'data/test_data.csv')
        
        # Train models
        (trainer.prepare_features()
             .train_linear_regression()
             .train_random_forest()
             .train_gradient_boosting()
             .select_best_model()
             .save_model())
        
        print("\n✅ Model training completed successfully!")
        print("\nNext step: Deploy model to Azure")
        print("   Command: python scripts/deploy_model.py")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
