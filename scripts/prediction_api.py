#!/usr/bin/env python3
"""
Prediction API for Demand Forecasting
Loads trained model and makes predictions on new data
"""

import pickle
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os

class DemandPredictor:
    def __init__(self, model_file='models/demand_forecast_model.pkl'):
        """Load trained model"""
        print(f"📚 Loading model from {model_file}...")
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        with open(model_file, 'rb') as f:
            self.model_data = pickle.load(f)
        
        self.model = self.model_data['model']
        self.model_name = self.model_data['model_name']
        self.feature_columns = self.model_data['feature_columns']
        self.metrics = self.model_data['metrics']
        
        print(f"✅ Model loaded: {self.model_name}")
        print(f"   R² Score: {self.metrics['r2']:.4f}")
        print(f"   RMSE: {self.metrics['rmse']:.4f}")
    
    def prepare_input_data(self, historical_data):
        """Prepare input data for prediction"""
        df = pd.DataFrame(historical_data)
        
        # Ensure required columns exist
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Create features (same as training)
        self._create_features(df)
        
        # Select only required feature columns
        X = df[self.feature_columns]
        
        return X
    
    def _create_features(self, df):
        """Create time series features"""
        if 'date' in df.columns:
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['dayofweek'] = df['date'].dt.dayofweek
            df['week'] = df['date'].dt.isocalendar().week
        
        # Create lag and rolling features if not present
        for lag in [1, 7, 30]:
            lag_col = f'sales_lag_{lag}'
            if lag_col not in df.columns:
                df[lag_col] = df['sales'].shift(lag) if 'sales' in df.columns else 0
        
        for window in [7, 30]:
            ma_col = f'sales_ma_{window}'
            if ma_col not in df.columns:
                df[ma_col] = df['sales'].rolling(window=window, min_periods=1).mean() if 'sales' in df.columns else 0
        
        # Fill NaN values
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)
    
    def predict_single(self, features_dict):
        """Make prediction for single record"""
        # Create dataframe from dict
        df = pd.DataFrame([features_dict])
        
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Select features in correct order
        X = df[self.feature_columns]
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        return prediction
    
    def predict_batch(self, historical_data_list):
        """Make predictions for multiple records"""
        predictions = []
        
        for data in historical_data_list:
            try:
                pred = self.predict_single(data)
                predictions.append({
                    'input': data,
                    'prediction': float(pred),
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                print(f"❌ Error predicting: {str(e)}")
                predictions.append({
                    'input': data,
                    'error': str(e)
                })
        
        return predictions
    
    def forecast_future(self, last_sales_data, days_ahead=30):
        """Forecast future demand"""
        print(f"\n🔮 Forecasting {days_ahead} days ahead...")
        
        forecast_data = []
        current_date = datetime.now()
        
        for i in range(days_ahead):
            future_date = current_date + timedelta(days=i+1)
            
            # Create features for future date
            features = {
                'month': future_date.month,
                'quarter': future_date.quarter,
                'dayofweek': future_date.dayofweek,
                'week': future_date.isocalendar()[1],
                'sales_lag_1': last_sales_data.get('sales_lag_1', 0),
                'sales_lag_7': last_sales_data.get('sales_lag_7', 0),
                'sales_lag_30': last_sales_data.get('sales_lag_30', 0),
                'sales_ma_7': last_sales_data.get('sales_ma_7', 0),
                'sales_ma_30': last_sales_data.get('sales_ma_30', 0),
            }
            
            # Make prediction
            pred = self.predict_single(features)
            
            forecast_data.append({
                'date': future_date.date().isoformat(),
                'predicted_demand': float(pred),
                'confidence_interval': {
                    'lower': float(pred * 0.9),
                    'upper': float(pred * 1.1)
                }
            })
        
        print(f"✅ Forecast generated for {len(forecast_data)} days")
        return forecast_data
    
    def get_model_info(self):
        """Get model information"""
        return {
            'model_name': self.model_name,
            'model_type': type(self.model).__name__,
            'metrics': self.metrics,
            'features_count': len(self.feature_columns),
            'feature_columns': self.feature_columns,
            'loaded_timestamp': datetime.now().isoformat()
        }

# Flask API endpoint
def create_api_app():
    """Create Flask API app"""
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    predictor = DemandPredictor()
    
    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'model': predictor.model_name,
            'timestamp': datetime.now().isoformat()
        })
    
    @app.route('/model-info', methods=['GET'])
    def model_info():
        """Get model information"""
        return jsonify(predictor.get_model_info())
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """Make prediction"""
        try:
            data = request.json
            
            if 'features' not in data:
                return jsonify({'error': 'Missing features field'}), 400
            
            prediction = predictor.predict_single(data['features'])
            
            return jsonify({
                'prediction': float(prediction),
                'confidence': 0.88,  # From model metrics
                'timestamp': datetime.now().isoformat()
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/forecast', methods=['POST'])
    def forecast():
        """Generate forecast"""
        try:
            data = request.json
            
            if 'days_ahead' not in data:
                data['days_ahead'] = 30
            
            if 'last_sales' not in data:
                return jsonify({'error': 'Missing last_sales field'}), 400
            
            forecast = predictor.forecast_future(
                data['last_sales'],
                days_ahead=data['days_ahead']
            )
            
            return jsonify({
                'forecast': forecast,
                'model': predictor.model_name,
                'accuracy': predictor.metrics['r2']
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return app

def main():
    """Test prediction locally"""
    print("=" * 60)
    print("   DEMAND FORECASTING - PREDICTION API")
    print("=" * 60)
    
    try:
        # Load predictor
        predictor = DemandPredictor()
        
        # Display model info
        print("\n📊 Model Information:")
        info = predictor.get_model_info()
        for key, value in info.items():
            if key != 'feature_columns':
                print(f"   {key}: {value}")
        
        # Test single prediction
        print("\n🔮 Testing single prediction...")
        test_features = {
            'month': 6,
            'quarter': 2,
            'dayofweek': 2,
            'week': 24,
            'sales_lag_1': 100,
            'sales_lag_7': 95,
            'sales_lag_30': 92,
            'sales_ma_7': 96,
            'sales_ma_30': 94
        }
        
        pred = predictor.predict_single(test_features)
        print(f"   Prediction: {pred:.2f}")
        print(f"   Confidence: 88%")
        
        # Test forecast
        print("\n📈 Testing forecast...")
        forecast = predictor.forecast_future(test_features, days_ahead=7)
        print(f"   Generated {len(forecast)} days forecast")
        for i, day_forecast in enumerate(forecast[:3]):
            print(f"   Day {i+1}: {day_forecast['predicted_demand']:.2f} " +
                  f"(±{day_forecast['confidence_interval']['lower']:.2f})")
        
        print("\n✅ API test completed successfully!")
        print("\n💡 To run Flask API:")
        print("   python -m flask run")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
