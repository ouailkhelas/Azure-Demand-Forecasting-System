#check models name first
import azure.functions as func
import pickle
import json
import os
from datetime import datetime, timedelta
import logging
from azure.storage.blob import BlobClient

logger = logging.getLogger(__name__)

model_data = None
model = None
feature_columns = None

def load_model():
    """Load model from Azure Blob Storage"""
    global model_data, model, feature_columns
    
    if model is None:
        logger.info("Loading model from Blob Storage...")
        
        try:
            connection_string = os.environ['AZURE_STORAGE_CONNECTION_STRING']
            
            blob_client = BlobClient.from_connection_string(
                connection_string,
                container_name='models',
                blob_name='demand_forecast_model.pkl'
            )
            
            model_bytes = blob_client.download_blob().readall()
            
            model_data = pickle.loads(model_bytes)
            model = model_data['model']
            feature_columns = model_data['feature_columns']
            
            logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            raise

def create_features(data_dict):
    """Create features from input data"""
    features = {}
    
    if 'date' in data_dict:
        date_obj = datetime.fromisoformat(data_dict['date'])
        features['month'] = date_obj.month
        features['quarter'] = (date_obj.month - 1) // 3 + 1
        features['dayofweek'] = date_obj.weekday()
        features['week'] = date_obj.isocalendar()[1]
    else:
        today = datetime.now()
        features['month'] = today.month
        features['quarter'] = (today.month - 1) // 3 + 1
        features['dayofweek'] = today.weekday()
        features['week'] = today.isocalendar()[1]
    
    features['sales_lag_1'] = data_dict.get('sales_lag_1', 0)
    features['sales_lag_7'] = data_dict.get('sales_lag_7', 0)
    features['sales_lag_30'] = data_dict.get('sales_lag_30', 0)
    
    features['sales_ma_7'] = data_dict.get('sales_ma_7', 0)
    features['sales_ma_30'] = data_dict.get('sales_ma_30', 0)
    
    return features

def make_prediction(features):
    """Make prediction using loaded model"""
    import numpy as np
    
    feature_array = np.array([[features[col] for col in feature_columns]])
    
    prediction = model.predict(feature_array)[0]
    
    return float(prediction)

app = func.FunctionApp()

@app.route(route='health', methods=['GET'])
def health(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint"""
    return func.HttpResponse(
        json.dumps({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat()
        }),
        status_code=200,
        mimetype='application/json'
    )

@app.route(route='predict', methods=['POST'])
def predict(req: func.HttpRequest) -> func.HttpResponse:
    """Make single prediction"""
    try:
        load_model()
        
        req_body = req.get_json()
        
        if 'features' not in req_body:
            return func.HttpResponse(
                json.dumps({'error': 'Missing features field'}),
                status_code=400,
                mimetype='application/json'
            )
        
        features = create_features(req_body['features'])
        
        prediction = make_prediction(features)
        
        return func.HttpResponse(
            json.dumps({
                'prediction': prediction,
                'confidence': 0.88,
                'model': model_data['model_name'],
                'timestamp': datetime.now().isoformat()
            }),
            status_code=200,
            mimetype='application/json'
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return func.HttpResponse(
            json.dumps({'error': str(e)}),
            status_code=500,
            mimetype='application/json'
        )

@app.route(route='forecast', methods=['POST'])
def forecast(req: func.HttpRequest) -> func.HttpResponse:
    """Generate forecast for multiple days"""
    try:
        load_model()
        
        req_body = req.get_json()
        
        days_ahead = req_body.get('days_ahead', 30)
        last_sales = req_body.get('last_sales', {})
        
        if days_ahead > 90:
            return func.HttpResponse(
                json.dumps({'error': 'Maximum forecast horizon is 90 days'}),
                status_code=400,
                mimetype='application/json'
            )
        
        # Generate forecast
        forecast_data = []
        current_date = datetime.now()
        
        for i in range(days_ahead):
            future_date = current_date + timedelta(days=i+1)
            
            features = {
                'date': future_date.isoformat(),
                'sales_lag_1': last_sales.get('sales_lag_1', 0),
                'sales_lag_7': last_sales.get('sales_lag_7', 0),
                'sales_lag_30': last_sales.get('sales_lag_30', 0),
                'sales_ma_7': last_sales.get('sales_ma_7', 0),
                'sales_ma_30': last_sales.get('sales_ma_30', 0)
            }
            
            feature_dict = create_features(features)
            
            pred = make_prediction(feature_dict)
            
            forecast_data.append({
                'date': future_date.date().isoformat(),
                'predicted_demand': pred,
                'confidence_interval': {
                    'lower': pred * 0.9,
                    'upper': pred * 1.1
                }
            })
        
        return func.HttpResponse(
            json.dumps({
                'forecast': forecast_data,
                'model': model_data['model_name'],
                'accuracy': model_data['metrics']['r2'],
                'generated_at': datetime.now().isoformat()
            }),
            status_code=200,
            mimetype='application/json'
        )
    
    except Exception as e:
        logger.error(f"Forecast error: {str(e)}")
        return func.HttpResponse(
            json.dumps({'error': str(e)}),
            status_code=500,
            mimetype='application/json'
        )

@app.route(route='batch-predict', methods=['POST'])
def batch_predict(req: func.HttpRequest) -> func.HttpResponse:
    """Make predictions for multiple records"""
    try:
        load_model()
        
        req_body = req.get_json()
        
        if 'records' not in req_body:
            return func.HttpResponse(
                json.dumps({'error': 'Missing records field'}),
                status_code=400,
                mimetype='application/json'
            )
        
        records = req_body['records']
        predictions = []
        
        for record in records:
            try:
                features = create_features(record)
                pred = make_prediction(features)
                
                predictions.append({
                    'input_id': record.get('id', len(predictions)),
                    'prediction': pred,
                    'status': 'success'
                })
            except Exception as e:
                predictions.append({
                    'input_id': record.get('id', len(predictions)),
                    'error': str(e),
                    'status': 'failed'
                })
        
        return func.HttpResponse(
            json.dumps({
                'predictions': predictions,
                'total': len(predictions),
                'successful': sum(1 for p in predictions if p['status'] == 'success')
            }),
            status_code=200,
            mimetype='application/json'
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return func.HttpResponse(
            json.dumps({'error': str(e)}),
            status_code=500,
            mimetype='application/json'
        )
