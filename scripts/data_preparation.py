#!/usr/bin/env python3
"""
Data Preparation Script for Demand Forecasting
Cleans, validates, and prepares data for ML model training
"""
#you can modify depend your need 
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class DataPreparation:
    def __init__(self, input_file):
        self.df = pd.read_csv(input_file)
        self.validation_results = {}
    
    def load_and_validate(self):
        """Load data and perform validation checks"""
        print("📊 Loading data...")
        print(f"   Shape: {self.df.shape}")
        print(f"   Columns: {self.df.columns.tolist()}")
        
        required_cols = ['date', 'sales', 'product_id']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print("✅ Validation passed")
        return self
    
    def handle_missing_values(self):
        """Handle missing data"""
        print("\n🔧 Handling missing values...")
        
        initial_missing = self.df.isnull().sum().sum()
        print(f"   Initial missing values: {initial_missing}")
        
        self.df['sales'] = self.df.groupby('product_id')['sales'].fillna(method='ffill')
        self.df['sales'] = self.df.groupby('product_id')['sales'].fillna(method='bfill')
        
        self.df = self.df.dropna(subset=['sales'])
        
        final_missing = self.df.isnull().sum().sum()
        print(f"   Final missing values: {final_missing}")
        print("✅ Missing values handled")
        return self
    
    def remove_outliers(self, std_threshold=3):
        """Remove statistical outliers"""
        print("\n🔍 Removing outliers...")
        
        initial_rows = len(self.df)
        
        # Calculate z-score per product
        self.df['z_score'] = self.df.groupby('product_id')['sales'].transform(
            lambda x: np.abs((x - x.mean()) / x.std())
        )
        
        # Remove rows with z-score > threshold
        self.df = self.df[self.df['z_score'] <= std_threshold]
        self.df = self.df.drop('z_score', axis=1)
        
        removed = initial_rows - len(self.df)
        print(f"   Removed {removed} outlier rows ({removed/initial_rows*100:.2f}%)")
        print("✅ Outliers removed")
        return self
    
    def create_features(self):
        """Create time series features"""
        print("\n✨ Creating features...")
        
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        self.df['month'] = self.df['date'].dt.month
        self.df['quarter'] = self.df['date'].dt.quarter
        self.df['dayofweek'] = self.df['date'].dt.dayofweek
        self.df['week'] = self.df['date'].dt.isocalendar().week
        
        self.df['sales_lag_1'] = self.df.groupby('product_id')['sales'].shift(1)
        self.df['sales_lag_7'] = self.df.groupby('product_id')['sales'].shift(7)
        self.df['sales_lag_30'] = self.df.groupby('product_id')['sales'].shift(30)
        
        self.df['sales_ma_7'] = self.df.groupby('product_id')['sales'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        self.df['sales_ma_30'] = self.df.groupby('product_id')['sales'].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean()
        )
        
        self.df = self.df.fillna(method='bfill').fillna(method='ffill')
        
        print(f"   Created features: {self.df.columns.tolist()}")
        print("✅ Features created")
        return self
    
    def normalize_data(self):
        """Normalize numeric features"""
        print("\n📐 Normalizing data...")
        
        from sklearn.preprocessing import MinMaxScaler
        
        numeric_cols = ['sales', 'sales_lag_1', 'sales_lag_7', 'sales_lag_30', 
                       'sales_ma_7', 'sales_ma_30']
        
        scaler = MinMaxScaler()
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        
        print(f"   Normalized {len(numeric_cols)} features")
        print("✅ Data normalized")
        return self
    
    def split_data(self, test_size=0.2):
        """Split data into train/test"""
        print("\n📋 Splitting data...")
        
        self.df = self.df.sort_values('date')
        
        split_idx = int(len(self.df) * (1 - test_size))
        
        train_data = self.df[:split_idx]
        test_data = self.df[split_idx:]
        
        print(f"   Train: {len(train_data)} rows ({len(train_data)/len(self.df)*100:.1f}%)")
        print(f"   Test: {len(test_data)} rows ({len(test_data)/len(self.df)*100:.1f}%)")
        print("✅ Data split completed")
        
        return train_data, test_data
    
    def get_statistics(self):
        """Print data statistics"""
        print("\n📈 Data Statistics:")
        print(f"   Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"   Products: {self.df['product_id'].nunique()}")
        print(f"   Avg sales: {self.df['sales'].mean():.2f}")
        print(f"   Sales std: {self.df['sales'].std():.2f}")
        print(f"   Min sales: {self.df['sales'].min():.2f}")
        print(f"   Max sales: {self.df['sales'].max():.2f}")
    
    def save_processed_data(self, output_file):
        """Save processed data to CSV"""
        print(f"\n💾 Saving data to {output_file}...")
        self.df.to_csv(output_file, index=False)
        print(f"✅ Saved {len(self.df)} rows")
        return self

def main():
    """Main execution"""
    print("=" * 60)
    print("   DEMAND FORECASTING - DATA PREPARATION")
    print("=" * 60)
    
    try:
        prep = DataPreparation('data/sample_sales.csv')
        
        (prep.load_and_validate()
            .handle_missing_values()
            .remove_outliers()
            .create_features()
            .normalize_data()
            .get_statistics())
        
        train_data, test_data = prep.split_data(test_size=0.2)
        
        prep.save_processed_data('data/train_data.csv')
        test_data.to_csv('data/test_data.csv', index=False)
        
        print("\n✅ Data preparation completed successfully!")
        print("\nNext step: Run model training")
        print("   Command: python scripts/model_training.py")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
