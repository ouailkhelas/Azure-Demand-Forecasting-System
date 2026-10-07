# Demand Forecasting - Deployment Guide
## 📋 Prerequisites

### Required Software --you can use docker containers 
- Python 3.8+ 
- Azure CLI 2.40+
- Git
- PowerShell or Bash terminal

### Azure Services
- Azure subscription
- Azure Storage Account
- Azure SQL Database
- Azure Functions
- Azure Machine Learning

---
## 🚀 PHASE 1: Local Setup 
### Step 1: Create Project Folder
### Step 2: Create Python Virtual Environment
### Step 3: Install Dependencies
### Step 4: Create Folder Structure
### Step 5: Download Sample Data

## 🔧 PHASE 2: Data Preparation

## 🤖 PHASE 3: Model Training
**What it does:**
- Trains 3 different models:
  - Linear Regression
  - Random Forest
  - Gradient Boosting
- Evaluates each model
- Selects best performer
- Saves trained model

## 🧪 PHASE 4: Local Testing 

## ☁️ PHASE 5: Azure Setup
### Step 1: Login to Azure
### Step 2: Set Subscription
### Step 3: Create Resource Group
### Step 4: Create Storage Account
### Step 5: Create SQL Database
```
# Create SQL Server
# Create SQL Database
```
### Step 6: Upload Model to Blob Storage

## 🔌 PHASE 6: Deploy Azure Functions
### Step 1: Create Function App
### Step 2: Copy Function Code
### Step 3: Deploy Function
### Step 4: Set Environment Variables

## 🌐 PHASE 7: Test API Endpoints
### Get Function URL
### Test Health Endpoint
### Test Prediction Endpoint
### Test Forecast Endpoint

## 📊 PHASE 8: Connect Power BI Dashboard
### Step 1: Create Power BI Report
1. Open Power BI Desktop
2. Get Data → Web → Paste API URL
3. Add API Key to headers

### Step 2: Create Visualizations
- Line chart: Predicted demand over time
- Gauge: Model accuracy (R² score)
- Table: Confidence intervals
- KPI: Forecast vs actual .....

### Step 3: Publish Dashboard
