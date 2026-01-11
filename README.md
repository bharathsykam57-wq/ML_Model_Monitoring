# Machine Learning Model Monitoring & Trust System

*“I built this project to understand how real ML systems fail after deployment.”*

## Overview

Machine learning models are often evaluated only during training and validation.  
However, **most real-world failures happen after deployment**, when models encounter changing data, evolving user behavior, and hidden bias.

This project demonstrates how to **monitor, diagnose, and respond** to post-deployment issues in machine learning systems, including:

- Data drift in production inputs  
- Performance degradation over time  
- Bias and fairness risks across sensitive groups  

The system simulates a realistic production environment and implements **industry-grade monitoring workflows** used by data science and ML engineering teams.

## Problem Statement

Once deployed, machine learning models are exposed to real-world data that may differ significantly from training data. These changes often happen silently and can lead to:

- Degraded model performance  
- Incorrect or unfair predictions  
- Business and regulatory risks  

This project builds a **model monitoring and trust layer** that continuously evaluates model behavior after deployment and supports informed human decision-making.

## Business Impact

Unchecked model degradation can cause:

- Financial loss due to incorrect predictions  
- Poor customer experience  
- Bias and fairness violations  
- Loss of trust in ML-driven decisions  

Early detection of drift, performance drops, and bias enables teams to **intervene before serious damage occurs**.

## Target Audience

**Primary users**
- Data Scientists maintaining deployed models  
- ML Engineers responsible for production reliability  

**Secondary users**
- Product Managers relying on ML outputs  
- Risk and compliance stakeholders monitoring fairness  

## Use Cases

- Detect input data drift before accuracy drops  
- Monitor precision, recall, and ROC-AUC over time  
- Identify bias across sensitive attributes  
- Trigger alerts and retraining recommendations  

## System Architecture

1. **Reference Dataset**
   - Represents expected data distribution at deployment time  

2. **Production Batches**
   - Simulated incoming data over time  

3. **Monitoring Layer**
   - Performance monitoring  
   - Data drift detection & severity classification  
   - Bias & fairness monitoring  

4. **Alert Engine**
   - Raises alerts when thresholds are violated  

5. **Decision Engine**
   - Recommends retraining or escalation based on signals  

6. **Visualization Layer**
   - Professional dashboards for trends and diagnostics  

## Key Features

### Performance Monitoring
- Tracks precision, recall, ROC-AUC per batch  
- Stores both snapshot reports and time-series metrics  

### Data Drift Detection
- Feature-level drift detection  
- Drift severity classification: **LOW / MEDIUM / HIGH**  
- Explicit separation of schema issues vs true drift  

### Bias & Fairness Monitoring
- Group-wise recall tracking across sensitive attributes  
- Minimum group size enforcement  
- Recall gap–based bias detection  

### Alert Engine
- Unified alerts across performance, drift, and bias  
- Human-readable explanations  

### Retraining Recommendation
- Converts monitoring signals into concrete actions:
  - NO_ACTION  
  - RETRAIN  
  - ESCALATE_FAIRNESS  

## Dataset review

- **IBM Telco Customer Churn Dataset**
- Includes demographics, services, pricing, and churn labels  
- Suitable for:
  - Drift simulation  
  - Bias analysis  
  - Monitoring demonstrations  

**Note:** Production data is simulated to reflect realistic batch monitoring scenarios.

## Technology Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- SHAP (model explainability)  
- Matplotlib / Seaborn  
- Joblib  

## Project Structure
- ├── data_pipeline/        # Cleaning and data splitting
- ├── models/               # Training, metadata, explainability
- ├── monitoring/           # Drift, performance, bias, alerts
- ├── dashboard/            # Visualization scripts & outputs
- ├── docs/                 # Problem definition & dataset selection
- ├── data/                 # Raw, clean, reference, production data


## Limitations

- Production data is simulated (not live streaming)
- No automated retraining pipeline
- No real-time inference monitoring
- No deployment infrastructure (Docker/Kubernetes)

These exclusions are intentional to keep the project **focused on monitoring concepts**.


## What This Project Demonstrates

- Understanding of **post-deployment ML failure modes**
- Ability to design **monitoring systems**, not just models
- Practical handling of **data drift, bias, and governance**
- Clear separation between **signals and decisions**
- Industry-aligned ML lifecycle thinking


## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Run data pipeline:
   ```bash
   python data_pipeline/data_cleaning.py
   python data_pipeline/split_reference_production.py
3. Train baseline model:
   ```bash
   python models/train_baseline_model.py
4. Run monitoring:
   ```bash
   python monitoring/performance_monitoring.py
   python monitoring/data_drift.py
   python monitoring/bias_monitoring.py
5. Trigger alerts:
   ```bash
   python monitoring/alert_engine.py
