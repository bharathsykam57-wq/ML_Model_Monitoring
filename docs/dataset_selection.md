# Data Selection

## Problem Alignment
The goal of this project is to monitor machine learning models after deployment for data drift, performance degradation, and bias. Customer churn prediction is a common real-world use case where models are deployed in production and continuously exposed to changing customer behaviour.

Changes in customer usage patterns, pricing or contract-structures can cause input feature distributions to drift over time. This makes churn prediction a suitable problem for demonstrating post-deployment monitoring and trust-related challenges in machine learning systems.

## Dataset Description
The IBM Telco Customer Churn dataset contains customer-level information from a telecommunications company. It includes demographic details, service usage details, contract information and a binary target variable indicating whether a customer has churned or not.

The dataset consists of multiple numerical and categorical columns such as tenure, monthly charges, contract type, payment method and customer demographics. These features reflect realistic inputs commonly used in customer churn prediction models.

## Why this Dataset is Suitable
- It represents a real business problem where model predictions directly influence operational and financial decisions. 
- The presence of demographic attributes allows for fairness and bias analysis across different customer groups.
- The feature distributions can be partitioned over time to simulate reference data and production data batches enabling real data drift and   performance monitoring. This matches well with the objectives of post-deployment model monitoring.

## Limitations
The dataset represents historical customer data rather than a true time-series collected in production, As a result, production data drift must be simulated rather than observed naturally.

Despite this limitation, the dataset is sufficient for demonstrating core monitoring concepts such as drift detection, performance monitoring and bias analysis.