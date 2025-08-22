# FPL-ML: Fantasy Premier League Machine Learning Pipeline

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)  
![License](https://img.shields.io/badge/license-MIT-green.svg)  
![Last Commit](https://img.shields.io/github/last-commit/kyupralis24/fpl-ml)  
![Repo Size](https://img.shields.io/github/repo-size/kyupralis24/fpl-ml)  
![Issues](https://img.shields.io/github/issues/kyupralis24/fpl-ml)  

A comprehensive machine learning system for Fantasy Premier League (FPL) that automates data collection, feature engineering, predictive modeling, and squad optimization to maximize weekly points.  

---

## Overview

This repository implements an end-to-end pipeline that processes FPL API data through machine learning models to generate optimal squad selections. The system operates on a weekly cadence, training models on historical data and predicting player performance for upcoming gameweeks.

---

## System Architecture

The pipeline consists of three main components:

```mermaid
flowchart TD
    subgraph "Data Pipeline"
        FETCH["fetch_gw.py<br/>API Data Ingestion"]
        FEATURES["update_features_weekly.py<br/>Feature Engineering"]
    end
    
    subgraph "ML Pipeline"
        TRAIN["train_model_weekly.py<br/>Model Training"]
        PREDICT["predict_next_gw.py<br/>Point Predictions"]
    end
    
    subgraph "Optimization"
        SQUAD["select_squad.py<br/>Squad Selection"]
    end
    
    FETCH --> FEATURES
    FEATURES --> TRAIN
    TRAIN --> PREDICT
    PREDICT --> SQUAD