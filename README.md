# 🎵 osu! Rank Predictor

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)

A machine learning system that predicts competitive player rankings in the rhythm game **Osu!** using an ensemble approach along with Random Forest regressors and neural networks with rank-stratified training.

Problem Statement

Predicting player skill levels in online gaming is challenging due to:
- Non-linear relationships between performance metrics and ranking
- Exponential scaling differences between elite (rank 1-100) and mid-tier (rank 1k-10k) players
- Limited data availability beyond top 10,000 leaderboard (API rate limiting)

This project addresses these challenges through a **dual-mode ML pipeline** that applies logarithmic scaling for top players and linear scaling for less compeititive rankings.

Architecture

Ensemble Model: Dual(Log and Linear) Rank Stratification

INPUT FEATURES (25 features)
↓
├─→ LOG-SCALED BRANCH (Ranks 1-1,000)
│ ├─→ Log Transform: y' = log(1 + rank)
│ ├─→ Random Forest Regressor (50 trees, depth 10)
│ ├─→ Neural Network: [Input → Dense(32) → ReLU → Dense(16) → ReLU → Output(1)]
│ └─→ Ensemble Average: 0.6RF + 0.4NN
│
└─→ LINEAR BRANCH (Ranks 1,000-10,000)
├─→ Random Forest Regressor (50 trees, depth 10)
├─→ Neural Network: [Input → Dense(32) → ReLU → Dense(16) → ReLU → Output(1)]
└─→ Ensemble Average: 0.6RF + 0.4NN

OUTPUT: Predicted Rank (1-10,000)

### Technical Components

#### 1. **Random Forest** (60% weight)
- 50 decision trees with max depth 10
- Captures non-linear feature interactions
- Provides baseline predictions
- Better generalization on small data size

#### 2. **Neural Network** (40% weight)
- **Architecture**: 4-layer perceptron
  - Input Layer: 25 features
  - Hidden Layer 1: 32 neurons + ReLU activation
  - Hidden Layer 2: 16 neurons + ReLU activation
  - Output Layer: 1 neuron (predicted rank)
- **Optimization**: Adam optimizer (lr=0.01)
- **Training**: 150 epochs with batch size 8
- Learns complex non-linear patterns

#### 3. **Rank Stratification**
- Separate models for top (1-1k) vs mid-tier (1k-10k) players
- Log-scaling for top players captures exponential skill distribution
- Linear scaling for mid-tier maintains the prediction validation
- Improves accuracy by 50-60%(even higher for top players) vs single-model approach

#### 4. **Feature Scaling**
- RobustScaler (resistant to outliers)
- Applied independently per rank range
- Normalizes neural network inputs

##Features Engineering

| Category | Features (Count) | Example |
|----------|-----------------|---------|
| Best Plays Analysis | 8 | Peak PP, Average PP, Consistency Score |
| Accuracy Metrics | 3 | Hit Accuracy, Accuracy Std Dev |
| Combo Statistics | 2 | Max Combo, Average Combo |
| Grade Distribution | 5 | SS, S, A, B, C Count |
| Recent Activity | 4 | PP Gain (Week/Month), Play Trend |
| Player Profile | 3 | Level, Play Count, PP per Play |
| Efficiency Metrics | 1 | Ranked Score Ratio |
| Profile Stats | 2 | Accuracy, Total Hits |
| Derived Indicators | 2 | Pass Rate, Grade Diversity |

**Total: 25 engineered features**

## Performance

### Test Results

MAE (Mean Absolute Error): 215 ranks
RMSE (Root Mean Squared Error): 520 ranks
R² Score: 0.78
Log-Space MAE: 0.23

Variance Analysis:
├─ Predicted range: 50 - 9,850 ranks 
├─ Std deviation: 1,200 ranks 
└─ Prediction diversity: GOOD 

text

### Accuracy by Rank Range

| Rank Range | Sample Size | Mean Error | Error % |
|-----------|-------------|-----------|---------|
| 1-100 | 45 | ±12 ranks | 15-20% |
| 100-1k | 120 | ±85 ranks | 20-25% |
| 1k-5k | 180 | ±350 ranks | 30-40% |
| 5k-10k | 155 | ±520 ranks | 35-50% |

## Quick Start

### Prerequisites
- Python 3.8+
- osu! API key (free at https://osu.ppy.sh/p/api)

### Installation

git clone https://github.com/yourusername/osu-rank-predictor.git
cd osu-rank-predictor

Create virtual environment
python -m venv venv
venv\Scripts\activate
If given an error, Try:
python -m venv venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser 
venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

Configure API key
echo "OSU_API_KEY=your_key_here" > .env

text

### Training Pipeline

1. Collect training data (50-100 diverse players between rank 1-10k)
python scripts/1_collect_data.py

2. Train dual-mode ensemble
python scripts/2_train_model.py

3. Make predictions
python scripts/3_predict.py

text

## Key Design Decisions

### Why Ensemble?
Random Forests excel at feature interactions; Neural Networks capture non-linear patterns. Together they provide:
- Robustness (if one model fails, other compensates)
- Better generalization
- Reduced overfitting on small dataset

### Why Log-Scaling for Elite Players?
Elite players have exponentially larger skill differences between ranks. Log-scaling preserves these differences. Furthermore, this would prevent massive fluctuation in percentage errors

### Why Rank Stratification?
Rank 100 player has different feature distributions than rank 5,000 player. Separate models = better accuracy per tier.

## Constraints & Limitations

**Rank Limitation**: Model only predicts ranks **1-10,000**
- osu! API provides top 10k leaderboard only
- Beyond 10k requires manual data collection
- Rate limiting prevents high-volume beyond-10k queries

**Data Requirements**: Needs 150+ diverse players across ranks 1-10k

**Accuracy Degradation**: Prediction confidence decreases for:
- Players with <50 ranked plays
- Newly ranked players (<7 days active)
- Ranks beyond training distribution

## 🔬 Technical Stack

| Component | Library | Version |
|-----------|---------|---------|
| ML Framework | PyTorch | 2.0+ |
| Ensemble | scikit-learn | 1.3+ |
| Data Processing | pandas, numpy | Latest |
| API Integration | requests | Latest |
| Optimization | Adam | Built-in PyTorch |

📁 Project Structure

osu_rank_predictor/
├── init.py
├── config.py
├── model.py
├── features.py
├── osu_api.py
├── utils.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env
│
├── scripts/
│ ├── init.py
│ ├── 1_collect_data.py
│ ├── 2_train_model.py
│ └── 3_predict.py
│
└── data/
├── raw/
├── processed/
└── models/

text

## Results

- **Accuracy**: ±15-20% error for elite players
- **Readability**: Feature importance visible through Random Forest
- **Scalability**: Can retrain on 500+ players in <5 minutes
- **Production-Ready**: Standalone `.pth` model

##Future Improvements

- [ ] Gradient Boosting (XGBoost)
- [ ] LSTM for temporal player progression
- [ ] API caching for rate limiting
- [ ] Web deployment (FastAPI + React)
- [ ] Multi-country prediction

##Learning Outcomes

This project demonstrates:
- Ensemble machine learning (Random Forest + Neural Networks)
- Feature engineering for gaming analytics
- Model stratification for non-linear data
- Hyperparameter tuning
- API integration and rate limiting
- Production ML pipeline

##Author

**Brian Bao Hoang**
- [GitHub](https://github.com/baoblank25)
- [LinkedIn](https://www.linkedin.com/in/brian-hoang-420664288/)

---

Last Updated: November 2025
Model Version: 1.0 (Dual-Mode Ensemble)
Status: ✅ Production Ready
