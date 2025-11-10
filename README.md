# Ensemble Learning for Bike Sharing Demand
---

**Author:** Priyavrata Tiwari  
**Roll Number:** AE22B007 

---

##  Overview
This notebook implements and compares several ensemble methods to predict hourly bike sharing demand (`cnt`) using the UCI Bike Sharing Dataset (`hour.csv`).  
The notebook evaluates the following models:

- Decision Tree Regressor (baseline)
- Linear Regression (baseline)
- Bagging Regressor (bagging)
- Gradient Boosting Regressor (boosting)
- Stacking Regressor (stacked ensemble of base learners)

Key design choices:
- Temporal (time-ordered) train/test split (no random shuffle) to respect the time-series nature.
- Drop columns not used for prediction (`instant`, `dteday`, `casual`, `registered`) as `cnt` is the target.
- One-hot encode categorical time features (`season`, `weathersit`, `mnth`, `hr`, `weekday`).
- Standard scaling of numerical features within a pipeline.
- Metrics reported: Root Mean Square Error (RMSE) and R².

## Notebook structure (high-level)
1. **Imports & dependencies**  
   Import libraries (numpy, pandas, matplotlib, seaborn, scikit-learn, etc.)

2. **Data loading & quick EDA**
   - Load `hour.csv`
   - Inspect basic statistics and missing values
   - Visualizations (counts over time, correlation heatmap)

3. **Preprocessing**
   - Drop `instant`, `dteday`, `casual`, `registered`
   - One-hot encoding of categorical features (`season`, `weathersit`, `mnth`, `hr`, `weekday`)
   - Define numeric feature list and column transformer (scaling + passthrough for dummies)
   - Keep chronological order for splitting

4. **Train / Test split**
   - Use first N% rows for training and remainder for testing (temporal split)

5. **Model definitions & pipelines**
   - Decision Tree
   - Linear Regression (with pipeline/scaling)
   - BaggingRegressor wrapping a DecisionTree
   - GradientBoostingRegressor
   - StackingRegressor combining a set of base learners with a final estimator

6. **Training & Predictions**
   - Fit models on the training set
   - Predict on test set

7. **Evaluation & Visualization**
   - Compute RMSE and R² for each model
   - Plot actual vs predicted for the best model
   - Summarize results in a table and discuss findings

8. **Final analysis & conclusions**
   - Comparative summary of metrics
   - Why ensemble methods (bagging/boosting/stacking) perform better
   - Recommendations and limitations

## Dataset
This notebook expects the UCI Bike Sharing Dataset `hour.csv`. If not present, download from the UCI Bike Sharing repository and place `hour.csv` in the notebook directory.

Example dataset structure (columns):
- instant, dteday, season, yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed, casual, registered, cnt

**Note:** The notebook drops `instant`, `dteday`, `casual`, `registered`.

## Reproducibility & Environment
Recommended Python version: **3.9+**

Create a `requirements.txt` with the following minimal set (the notebook may import more for plotting; add if used):

```text
numpy>=1.21
pandas>=1.3
scikit-learn>=1.0
matplotlib>=3.4
seaborn>=0.11
jupyterlab
nbformat
