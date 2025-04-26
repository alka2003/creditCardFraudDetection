# Credit Card Fraud Detection System
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_auc_score, precision_recall_curve, auc,
                            average_precision_score, PrecisionRecallDisplay)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import time
from datetime import datetime
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# 1. Data Loading and Initial Exploration
def load_data():
    train = pd.read_csv('fraudTrain.csv')
    test = pd.read_csv('fraudTest.csv')
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud percentage: {df['is_fraud'].mean()*100:.4f}%")
    print("\nMissing values:\n", df.isnull().sum())
    
    return df

# 2. Feature Engineering
def feature_engineering(df):
    # Convert datetime features
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    
    # Time-based features
    df['trans_hour'] = df['trans_date_trans_time'].dt.hour
    df['trans_day'] = df['trans_date_trans_time'].dt.day
    df['trans_dayofweek'] = df['trans_date_trans_time'].dt.dayofweek
    df['trans_month'] = df['trans_date_trans_time'].dt.month
    
    # Age at time of transaction
    df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days / 365
    
    # Distance between customer and merchant
    df['distance'] = df.apply(lambda x: geodesic((x['lat'], x['long']), 
                              (x['merch_lat'], x['merch_long'])).km, axis=1)
    
    # Transaction frequency and amount statistics
    df['trans_freq'] = df.groupby('cc_num')['cc_num'].transform('count')
    df['avg_amt'] = df.groupby('cc_num')['amt'].transform('mean')
    df['amt_to_avg_ratio'] = df['amt'] / (df['avg_amt'] + 0.01)
    
    # Time since last transaction
    df = df.sort_values(['cc_num', 'trans_date_trans_time'])
    df['time_since_last'] = df.groupby('cc_num')['unix_time'].diff()
    
    # Drop unnecessary columns
    cols_to_drop = ['trans_date_trans_time', 'cc_num', 'merchant', 'first', 'last', 
                    'street', 'city', 'zip', 'lat', 'long', 'job', 'dob', 
                    'trans_num', 'unix_time', 'merch_lat', 'merch_long']
    df = df.drop(cols_to_drop, axis=1)
    
    return df

# 3. Data Visualization
def visualize_data(df):
    # Fraud distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='is_fraud', data=df)
    plt.title('Distribution of Fraudulent Transactions')
    plt.show()
    
    # Transaction amount distribution
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='is_fraud', y='amt', data=df[df['amt'] < 1000])
    plt.title('Transaction Amount by Fraud Status')
    plt.show()
    
    # Hourly transaction patterns
    plt.figure(figsize=(12, 6))
    sns.countplot(x='trans_hour', hue='is_fraud', data=df)
    plt.title('Hourly Transaction Patterns by Fraud Status')
    plt.show()
    
    # Category analysis
    plt.figure(figsize=(14, 6))
    cat_fraud = df.groupby('category')['is_fraud'].mean().sort_values(ascending=False)
    cat_fraud.plot(kind='bar')
    plt.title('Fraud Percentage by Transaction Category')
    plt.ylabel('Fraud Percentage')
    plt.show()

# 4. Data Preprocessing
def preprocess_data(df):
    # Separate features and target
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    # Identify categorical columns
    categorical_cols = ['gender', 'category', 'state']
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    
    # Create transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)])
    
    return X, y, preprocessor

# 5. Model Building and Evaluation
def build_models(X, y, preprocessor):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
        'XGBoost': XGBClassifier(scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
                                eval_metric='aucpr', random_state=42),
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    }
    
    # Evaluate models
    results = {}
    for name, model in models.items():
        pipeline = imbpipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)])
        
        print(f"\nTraining {name}...")
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        # Metrics
        precision = average_precision_score(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        print(f"\n{name} Results:")
        print(f"Training time: {train_time:.2f} seconds")
        print(f"Average Precision: {precision:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Store results
        results[name] = {
            'model': pipeline,
            'precision': precision,
            'roc_auc': roc_auc,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Plot Precision-Recall curve
        display = PrecisionRecallDisplay.from_estimator(
            pipeline, X_test, y_test, name=name)
        _ = display.ax_.set_title(f'Precision-Recall Curve - {name}')
    
    return results

# 6. Feature Importance Analysis
def analyze_feature_importance(results, X):
    best_model = results['Random Forest']['model']
    
    # Get feature names after preprocessing
    preprocessor = best_model.named_steps['preprocessor']
    feature_names = (list(preprocessor.transformers_[0][1].get_feature_names_out()) +
                    list(preprocessor.transformers_[1][1].get_feature_names_out()))
    
    # Get feature importances
    importances = best_model.named_steps['classifier'].feature_importances_
    
    # Create DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', 
                data=feature_importance.head(20))
    plt.title('Top 20 Important Features')
    plt.tight_layout()
    plt.show()
    
    return feature_importance

# Main execution
if __name__ == "__main__":
    # Load and prepare data
    df = load_data()
    df = feature_engineering(df)
    visualize_data(df)
    
    # Preprocess and model
    X, y, preprocessor = preprocess_data(df)
    results = build_models(X, y, preprocessor)
    
    # Analyze results
    feature_importance = analyze_feature_importance(results, X)
    
    # Save best model
    best_model = results['Random Forest']['model']
    import joblib
    joblib.dump(best_model, 'fraud_detection_model.pkl')
    print("\nBest model saved as 'fraud_detection_model.pkl'")