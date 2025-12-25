"""
Train Employee Attrition Prediction Models
Compares Logistic Regression, Random Forest, and XGBoost
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class AttritionPredictor:
    def __init__(self, data_path='data/employee_data.csv'):
        self.df = pd.read_csv(data_path)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
    def preprocess_data(self):
        """Prepare data for modeling"""
        print("Preprocessing data...")
        
        # Separate features and target
        X = self.df.drop(['EmployeeID', 'Attrition'], axis=1)
        y = self.df['Attrition']
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        
        # Scale features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        # Save scaler and encoders
        joblib.dump(scaler, 'models/scaler.pkl')
        joblib.dump(label_encoders, 'models/label_encoders.pkl')
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Class distribution after SMOTE: {np.bincount(self.y_train)}")
        
    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        print("\nTraining Logistic Regression...")
        
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(self.X_train, self.y_train)
        
        self.models['Logistic Regression'] = lr
        self._evaluate_model(lr, 'Logistic Regression')
        
        joblib.dump(lr, 'models/logistic_regression.pkl')
        
    def train_random_forest(self):
        """Train Random Forest with hyperparameter tuning"""
        print("\nTraining Random Forest...")
        
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train)
        
        self.models['Random Forest'] = rf
        self._evaluate_model(rf, 'Random Forest')
        
        joblib.dump(rf, 'models/random_forest.pkl')
        
    def train_xgboost(self):
        """Train XGBoost classifier"""
        print("\nTraining XGBoost...")
        
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        xgb.fit(self.X_train, self.y_train)
        
        self.models['XGBoost'] = xgb
        self._evaluate_model(xgb, 'XGBoost')
        
        joblib.dump(xgb, 'models/xgboost.pkl')
        
    def _evaluate_model(self, model, name):
        """Evaluate model performance"""
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        metrics = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred),
            'Recall': recall_score(self.y_test, y_pred),
            'F1-Score': f1_score(self.y_test, y_pred),
            'ROC-AUC': roc_auc_score(self.y_test, y_pred_proba)
        }
        
        self.results[name] = metrics
        
        print(f"\n{name} Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
    def compare_models(self):
        """Compare all models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(4)
        print(results_df)
        
        # Find best model
        best_model = results_df['ROC-AUC'].idxmax()
        print(f"\nBest Model: {best_model}")
        print(f"ROC-AUC: {results_df.loc[best_model, 'ROC-AUC']:.4f}")
        
        # Save results
        results_df.to_csv('models/model_comparison.csv')
        
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        rf = self.models.get('Random Forest')
        if rf is None:
            return
        
        feature_importance = pd.DataFrame({
            'Feature': self.df.drop(['EmployeeID', 'Attrition'], axis=1).columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False).head(15)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance, y='Feature', x='Importance')
        plt.title('Top 15 Feature Importance (Random Forest)', fontsize=14)
        plt.tight_layout()
        plt.savefig('visualizations/feature_importance.png', dpi=300)
        print("\nFeature importance plot saved to visualizations/feature_importance.png")
        
def main():
    # Initialize predictor
    predictor = AttritionPredictor()
    
    # Preprocess
    predictor.preprocess_data()
    
    # Train models
    predictor.train_logistic_regression()
    predictor.train_random_forest()
    predictor.train_xgboost()
    
    # Compare
    predictor.compare_models()
    
    # Feature importance
    predictor.plot_feature_importance()
    
    print("\n✅ All models trained and saved!")
    print("Models saved in: models/")
    
if __name__ == "__main__":
    main()