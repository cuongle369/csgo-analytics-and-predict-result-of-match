import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE  # Added for imbalance handling
from imblearn.pipeline import Pipeline as ImbPipeline  # Pipeline compatible with SMOTE
from joblib import dump, load

import warnings

warnings.filterwarnings('ignore')


def load_and_prepare_data(file_path):
    """
    Load and prepare data for modeling
    """
    # Load cleaned data
    df = pd.read_csv(file_path)

    # Separate features and target
    X = df.drop('result', axis=1)
    y = df['result']

    # Print class distribution to identify imbalance
    print("Target variable distribution (%):")
    distribution = y.value_counts(normalize=True) * 100
    print(distribution)
    if distribution.min() < 15:
        print("Warning: Detected potential class imbalance (minority class < 15%). SMOTE will be applied.")

    return X, y


def build_classification_models(X, y):
    """
    Build and evaluate multiple classification models with imbalance handling using SMOTE
    """
    # Split data into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, multi_class='multinomial'),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Neural Network': MLPClassifier(max_iter=1000, random_state=42)
    }

    # Dictionary to store model performance
    model_results = {}

    # Cross-validation setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n{'-' * 50}")
        print(f"Training {name}...")

        # Create a pipeline with SMOTE for imbalance handling
        pipeline = ImbPipeline([
            ('imputer', SimpleImputer(strategy='median')),  # Impute missing values
            ('smote', SMOTE(random_state=42)),  # Handle class imbalance by oversampling minority classes
            ('scaler', StandardScaler()),  # Scale features
            ('model', model)  # Classifier
        ])

        # Cross-validation with F1-weighted scoring to account for imbalance
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='f1_weighted')
        print(f"Cross-validation F1-weighted score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Train the model
        pipeline.fit(X_train, y_train)

        # Predictions
        y_pred = pipeline.predict(X_test)

        # Performance metrics
        accuracy = accuracy_score(y_test, y_pred)

        # For multi-class (Win, Lose, Tie)
        if len(np.unique(y)) > 2:
            y_pred_proba = pipeline.predict_proba(X_test)
            f1 = f1_score(y_test, y_pred, average='weighted')

            print(f"Test accuracy: {accuracy:.4f}")
            print(f"Weighted F1 score: {f1:.4f}")

            # Store results
            model_results[name] = {
                'cv_f1_weighted': cv_scores.mean(),
                'test_accuracy': accuracy,
                'f1_score': f1,
                'model': pipeline,
                'predictions': y_pred
            }
        else:
            # Binary classification (unlikely based on your context, but kept for completeness)
            roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
            f1 = f1_score(y_test, y_pred)

            print(f"Test accuracy: {accuracy:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}")
            print(f"F1 score: {f1:.4f}")

            model_results[name] = {
                'cv_f1_weighted': cv_scores.mean(),
                'test_accuracy': accuracy,
                'roc_auc': roc_auc,
                'f1_score': f1,
                'model': pipeline,
                'predictions': y_pred
            }

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion matrix as heatmap
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png')
        plt.show()
        plt.close()

    return model_results


def hyperparameter_tuning(X_train, y_train, best_model_name, best_model_pipeline):
    """
    Perform hyperparameter tuning on the best model
    """
    print(f"\n{'-' * 50}")
    print(f"Hyperparameter tuning for {best_model_name}...")

    # Define parameter grid based on the best model
    if best_model_name == 'Logistic Regression':
        param_grid = {
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'model__solver': ['lbfgs', 'saga'],
            'model__penalty': ['l2', 'none']
        }
    elif best_model_name == 'Random Forest':
        param_grid = {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }
    elif best_model_name == 'Gradient Boosting':
        param_grid = {
            'model__n_estimators': [100, 200, 300],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7],
            'model__subsample': [0.8, 1.0]
        }
    elif best_model_name == 'XGBoost':
        param_grid = {
            'model__n_estimators': [100, 200, 300],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7],
            'model__subsample': [0.8, 1.0],
            'model__colsample_bytree': [0.8, 1.0]
        }
    elif best_model_name == 'SVM':
        param_grid = {
            'model__C': [0.1, 1, 10, 100],
            'model__gamma': ['scale', 'auto', 0.1, 0.01],
            'model__kernel': ['rbf', 'poly', 'sigmoid']
        }
    else:  # Neural Network
        param_grid = {
            'model__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'model__activation': ['relu', 'tanh'],
            'model__alpha': [0.0001, 0.001, 0.01],
            'model__learning_rate': ['constant', 'adaptive']
        }

    # Grid search with cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        best_model_pipeline,
        param_grid,
        cv=skf,
        scoring='f1_weighted',  # Changed to f1_weighted for imbalance
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation F1-weighted score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def feature_importance_analysis(X, best_model):
    """
    Analyze feature importance from the best model
    """
    model = best_model.named_steps['model']
    feature_names = X.columns

    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
    else:
        print("Feature importance not available for this model type")
        return

    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()

    return feature_importance


def evaluate_best_model(X_test, y_test, best_model, model_name):
    """
    Final evaluation of the best model
    """
    print(f"\n{'-' * 50}")
    print(f"Final evaluation of {model_name}...")

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f'Confusion Matrix - Tuned {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_tuned_{model_name.replace(" ", "_").lower()}.png')
    plt.show()
    plt.close()

    return accuracy, y_pred


def run_advanced_analytics(file_path):
    """
    End-to-end advanced analytics pipeline with imbalance handling
    """
    print("Loading and preparing data...")
    X, y = load_and_prepare_data(file_path)

    # Split data for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nBuilding initial classification models with imbalance handling...")
    model_results = build_classification_models(X, y)

    # Find the best performing model
    best_model_name = max(model_results, key=lambda k: model_results[k]['test_accuracy'])
    best_model = model_results[best_model_name]['model']

    print(f"\nBest performing model: {best_model_name}")
    print(f"Accuracy: {model_results[best_model_name]['test_accuracy']:.4f}")

    # Hyperparameter tuning
    tuned_model = hyperparameter_tuning(X_train, y_train, best_model_name, best_model)

    # Feature importance analysis
    if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'Logistic Regression']:
        feature_importance = feature_importance_analysis(X, tuned_model)
        print("\nTop 5 most important features:")
        print(feature_importance.head(5))

    # Final evaluation
    final_accuracy, final_predictions = evaluate_best_model(X_test, y_test, tuned_model, best_model_name)

    print(f"\nFinal model accuracy: {final_accuracy:.4f}")

    # Save the trained model
    model_filename = f"best_model_{best_model_name.replace(' ', '_').lower()}.joblib"
    dump(tuned_model, model_filename)
    print(f"\nBest model saved as: {model_filename}")

    print("\nAdvanced analytics completed!")

    print("\nAdvanced analytics completed!")

    return tuned_model


# Example usage
if __name__ == "__main__":
    best_model = run_advanced_analytics('cleaned_match_data.csv')
