# ==============================================================================
# MODEL TRAINER MODULE FOR NETWORK SECURITY PROJECT
# ==============================================================================
# This module handles the training and evaluation of multiple machine learning 
# models, selects the best performing model, and tracks experiments using MLflow.
# It compares different algorithms and their hyperparameters to find the optimal solution.

# IMPORTING REQUIRED LIBRARIES
# ==============================================================================
import os                     # For operating system interface and file operations
import sys

import mlflow.sklearn                    # For system-specific parameters and functions

# PROJECT-SPECIFIC IMPORTS
from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object, load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data, evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

# MACHINE LEARNING ALGORITHMS FROM SCIKIT-LEARN
# ==============================================================================
from sklearn.linear_model import LogisticRegression        # Linear classification algorithm
from sklearn.metrics import r2_score                       # Regression metric (not used here)
from sklearn.neighbors import KNeighborsClassifier         # K-Nearest Neighbors classifier
from sklearn.tree import DecisionTreeClassifier            # Decision tree algorithm
from sklearn.ensemble import (                             # Ensemble methods (multiple models combined)
    AdaBoostClassifier,         # Adaptive Boosting
    GradientBoostingClassifier, # Gradient Boosting
    RandomForestClassifier,     # Random Forest (multiple decision trees)
)

import mlflow

import dagshub
#dagshub.init(repo_owner='tejasitankar10', repo_name='NetworkSecurity_ML', mlflow=True)


os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/tejasitankar10/NetworkSecurity_ML.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "tejasitankar10"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "5f751cd9eed1faf96580866b210e09fe8ae9d9b1"


# ==============================================================================
# MAIN MODEL TRAINER CLASS
# ==============================================================================
class ModelTrainer:
    """
    This class handles the complete model training pipeline including:
    1. Training multiple ML algorithms with different hyperparameters
    2. Evaluating and comparing model performances
    3. Selecting the best performing model
    4. Tracking experiments using MLflow
    5. Saving the final model for deployment
    
    WORKFLOW:
    1. Initialize with configuration and transformed data
    2. Load transformed train/test data
    3. Train multiple models with hyperparameter tuning
    4. Evaluate all models and select the best one
    5. Track experiments and metrics with MLflow
    6. Save the best model for production use
    """
    
    def __init__(self, model_trainer_config: ModelTrainerConfig, 
                 data_transformation_artifact: DataTransformationArtifact):
        """
        Constructor: Initialize the model trainer with configuration and data paths
        
        Args:
            model_trainer_config: Configuration for model training process
            data_transformation_artifact: Contains paths to transformed data files
        """
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    # ==============================================================================
    # MLFLOW EXPERIMENT TRACKING METHOD
    # ==============================================================================
    def track_mlflow(self, best_model, classificationmetric):
        with mlflow.start_run():
            f1_score = classificationmetric.f1_score
            precision_score = classificationmetric.precision_score
            recall_score = classificationmetric.recall_score

            mlflow.log_metric("f1_score",f1_score)
            mlflow.log_metric("precision_score",precision_score)
            mlflow.log_metric("recall_Score",recall_score)
            mlflow.sklearn.log_model(best_model,"model")



    # ==============================================================================
    # MAIN MODEL TRAINING AND EVALUATION METHOD
    # ==============================================================================
    def train_model(self, X_train, y_train, x_test, y_test):
        """
        Train multiple ML models, perform hyperparameter tuning, and select the best one
        
        ALGORITHM EXPLANATIONS:
        
        1. Random Forest: 
           - Combines multiple decision trees
           - Reduces overfitting through ensemble averaging
           - Good for both classification and feature importance
        
        2. Decision Tree:
           - Creates a tree-like model of decisions
           - Easy to interpret and visualize
           - Can overfit with deep trees
        
        3. Gradient Boosting:
           - Builds models sequentially, each correcting previous errors
           - Often achieves high accuracy
           - Can be prone to overfitting
        
        4. Logistic Regression:
           - Linear model for classification
           - Fast and interpretable
           - Assumes linear relationship between features and target
        
        5. AdaBoost (Adaptive Boosting):
           - Focuses on misclassified examples in subsequent iterations
           - Combines weak learners to create strong classifier
           - Less prone to overfitting than gradient boosting
        
        Args:
            X_train: Training features
            y_train: Training target labels
            x_test: Testing features  
            y_test: Testing target labels
            
        Returns:
            ModelTrainerArtifact: Contains trained model path and performance metrics
        """
        
        # ==================================================================
        # STEP 1: DEFINE MODELS TO TRAIN
        # ==================================================================
        # Dictionary of different ML algorithms to compare
        # verbose=1 enables progress output during training
        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            "Logistic Regression": LogisticRegression(verbose=1),
            "AdaBoost": AdaBoostClassifier(),
        }
        
        # ==================================================================
        # STEP 2: DEFINE HYPERPARAMETERS FOR TUNING
        # ==================================================================
        # Hyperparameter grids for each model
        # These parameters control model behavior and performance
        params = {
            "Decision Tree": {
                'criterion': ['gini', 'entropy', 'log_loss'],  # Splitting criteria
                # 'splitter': ['best','random'],                # Splitting strategy (commented out)
                # 'max_features': ['sqrt','log2'],              # Number of features to consider (commented out)
            },
            "Random Forest": {
                # 'criterion': ['gini', 'entropy', 'log_loss'], # Splitting criteria (commented out)
                # 'max_features': ['sqrt','log2',None],         # Features per tree (commented out)
                'n_estimators': [8, 16, 32, 128, 256]          # Number of trees in forest
            },
            "Gradient Boosting": {
                # 'loss': ['log_loss', 'exponential'],          # Loss function (commented out)
                'learning_rate': [.1, .01, .05, .001],         # Step size for gradient descent
                'subsample': [0.6, 0.7, 0.75, 0.85, 0.9],     # Fraction of samples for each tree
                # 'criterion': ['squared_error', 'friedman_mse'], # Splitting criteria (commented out)
                # 'max_features': ['auto','sqrt','log2'],       # Features per tree (commented out)
                'n_estimators': [8, 16, 32, 64, 128, 256]      # Number of boosting stages
            },
            "Logistic Regression": {},                          # No hyperparameters to tune
            "AdaBoost": {
                'learning_rate': [.1, .01, .001],              # Learning rate for weight updates
                'n_estimators': [8, 16, 32, 64, 128, 256]      # Number of weak learners
            }
        }
        
        # ==================================================================
        # STEP 3: TRAIN AND EVALUATE ALL MODELS
        # ==================================================================
        # evaluate_models function performs:
        # 1. Grid search with cross-validation for each model
        # 2. Finds best hyperparameters for each algorithm
        # 3. Trains models with best parameters
        # 4. Evaluates performance on test set
        # 5. Returns dictionary with model names and their scores
        model_report: dict = evaluate_models(
            X_train=X_train, y_train=y_train, 
            X_test=x_test, y_test=y_test,
            models=models, param=params
        )
        
        # ==================================================================
        # STEP 4: SELECT THE BEST MODEL
        # ==================================================================
        # Find the highest score from all models
        best_model_score = max(sorted(model_report.values()))

        # Find the model name corresponding to the best score
        # This finds the index of best score and uses it to get model name
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        
        # Get the actual model object
        best_model = models[best_model_name]
        
        # ==================================================================
        # STEP 5: EVALUATE BEST MODEL ON TRAINING DATA
        # ==================================================================
        # Make predictions on training data
        y_train_pred = best_model.predict(X_train)
        
        # Calculate training metrics (precision, recall, F1-score)
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
        
        # Track training metrics with MLflow
        self.track_mlflow(best_model, classification_train_metric)
        
        # ==================================================================
        # STEP 6: EVALUATE BEST MODEL ON TEST DATA
        # ==================================================================
        # Make predictions on test data
        y_test_pred = best_model.predict(x_test)
        
        # Calculate test metrics
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

        # Track test metrics with MLflow
        self.track_mlflow(best_model, classification_test_metric)

        # ==================================================================
        # STEP 7: PREPARE FINAL MODEL FOR DEPLOYMENT
        # ==================================================================
        # Load the preprocessor (data transformer) that was saved earlier
        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        
        # Create directory for saving the trained model
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        # Create a NetworkModel object that combines preprocessor and trained model
        # This ensures that the same data transformations are applied during prediction
        Network_Model = NetworkModel(preprocessor=preprocessor, model=best_model)
        
        # Save the complete model pipeline (preprocessor + model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=Network_Model)
        
        # Also save just the model for model pusher (deployment pipeline)
        save_object("final_model/model.pkl", best_model)

        # ==================================================================
        # STEP 8: CREATE AND RETURN MODEL TRAINER ARTIFACT
        # ==================================================================
        # Create artifact containing all important information about training
        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
        )
        
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact

    # ==============================================================================
    # MAIN ORCHESTRATION METHOD
    # ==============================================================================
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Main method that orchestrates the entire model training process
        
        WORKFLOW:
        1. Load transformed training and test data
        2. Separate features from target variables
        3. Train multiple models and select the best one
        4. Return training artifacts with model paths and metrics
        
        Returns:
            ModelTrainerArtifact: Contains paths and metrics for the trained model
        """
        try:
            # ==================================================================
            # STEP 1: GET FILE PATHS FOR TRANSFORMED DATA
            # ==================================================================
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # ==================================================================
            # STEP 2: LOAD TRANSFORMED DATA ARRAYS
            # ==================================================================
            # Load the numpy arrays that were saved by data transformation
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            # ==================================================================
            # STEP 3: SEPARATE FEATURES FROM TARGET VARIABLES
            # ==================================================================
            # Array slicing explanation:
            # [:, :-1] - All rows, all columns except the last one (features)
            # [:, -1]  - All rows, only the last column (target variable)
            
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],  # Training features (all columns except last)
                train_arr[:, -1],   # Training target (last column)
                test_arr[:, :-1],   # Test features (all columns except last)
                test_arr[:, -1],    # Test target (last column)
            )

            # ==================================================================
            # STEP 4: INITIATE MODEL TRAINING PROCESS
            # ==================================================================
            # Call the main training method with separated data
            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

# ==============================================================================
# SUMMARY OF THE ENTIRE MODEL TRAINING WORKFLOW
# ==============================================================================
"""
HIGH-LEVEL WORKFLOW EXPLANATION:

1. INITIALIZATION:
   - Set up model trainer with configuration and transformed data paths
   - Configure MLflow for experiment tracking

2. DATA LOADING:
   - Load transformed training and test arrays from files
   - Separate features (X) from target variables (y)

3. MODEL DEFINITION:
   - Define multiple ML algorithms to compare
   - Set hyperparameter grids for each algorithm

4. MODEL TRAINING & HYPERPARAMETER TUNING:
   - Use grid search with cross-validation
   - Train each model with different parameter combinations
   - Find best parameters for each algorithm

5. MODEL EVALUATION & SELECTION:
   - Evaluate all models on test data
   - Select the model with highest performance score
   - Calculate detailed metrics (precision, recall, F1-score)

6. EXPERIMENT TRACKING:
   - Log all metrics and models to MLflow
   - Enable comparison of different experiments
   - Store model versions for reproducibility

7. MODEL PERSISTENCE:
   - Combine best model with preprocessor
   - Save complete pipeline for deployment
   - Store model artifacts for future use

8. ARTIFACT CREATION:
   - Create comprehensive artifact with paths and metrics
   - Return information needed by subsequent pipeline stages

WHY THIS PROCESS IS IMPORTANT:
- Compares multiple algorithms objectively
- Finds optimal hyperparameters automatically
- Tracks experiments for reproducibility
- Selects best model based on performance metrics
- Prepares model for production deployment
- Provides detailed performance evaluation
- Enables model versioning and rollback capabilities

MACHINE LEARNING CONCEPTS EXPLAINED:
- Cross-validation: Technique to assess model performance reliably
- Hyperparameter tuning: Finding optimal model settings
- Ensemble methods: Combining multiple models for better performance
- Model evaluation metrics: Measuring classification performance
- Model registry: Centralized storage for trained models
- Pipeline: End-to-end workflow from data to prediction
"""