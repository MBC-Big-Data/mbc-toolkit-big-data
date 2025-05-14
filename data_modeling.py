import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet,
    SGDRegressor, SGDClassifier, Perceptron, PassiveAggressiveClassifier,
    RidgeClassifier, LogisticRegressionCV, TheilSenRegressor, HuberRegressor, RANSACRegressor
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, log_loss,
    matthews_corrcoef, cohen_kappa_score, mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score, max_error, median_absolute_error
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.utils.validation import check_is_fitted

def modeling_page():
    st.title("🤖 Smart Model Builder")
    
    with st.expander("💡 Modeling Guide", expanded=False):
        st.markdown("""
        **Understanding Machine Learning:**
        - **Regression**: Predict numerical values (like prices or temperatures)
        - **Classification**: Predict categories (like yes/no or types)
        - **Features**: The input data used to make predictions
        - **Target**: The value we want to predict
        - **Training**: Teaching the model using existing data
        """)

    if 'df' in st.session_state:
        df = st.session_state.df

        # Model Upload Section
        with st.expander("📤 Upload Pre-trained Model", expanded=False):
            uploaded_model = st.file_uploader("Choose a model file (.pkl)", type=["pkl"])
            if uploaded_model:
                try:
                    model = joblib.load(uploaded_model)
                    st.success("✅ Model loaded successfully!")
                    
                    # Model type detection
                    model_type = "classification" if hasattr(model, "predict_proba") else "regression"
                    st.session_state.current_model = model
                    st.session_state.model_type = model_type

                except Exception as e:
                    st.error(f"🚨 Error loading model: {str(e)}")

        # Main Modeling Section
        st.subheader("🔧 Build New Model")
        tab1, tab2 = st.tabs(["⚙️ Setup", "📊 Results"])

        with tab1:
            task_type = st.radio("**What are you predicting?**", 
                                ["Regression", "Classification"],
                                help="Numbers = Regression, Categories = Classification")
            
            target_column = st.selectbox("**Select target to predict:**", df.columns,
                                        help="The value you want the model to learn to predict")
            
            model_options = {
                "Regression": [
                    "Random Forest", "Gradient Boosting", "XGBoost", "Linear Regression",
                    "Ridge Regression", "Lasso Regression", "ElasticNet Regression",
                    "SGD Regression", "Theil-Sen Estimator", "Huber Regression", "RANSAC Regression"
                ],
                "Classification": [
                    "Logistic Regression", "Logistic Regression CV", "Random Forest",
                    "Gradient Boosting", "XGBoost", "Bagging", "SGD Classifier",
                    "Perceptron", "Passive Aggressive Classifier", "Ridge Classifier"
                ]
            }
            
            model_type = st.selectbox("**Choose algorithm:**", model_options[task_type],
                                    help="Different algorithms work best for different problems")
            
            st.markdown("---")
            if st.button("🚀 Train Model", use_container_width=True):
                st.session_state.train_model = True

        if st.session_state.get('train_model', False):
            with tab2:
                with st.spinner("🧠 Training model... This may take a few minutes"):
                    try:
                        X = df.drop(columns=[target_column])
                        y = df[target_column]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                        # Model configuration
                        model_classes = {
                            "Random Forest": (RandomForestClassifier, RandomForestRegressor),
                            "Gradient Boosting": (GradientBoostingClassifier, GradientBoostingRegressor),
                            "XGBoost": (xgb.XGBClassifier, xgb.XGBRegressor),
                            "Logistic Regression": (LogisticRegression, None),
                            "Bagging": (BaggingClassifier, None),
                            "Linear Regression": (None, LinearRegression),
                            "Ridge Regression": (None, Ridge),
                            "Lasso Regression": (None, Lasso),
                            "ElasticNet Regression": (None, ElasticNet),
                            "SGD Regression": (None, SGDRegressor),
                            "Theil-Sen Estimator": (None, TheilSenRegressor),
                            "Huber Regression": (None, HuberRegressor),
                            "RANSAC Regression": (None, RANSACRegressor),
                            "SGD Classifier": (SGDClassifier, None),
                            "Perceptron": (Perceptron, None),
                            "Passive Aggressive Classifier": (PassiveAggressiveClassifier, None),
                            "Ridge Classifier": (RidgeClassifier, None),
                            "Logistic Regression CV": (LogisticRegressionCV, None),
                        }

                        model_class_tuple = model_classes[model_type]
                        model_class = model_class_tuple[0] if task_type == "Classification" else model_class_tuple[1]
                        model = model_class()

                        # Hyperparameter tuning
                        params_grid = {
                            "Random Forest": {'n_estimators': [100, 200], 'max_depth': [None, 10], 'min_samples_split': [2, 5]},
                            "Gradient Boosting": {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.2], 'max_depth': [3, 5]},
                            "XGBoost": {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.2], 'max_depth': [3, 5]},
                            "Logistic Regression": {'C': [0.1, 1], 'solver': ['liblinear', 'lbfgs']},
                            "Bagging": {'n_estimators': [10, 50], 'max_samples': [0.7, 1.0]},
                            "Linear Regression": {'fit_intercept': [True, False]},
                            "Ridge Regression": {'alpha': [0.1, 1.0], 'solver': ['auto', 'svd']},
                            "Lasso Regression": {'alpha': [0.1, 1.0], 'selection': ['cyclic', 'random']},
                            "ElasticNet Regression": {'alpha': [0.1, 1.0], 'l1_ratio': [0.5, 0.8]},
                            "SGD Regression": {'loss': ['squared_error', 'huber'], 'alpha': [0.0001, 0.001]},
                            "Theil-Sen Estimator": {'n_subsamples': [50, 100]},
                            "Huber Regression": {'epsilon': [1.35, 1.5], 'alpha': [0.001, 0.01]},
                            "RANSAC Regression": {'min_samples': [0.5, None]},
                            "SGD Classifier": {'loss': ['hinge', 'log_loss'], 'alpha': [0.0001, 0.001]},
                            "Perceptron": {'penalty': ['l2', 'l1'], 'alpha': [0.0001, 0.001]},
                            "Passive Aggressive Classifier": {'C': [0.1, 1.0], 'loss': ['hinge', 'squared_hinge']},
                            "Ridge Classifier": {'alpha': [0.1, 1.0], 'solver': ['auto', 'svd']},
                            "Logistic Regression CV": {'Cs': [10, 100], 'solver': ['liblinear', 'lbfgs']}
                        }

                        grid_search = GridSearchCV(model, params_grid[model_type], cv=3, 
                                                 scoring='accuracy' if task_type == "Classification" else 'r2')
                        grid_search.fit(X_train, y_train)
                        best_model = grid_search.best_estimator_
                        y_pred = best_model.predict(X_test)

                        # Display results
                        st.subheader("📊 Model Performance")
                        
                        if task_type == "Classification":
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
                                st.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.2%}")
                            with col2:
                                st.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.2%}")
                                st.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.2%}")

                            st.markdown("**Confusion Matrix**")
                            cm = confusion_matrix(y_test, y_pred)
                            fig, ax = plt.subplots()
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                      xticklabels=np.unique(y), yticklabels=np.unique(y))
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Actual')
                            st.pyplot(fig)

                        else:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
                                st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
                            with col2:
                                st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
                                st.metric("Max Error", f"{max_error(y_test, y_pred):.2f}")

                            fig, ax = plt.subplots()
                            sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.3})
                            ax.set_xlabel('True Values')
                            ax.set_ylabel('Predicted Values')
                            ax.set_title('True vs Predicted Values')
                            st.pyplot(fig)

                        # Model saving
                        model_filename = "trained_model.pkl"
                        joblib.dump(best_model, model_filename)
                        
                        with open(model_filename, "rb") as f:
                            st.download_button(
                                label="📥 Download Trained Model",
                                data=f,
                                file_name=model_filename,
                                mime="application/octet-stream",
                                use_container_width=True
                            )

                    except Exception as e:
                        st.error(f"🚨 Training failed: {str(e)}")

    else:
        st.warning("⚠️ Please upload your data first on the Data page!")
