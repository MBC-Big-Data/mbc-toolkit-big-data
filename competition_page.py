import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def competition():
    st.title("🏆 Kaggle-style Competition")
    
    with st.expander("💡 Competition Guide", expanded=False):
        st.markdown("""
        **How to Participate:**
        1. **Upload Files**: Trained model + Test data + Sample submission
        2. **Generate Predictions**: Let the model make predictions on test data
        3. **Download Results**: Get submission file ready for competition upload
        4. **Submit**: Upload your CSV to the competition platform
        """)

    # Session state initialization
    if 'submission' not in st.session_state:
        st.session_state.submission = None
    if 'target_column' not in st.session_state:
        st.session_state.target_column = None
    
    # Upload section
    st.subheader("📤 Step 1: Upload Required Files")
    with st.expander("Click to expand upload section", expanded=True):
        col1, col2, col3 = st.columns(3)
        model_file = col1.file_uploader("Trained Model (.pkl)", 
                                      type=["pkl"],
                                      help="Your trained machine learning model file")
        test_file = col2.file_uploader("Test Data (CSV)", 
                                     type=["csv"],
                                     help="Data for making predictions (without target column)")
        sample_file = col3.file_uploader("Sample Submission (CSV)", 
                                        type=["csv"],
                                        help="Example submission format from competition")
    
    # Load data
    if model_file and test_file and sample_file:
        try:
            with st.spinner("⏳ Loading files..."):
                model = joblib.load(model_file)
                test_df = pd.read_csv(test_file).drop(columns=['Unnamed: 0'], errors='ignore')
                sample_sub = pd.read_csv(sample_file).drop(columns=['Unnamed: 0'], errors='ignore')
                
                # Store in session state
                st.session_state.model = model
                st.session_state.test_data = test_df
                st.session_state.sample_sub = sample_sub
                
                # Auto-detect target column
                target_options = [col for col in sample_sub.columns if col != 'ID']
                if target_options:
                    st.session_state.target_column = target_options[0]
                
                st.toast("✅ All files loaded successfully!", icon="🎉")
        except Exception as e:
            st.error(f"🚨 Error loading files: {str(e)}")
    
    # Data preview section
    if 'test_data' in st.session_state and 'sample_sub' in st.session_state:
        st.subheader("🔍 Step 2: Data Preview")
        with st.expander("View test data and sample format", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Test Data Preview**")
                st.caption("Data your model will use for predictions")
                st.dataframe(st.session_state.test_data.head(), use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("**Sample Submission Format**")
                st.caption("Required format for competition submission")
                st.dataframe(st.session_state.sample_sub.head(), use_container_width=True, hide_index=True)
    
    # Target column selection
    if 'sample_sub' in st.session_state and st.session_state.sample_sub is not None:
        st.subheader("🎯 Step 3: Select Target Column")
        target_options = [col for col in st.session_state.sample_sub.columns if col != 'ID']
        if target_options:
            new_target = st.selectbox(
                "Choose prediction column name:",
                options=target_options,
                index=0,
                help="Select which column to populate with predictions",
                key='target_select'
            )
            st.session_state.target_column = new_target
    
    # Prediction section
    if 'model' in st.session_state and 'test_data' in st.session_state and st.session_state.target_column:
        st.subheader("🔮 Step 4: Generate Predictions")
        with st.expander("Make and review predictions", expanded=True):
            if st.button("🚀 Generate Predictions", use_container_width=True):
                try:
                    with st.spinner("🧠 Making predictions..."):
                        # Preprocess test data
                        test_df = st.session_state.test_data.copy().drop(columns=['Unnamed: 0'], errors='ignore')
                        X_test = test_df.drop('ID', axis=1, errors='ignore')
                        
                        # Handle categorical features
                        categorical_cols = X_test.select_dtypes(include=['object']).columns
                        if not categorical_cols.empty:
                            encoder = OneHotEncoder(handle_unknown='ignore')
                            X_test_encoded = encoder.fit_transform(X_test[categorical_cols])
                            X_test = X_test.drop(categorical_cols, axis=1)
                            X_test = pd.concat([X_test, pd.DataFrame(X_test_encoded.toarray())], axis=1)
                            st.toast(f"🔡 Encoded {len(categorical_cols)} categorical features", icon="ℹ️")
                        
                        # Align features with model
                        if hasattr(st.session_state.model, 'feature_names_in_'):
                            missing_features = set(st.session_state.model.feature_names_in_) - set(X_test.columns)
                            for feature in missing_features:
                                X_test[feature] = 0
                            X_test = X_test[st.session_state.model.feature_names_in_]
                        
                        # Make predictions
                        predictions = st.session_state.model.predict(X_test)
                        
                        # Create submission
                        submission = st.session_state.sample_sub.copy().drop(columns=['Unnamed: 0'], errors='ignore')
                        submission[st.session_state.target_column] = predictions.astype(int)
                        st.session_state.submission = submission
                        
                        st.success(f"✅ Successfully predicted {len(predictions)} samples!")
                        st.balloons()

                    st.subheader("📝 Step 5: Review & Download")
                    edited_sub = st.data_editor(
                        st.session_state.submission, 
                        use_container_width=True, 
                        hide_index=True,
                        disabled=['ID']
                    )
                
                    csv = edited_sub.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Download Submission CSV",
                        data=csv,
                        file_name="submission.csv",
                        mime="text/csv",
                        use_container_width=True,
                        help="Download your competition-ready submission file"
                    )
                    
                except Exception as e:
                    st.error(f"🚨 Prediction failed: {str(e)}")

    # Initial state message
    if not st.session_state.get('model'):
        st.info("ℹ️ Please upload all required files to begin (model, test data, sample submission)")
