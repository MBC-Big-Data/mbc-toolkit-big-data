import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

def data_preprocessing_page():
    st.title("🔧 Data Preparation Center")
    
    with st.expander("💡 Preprocessing Guide", expanded=False):
        st.markdown("""
        **Understanding Data Preparation:**
        - **Feature Scaling**: Adjust numerical ranges for better model performance
        - **Feature Engineering**: Create new insights from existing data
        - **Encoding**: Convert text categories to numerical values
        - **Custom Operations**: Write your own transformation logic
        """)

    if 'df' in st.session_state:
        df = st.session_state.df

        st.subheader("🔍 Current Dataset Preview")
        st.dataframe(df.head(), use_container_width=True, hide_index=True)
        st.caption("💡 Tip: Processed changes will update this preview automatically")

        st.subheader("✨ Preparation Tools")
        tab1, tab2, tab3 = st.tabs(["📏 Scale Features", "⚙️ Engineer Features", "🔠 Encode Categories"])

        with tab1:
            feature_scaling(df)

        with tab2:
            feature_engineering(df)

        with tab3:
            encoding(df)
    else:
        st.warning("⚠️ Please upload your data first on the Data page!")

def feature_scaling(df):
    st.markdown("""
    **📐 Adjust Numerical Ranges**  
    *Standardize values for better model performance*
    """)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_columns:
        st.warning("⚠️ No numerical columns found for scaling")
        return

    col1, col2 = st.columns([2, 3])
    with col1:
        scaler_type = st.radio("**Scaling Method**:", 
                             ["Min-Max Scaling", "Standardization", "Robust Scaling"],
                             help="Choose how to adjust numerical ranges")
        
        scaler = {
            "Min-Max Scaling": ("🔢 Scale between 0-1", MinMaxScaler()),
            "Standardization": ("📊 Mean=0, Std=1", StandardScaler()),
            "Robust Scaling": ("🛡️ Outlier-resistant", RobustScaler())
        }[scaler_type]

        st.caption(scaler[0])

    with col2:
        selected_columns = st.multiselect("Select columns to scale:", numeric_columns,
                                        help="Choose numerical features to transform")
        
        if selected_columns and st.button("⚡ Apply Scaling", use_container_width=True):
            try:
                df[selected_columns] = scaler[1].fit_transform(df[selected_columns])
                st.session_state.df = df
                st.toast(f"✅ {scaler_type} applied to {len(selected_columns)} columns", icon="🎯")
                st.rerun()
            except Exception as e:
                st.error(f"🚨 Scaling failed: {str(e)}")

def feature_engineering(df):
    st.markdown("""
    **⚙️ Create New Features**  
    *Derive new insights from existing data*
    """)
    
    action = st.radio("**Choose Operation**:", 
                    ["➕ Create New Feature", "✂️ Split Feature", "💻 Custom Code"],
                    horizontal=True)

    if action == "➕ Create New Feature":
        col1, col2 = st.columns(2)
        with col1:
            col_a = st.selectbox("First Column", df.columns)
            col_b = st.selectbox("Second Column", df.columns)
        with col2:
            operation = st.selectbox("Operation", ["Add", "Subtract", "Multiply", "Divide"])
            new_name = st.text_input("New Feature Name")
        
        if st.button("✨ Create Feature", use_container_width=True):
            try:
                if operation == "Add":
                    df[new_name] = df[col_a] + df[col_b]
                elif operation == "Subtract":
                    df[new_name] = df[col_a] - df[col_b]
                elif operation == "Multiply":
                    df[new_name] = df[col_a] * df[col_b]
                elif operation == "Divide":
                    df[new_name] = df[col_a] / df[col_b].replace(0, np.nan)
                st.session_state.df = df
                st.success(f"Created new feature: '{new_name}'")
            except Exception as e:
                st.error(f"🚨 Creation failed: {str(e)}")

    elif action == "✂️ Split Feature":
        col1, col2 = st.columns(2)
        with col1:
            split_col = st.selectbox("Column to split", df.columns)
            delimiter = st.text_input("Split delimiter", value=",")
        with col2:
            prefix = st.text_input("New column prefix", value=split_col)
        
        if st.button("✂️ Split Column", use_container_width=True):
            try:
                split_df = df[split_col].str.split(delimiter, expand=True)
                for i in range(split_df.shape[1]):
                    df[f"{prefix}_{i+1}"] = split_df[i]
                st.session_state.df = df
                st.success(f"Split {split_col} into {split_df.shape[1]} columns")
            except Exception as e:
                st.error(f"🚨 Splitting failed: {str(e)}")

    elif action == "💻 Custom Code":
        st.markdown("""
        **Write Custom Transformations**  
        *Example:*  
        `df['new_feature'] = df['height'] * df['weight']`  
        `df = df.drop(columns=['old_column'])`
        """)
        code = st.text_area("Python code (df is your dataframe):", height=150)
        if st.button("🚀 Execute Code", use_container_width=True):
            try:
                exec_env = {'df': df, 'np': np, 'pd': pd}
                exec(code, exec_env)
                st.session_state.df = exec_env['df']
                st.toast("✅ Code executed successfully!", icon="🎉")
                st.rerun()
            except Exception as e:
                st.error(f"🚨 Execution error: {str(e)}")

def encoding(df):
    st.markdown("""
    **🔡 Convert Categories to Numbers**  
    *Prepare text data for machine learning*
    """)
    
    method = st.radio("**Encoding Method**:", 
                    ["🏷️ Label Encoding", "🔥 One-Hot Encoding", "🔢 Ordinal Encoding"],
                    horizontal=True)

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_cols:
        st.warning("⚠️ No categorical columns found")
        return

    col = st.selectbox("Select column to encode:", categorical_cols)

    if method == "🏷️ Label Encoding":
        st.caption("Convert categories to unique numerical codes")
        if st.button("Apply Label Encoding", use_container_width=True):
            df[col] = df[col].astype('category').cat.codes
            st.session_state.df = df
            st.success(f"Label encoded {col}")

    elif method == "🔥 One-Hot Encoding":
        st.caption("Create separate columns for each category")
        if st.button("Apply One-Hot Encoding", use_container_width=True):
            df = pd.get_dummies(df, columns=[col], prefix=col)
            st.session_state.df = df
            st.success(f"Created {len(df[col].unique())} new columns")

    elif method == "🔢 Ordinal Encoding":
        st.caption("Assign ordered numerical values based on category importance")
        categories = st.multiselect("Set category order:", 
                                  df[col].unique().tolist(),
                                  default=df[col].unique().tolist())
        if categories and st.button("Apply Ordinal Encoding", use_container_width=True):
            df[col] = df[col].astype(pd.CategoricalDtype(categories=categories, ordered=True)).cat.codes
            st.session_state.df = df
            st.success(f"Ordinal encoded {col} with custom order")
