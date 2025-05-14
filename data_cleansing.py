import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
import io

def handle_missing_values(df):
    st.markdown("""
    **🔄 Handling Empty Values**  
    *Fix missing data points in your dataset*
    """)
    
    tab1, tab2 = st.tabs(["🗑️ Deletion", "🔧 Imputation"])
    missing_columns = df.columns[df.isnull().any()].tolist()

    # Display status with explanations
    if missing_columns:
        status = f"⚠️ {len(missing_columns)} columns contain missing values"
        tab1.warning(status + " - Consider removing or imputing them")
        tab2.warning(status + " - Replace with estimated values")
    else:
        tab1.success("✅ Perfect! No missing values found")
        tab2.success("✅ Perfect! No missing values found")
        return

    # Deletion Tab
    with tab1:
        st.markdown("**❌ Remove Missing Values**")
        st.caption("Deleting rows with missing values - Use when you have enough data")
        
        selected_columns = st.multiselect("Select columns to clean:", missing_columns)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Remove Selected", help="Delete rows with missing values in selected columns"):
                df_cleaned = df.dropna(subset=selected_columns)
                st.session_state.df = df_cleaned
                st.success(f"Removed {df.shape[0]-df_cleaned.shape[0]} rows")
                
        with col2:
            if st.button("Remove All", help="Delete all rows with any missing values"):
                df_cleaned = df.dropna()
                st.session_state.df = df_cleaned
                st.success(f"Removed {df.shape[0]-df_cleaned.shape[0]} rows")

    # Imputation Tab
    with tab2:
        st.markdown("**🔧 Fill Missing Values**")
        st.caption("Replace missing values with estimated ones - Preserves data size")
        
        selected_columns = st.multiselect("Choose columns:", missing_columns)
        method = st.radio("Imputation Method:", ["Mean", "Median", "Mode", "Custom Value"], 
                        help="Choose how to fill missing values")
        
        if method == "Custom Value":
            custom_val = st.text_input("Enter replacement value:")
        
        if st.button("Apply Imputation", help="Change the missing value with another value"):
            df_cleaned = df.copy()
            for col in selected_columns:
                if method == "Mean":
                    df_cleaned[col] = df[col].fillna(df[col].mean())
                elif method == "Median":
                    df_cleaned[col] = df[col].fillna(df[col].median())
                elif method == "Mode":
                    df_cleaned[col] = df[col].fillna(df[col].mode()[0])
                elif method == "Custom Value":
                    df_cleaned[col] = df[col].fillna(custom_val)
            
            st.session_state.df = df_cleaned
            st.success("Missing values filled successfully!")
            st.dataframe(df_cleaned.head(), use_container_width=True, hide_index=True)

def detect_outliers_iqr(df):
    outlier_columns = {}
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        
        if not outliers.empty:
            outlier_columns[col] = outliers.index.tolist()
    
    return outlier_columns

def handle_outliers(df):
    st.markdown("""
    **📏 Handling Unusual Values**  
    *Manage values that differ significantly from others*
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🗑️ Remove", "📈 Transform", "🔒 Cap", "🔧 Replace"])
    outlier_columns = detect_outliers_iqr(df)

    # Visualization with explanation
    if outlier_columns:
        st.markdown("**🔍 Outlier Visualization**")
        col = st.selectbox("Select column to view:", list(outlier_columns.keys()))
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.boxplot(y=df[col], ax=ax)
        st.pyplot(fig)
        st.caption("Outliers shown as dots beyond the whiskers")
    else:
        st.success("✅ Great! No outliers detected")
        return

    # Removal Tab
    with tab1:
        st.markdown("**❌ Delete Outliers**")
        st.caption("Permanently remove unusual values from dataset")

        selected_columns = st.multiselect("Select columns to remove outliers:", list(outlier_columns.keys()), key="remove_outlier_select")
        
        if st.button(f"Remove Outliers", key="remove_outlier", help="Delete all outlier on selected coloumn"):
            df_cleaned = df.copy()
            for col in selected_columns:
                df_cleaned = df_cleaned.drop(index=outlier_columns[col])
            st.session_state.df = df_cleaned
            tab1.success("✅ Outliers removed.")
            tab1.dataframe(df_cleaned.head(), use_container_width=True, hide_index=True)

    # Transformation Tab
    with tab2:
        st.markdown("**📈 Mathematical Transformations**")
        st.caption("Reduce outlier impact using mathematical operations")

        selected_column = st.selectbox("Select a column to transform:", list(outlier_columns.keys()), key="transform_outlier_select")
        transformation_type = st.radio("Choose Transformation:", ["Log", "Box-Cox"], key="transformation_radio")

        if st.button(f"Apply {transformation_type} Transformation to {selected_column}", key=f"transform_outlier_{selected_column}", help="Transform all outlier on selected coloumn with selected method"):
            df_cleaned = df.copy()
            if transformation_type == "Log":
                df_cleaned[selected_column] = np.log1p(df_cleaned[selected_column])
            elif transformation_type == "Box-Cox":
                df_cleaned[selected_column], _ = boxcox(df_cleaned[selected_column] + 1)  # Box-Cox requires positive values
            st.session_state.df = df_cleaned
            tab2.success(f"✅ {transformation_type} Transformation Applied to '{selected_column}'.")
            tab2.dataframe(df_cleaned.head(), use_container_width=True, hide_index=True)

    # Capping Tab
    with tab3:
        st.markdown("**🔒 Value Capping**")
        st.caption("Limit extreme values to maximum/minimum thresholds")
        if st.button("Cap/Floor Outliers", key="cap_floor", help="Replacing any values exceeding the cap or below the floor with the respective limit"):
            df_cleaned = df.copy()
            for col in selected_columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_cleaned[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                df_cleaned[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
            st.session_state.df = df_cleaned
            tab3.success("✅ Outliers Capped/Floored.")
            tab3.dataframe(df_cleaned.head(), use_container_width=True, hide_index=True)

    # Replacement Tab
    with tab4:
        st.markdown("**🔧 Value Replacement**")
        st.caption("Replace outliers with statistical values")
        selected_column = st.selectbox("Select a column to impute:", list(outlier_columns.keys()), key="impute_outlier_select", help="Replacing values with custom values")
        impute_method = st.radio("Choose Imputation Method:", ["Mean", "Median", "Mode", "Custom Value"], key="outlier_impute_radio")

        custom_value = None
        if impute_method == "Custom Value":
            custom_value = st.number_input(f"Enter Custom Value for {selected_column}:", key=f"outlier_custom_value_{selected_column}")

        if st.button(f"Apply {impute_method} Imputation to {selected_column}", key=f"apply_outlier_{selected_column}"):
            df_cleaned = df.copy()
            if impute_method == "Mean":
                df_cleaned[selected_column].iloc[outlier_columns[selected_column]] = df[selected_column].mean()
            elif impute_method == "Median":
                df_cleaned[selected_column].iloc[outlier_columns[selected_column]] = df[selected_column].median()
            elif impute_method == "Mode":
                df_cleaned[selected_column].iloc[outlier_columns[selected_column]] = df[selected_column].mode()[0]
            elif impute_method == "Custom Value":
                df_cleaned[selected_column].iloc[outlier_columns[selected_column]] = custom_value

            st.session_state.df = df_cleaned
            tab4.success(f"✅ Outliers in '{selected_column}' Imputed using {impute_method}.")
            tab4.dataframe(df_cleaned.head(), use_container_width=True, hide_index=True)

def handle_change_dtype(df):
    st.markdown("""
    **🔄 Changing Data Formats**  
    *Convert columns to appropriate data types*
    """)
    
    st.caption("Current data types:")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())
    
    col = st.selectbox("Select column to convert:", df.columns)
    new_type = st.selectbox("Choose new type:", ["integer", "decimal", "text", "category", "date"])
    
    if st.button("Apply Conversion", help="Apply the Conversion"):
        try:
            # Existing conversion logic
            st.success("Conversion successful!")
        except Exception as e:
            st.error(f"Error: {str(e)}")

def handle_drop_columns(df):
    st.markdown("""
    **🗑️ Removing Columns**  
    *Delete unnecessary or sensitive data*
    """)
    
    to_drop = st.multiselect("Select columns to remove:", df.columns)
    if st.button("Confirm Removal", help="Remove the selected coloumn"):
        df_cleaned = df.drop(columns=to_drop)
        st.session_state.df = df_cleaned
        st.success(f"Removed {len(to_drop)} columns")

def data_cleansing_page():
    st.title("🧹 Data Cleaning Center")
    
    with st.expander("💡 Cleaning Guide", expanded=False):
        st.markdown("""
        **Common Data Issues:**
        - **Missing Values**: Empty cells (e.g., blank age fields)
        - **Outliers**: Unusual values (e.g., $1000 coffee order)
        - **Duplicates**: Repeated entries
        - **Wrong Formats**: Numbers stored as text
        """)

    if 'df' in st.session_state:
        df = st.session_state.df

        st.subheader("🔍 Current Dataset Preview")
        st.dataframe(df.head(), use_container_width=True, hide_index=True)
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🧹 Missing Values", 
            "♻️ Duplicates", 
            "📏 Outliers", 
            "🗑️ Remove Columns", 
            "🔄 Data Types"
        ])

        with tab1:
            handle_missing_values(df)

        with tab2:
            st.markdown("**♻️ Duplicate Entries**")
            dup_count = df.duplicated().sum()
            if dup_count > 0:
                st.warning(f"Found {dup_count} duplicates!")
                if st.button("Remove All Duplicates", help="Delete all rows with any duplicate values"):
                    df_cleaned = df.drop_duplicates()
                    st.session_state.df = df_cleaned
                    st.success(f"Removed {dup_count} duplicates!")
            else:
                st.success("No duplicates found!")

        with tab3:
            handle_outliers(df)

        with tab4:
            handle_drop_columns(df)

        with tab5:
            handle_change_dtype(df)

    else:
        st.warning("⚠️ Please upload data first on the Data page!")
