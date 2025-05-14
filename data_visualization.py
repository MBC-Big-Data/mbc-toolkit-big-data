import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Added for heatmap visualization

def data_visualization_page():
    st.title("📊 Data Visualization")

    # Help expander
    with st.expander("💡 Visualization Guide"):
        st.markdown("""
        **Understanding Visualization Types:**
        - **Histogram**: Shows distribution of numerical values
        - **Boxplot**: Displays data spread and outliers
        - **Piechart**: Shows percentage composition of categories
        - **Barplot**: Displays frequency of values
        - **Scatterplot**: Shows relationship between two numerical columns
        - **Heatmap**: Reveals correlations between numerical features
        """)

    if 'df' in st.session_state:
        df = st.session_state.df

        st.write("📋 Data Preview:")
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

        # Data type selection
        data_option = st.radio("📊 Choose data type:", 
                             ["Numeric Only", "Include Categorical"],
                             help="Select whether to include text categories in analysis")

        if data_option == "Numeric Only":
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = []
        else:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Main tabs
        univariat, bivariat = st.tabs(["📈 Univariat", "📉 Bivariat"])

        # Univariat tabs
        with univariat:
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Histogram", "📦 Boxplot", "🍰 Piechart", "📑 Barplot"])
            
            # Histogram
            with tab1:
                st.markdown("**Distribution Analysis**  \n_Visualize value distribution_")
                if numeric_cols:
                    selected_col = st.selectbox("Select numerical column:", numeric_cols, key="hist_col")
                    fig, ax = plt.subplots()
                    sns.histplot(df[selected_col], kde=True, bins=20)
                    st.pyplot(fig)
                    st.caption(f"💡 This shows how values in '{selected_col}' are distributed")
                else:
                    st.warning("No numeric columns available")

            # Boxplot
            with tab2:
                st.markdown("**Spread Analysis**  \n_Identify outliers and range_")
                if numeric_cols:
                    selected_col = st.selectbox("Select numerical column:", numeric_cols, key="box_col")
                    fig, ax = plt.subplots()
                    sns.boxplot(x=df[selected_col])
                    st.pyplot(fig)
                    st.caption(f"💡 The box shows middle 50% of values, whiskers show typical range")
                else:
                    st.warning("No numeric columns available")

            # Piechart
            with tab3:
                st.markdown("**Composition Analysis**  \n_View category percentages_")
                if categorical_cols:
                    selected_col = st.selectbox("Select category column:", categorical_cols, key="pie_col")
                    fig, ax = plt.subplots()
                    df[selected_col].value_counts().plot.pie(autopct='%1.1f%%')
                    st.pyplot(fig)
                else:
                    st.warning("No categorical columns available")

            # Barplot
            with tab4:
                st.markdown("**Frequency Analysis**  \n_Count occurrences of values_")
                if categorical_cols:
                    selected_col = st.selectbox("Select category column:", categorical_cols, key="bar_col")
                    fig, ax = plt.subplots()
                    sns.countplot(x=df[selected_col])
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                else:
                    st.warning("No categorical columns available")

        # Bivariat tabs
        with bivariat:
            tab5, tab6, tab7, tab8 = st.tabs(["🔵 Scatterplot - 2 coloumn", 
                                              "📊 Barplot - 2 coloumn", 
                                              "📦 Boxplot - 2 coloumn", 
                                              "🔥 Heatmap"])

            # Scatterplot
            with tab5:
                st.markdown("**Relationship Analysis**  \n_Compare two numerical columns_")
                if len(numeric_cols) >= 2:
                    col1, col2 = st.columns(2)
                    with col1: x_col = st.selectbox("X-axis:", numeric_cols, key="scatter_x")
                    with col2: y_col = st.selectbox("Y-axis:", numeric_cols, key="scatter_y")
                    fig, ax = plt.subplots()
                    sns.scatterplot(data=df, x=x_col, y=y_col)
                    st.pyplot(fig)
                else:
                    st.warning("Need at least 2 numerical columns")

            # Barplot (2 columns)
            with tab6:
                st.markdown("**Comparison Analysis**  \n_Categorical vs Numerical values_")
                if categorical_cols and numeric_cols:
                    col1, col2 = st.columns(2)
                    with col1: cat_col = st.selectbox("Category:", categorical_cols, key="bar2_cat")
                    with col2: num_col = st.selectbox("Numerical:", numeric_cols, key="bar2_num")
                    fig, ax = plt.subplots()
                    sns.barplot(x=cat_col, y=num_col, data=df)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                else:
                    st.warning("Need both categorical and numerical columns")

            # Boxplot (2 columns)
            with tab7:
                st.markdown("**Group Distribution**  \n_Compare numerical spread across categories_")
                if categorical_cols and numeric_cols:
                    col1, col2 = st.columns(2)
                    with col1: cat_col = st.selectbox("Category:", categorical_cols, key="box2_cat")
                    with col2: num_col = st.selectbox("Numerical:", numeric_cols, key="box2_num")
                    fig, ax = plt.subplots()
                    sns.boxplot(x=cat_col, y=num_col, data=df)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                else:
                    st.warning("Need both categorical and numerical columns")

            # Heatmap
            with tab8:
                st.markdown("**Correlation Analysis**  \n_See relationships between numerical features_")
                if len(numeric_cols) >= 2:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
                    st.pyplot(fig)
                    st.caption("💡 Values close to 1 (red) = Strong positive relationship  \n"
                             "💡 Values close to -1 (blue) = Strong negative relationship")
                else:
                    st.warning("Need at least 2 numerical columns")

    else:
        st.warning("⚠️ Please upload data first on the Data page!")
