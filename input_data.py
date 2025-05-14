import streamlit as st
import pandas as pd
import io

def data_page():
    st.title("📊 Data Explorer")
    st.header("🚀 Upload Your Dataset", divider="rainbow")
    
    with st.expander("💡 **Quick Guide for Beginners**"):
        st.markdown("""
        **Welcome! Here's what you need to know:**
        - **CSV File**: A simple spreadsheet format (like Excel but simpler)
        - **Rows**: Each line in your data (like one customer's information)
        - **Columns**: Different types of information (like "Name" or "Age")
        - **Null Values**: Missing information spots in your data
        - **Duplicates**: Repeated/identical entries in your data
        """)

    # Check for existing data first
    if 'df' in st.session_state and st.session_state.df is not None:
        st.success("✅ Dataset already loaded! You can upload a new file below if needed.")
        df = st.session_state.df
    else:
        df = None

    # File uploader (always visible)
    uploaded_file = st.file_uploader("📤 Upload your data file (CSV format)", 
                                   type=["csv"],
                                   help="Choose a CSV file - this is like a simple spreadsheet format")
    
    # Handle new file upload
    if uploaded_file is not None:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            df = st.session_state.df
            st.success("✅ File uploaded successfully!")
        except Exception as e:
            st.error(f"🚨 Oops! Something went wrong: {str(e)}")
            return

    # Show data analysis if data exists
    if 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        
        # Metrics Section with Explanations
        st.subheader("📈 Basic Dataset Information")
        cols = st.columns(4)
        metrics = [
            ("📏 Rows", len(df), "Total number of entries in your data"),
            ("📐 Columns", len(df.columns), "Different types of information tracked"),
            ("❓ Missing Values", df.isnull().sum().sum(), "Empty spots needing attention"),
            ("🔄 Duplicates", df.duplicated().sum(), "Repeated entries in your data")
        ]
        
        for col, (title, value, help_text) in zip(cols, metrics):
            with col:
                st.metric(title, value, help=help_text)
        
        # Tabs with User Guidance
        tab1, tab2, tab3, tab4 = st.tabs([
            "📂 Full Data View", 
            "⚠️ Missing Info", 
            "♻️ Repeated Entries", 
            "🔠 Data Formats"
        ])

        with tab1:
            st.markdown("**Your Complete Data** - Browse through all your information")
            st.dataframe(df, use_container_width=True, hide_index=True, height=300)
            st.caption("💡 Tip: Scroll horizontally to see all columns")

        with tab2:
            st.markdown("**Missing Information** - These columns have empty spots")
            null_counts = df.isnull().sum()
            if null_counts.sum() > 0:
                null_df = null_counts[null_counts > 0].reset_index()
                null_df.columns = ["Coloumn Name", "Missing Spots"]
                st.dataframe(null_df, use_container_width=True, hide_index=True)
                st.info("🔍 **Why this matters:** Missing data can affect analysis accuracy")
            else:
                st.success("🎉 Perfect! No missing information found!")

        with tab3:
            st.markdown("**Repeated Entries** - Exact duplicates in your data")
            dupes = df[df.duplicated()]
            if not dupes.empty:
                st.dataframe(dupes, use_container_width=True, hide_index=True)
                st.warning("⚠️ **Note:** Duplicates might mean repeated records")
            else:
                st.success("✨ Clean! No duplicate entries found!")

        with tab4:
            st.markdown("**Data Formats** - How different information is stored")
            dtype_df = df.dtypes.reset_index()
            dtype_df.columns = ["Coloumn Name", "Storage Format"]
            dtype_df["Storage Format"] = dtype_df["Storage Format"].astype(str)
            st.dataframe(dtype_df, use_container_width=True, hide_index=True)
            st.markdown("""
            **Format Guide:**
            - `object`: Text/General information
            - `int64`: Whole numbers
            - `float64`: Decimal numbers
            - `datetime`: Dates/Times
            """)
