import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Consoleflare Analytics Portal", page_icon='📊')
st.title(":rainbow[Data Analytics Portal]")
st.subheader(':grey[Explore Data With Ease]', divider='blue')

file = st.file_uploader("Drop csv or excel file here", type=['csv', 'xlsx'])

if file is not None:
    if file.name.endswith('.csv'):
        data = pd.read_csv(file)
    else:
        data = pd.read_excel(file)

    st.dataframe(data)
    st.info('File is successfully uploaded!')

    st.subheader(':rainbow[Basic Information of the dataset]', divider='blue')
    tab1, tab2, tab3, tab4 = st.tabs(['Summary', 'Top & Bottom Rows', 'Data Types', 'Columns'])

    # ✅ FIX 1: All tab blocks are now INSIDE the `if file is not None` block
    with tab1:
        st.write(f'There are {data.shape[0]} rows and {data.shape[1]} columns in the dataset.')
        st.subheader(':grey[Statistical Summary of the dataset]')
        st.dataframe(data.describe())

    with tab2:
        st.subheader(':grey[Top Rows]')
        # ✅ FIX 2: Use unique keys to avoid duplicate slider widget errors
        top_rows = st.slider('Select number of top rows to display', 1, data.shape[0], key='top_slider')
        st.dataframe(data.head(top_rows))
        st.subheader(':grey[Bottom Rows]')
        bottom_rows = st.slider('Select number of bottom rows to display', 1, data.shape[0], key='bottom_slider')
        st.dataframe(data.tail(bottom_rows))

    with tab3:
        st.subheader(':grey[Data Types of the dataset]')
        st.dataframe(data.dtypes)

    with tab4:
        st.subheader(':grey[Columns of the dataset]')
        st.write(list(data.columns))

    # ── Value Count Section ──────────────────────────────────────────────────
    st.subheader(':rainbow[Column Values To Count]', divider='blue')
    with st.expander('Value Count'):
        col1, col2 = st.columns(2)
        with col1:
            column = st.selectbox('Choose Column Name', options=list(data.columns))
        with col2:
            top_n = st.number_input("Top Rows", min_value=1, step=1)

    # ✅ FIX 3: Button and result block are INSIDE `if file is not None`
    if st.button('Count'):
        result = data[column].value_counts().reset_index().head(top_n)
        result.columns = [column, 'count']  # ✅ FIX 4: Explicitly name columns for compatibility
        st.dataframe(result)
        st.subheader("Visualization", divider='blue')
        fig = px.bar(data_frame=result, x=column, y='count')
        st.plotly_chart(fig)
        fig = px.line(data_frame=result, x=column, y='count')
        st.plotly_chart(fig)

    # ── GroupBy Section ──────────────────────────────────────────────────────
    st.subheader(':rainbow[GroupBy: Simplify Your Data Analysis]', divider='blue')
    st.write('The GroupBy lets you summarise your data by specific categories and groups.')
    with st.expander('GroupBy your Columns'):
        col1, col2, col3 = st.columns(3)
        with col1:
            groupby_cols = st.multiselect("Choose Your Columns to GroupBy", options=list(data.columns))
        with col2:
            operation_col = st.selectbox("Choose Column for Operation", options=list(data.columns))
        with col3:
            operation = st.selectbox('Choose Operation To Perform',
                options=['sum', 'mean', 'max', 'min', 'count', 'median', 'std', 'var'])

    # ✅ FIX 5: GroupBy result block also inside the file check
    if groupby_cols:
        result = data.groupby(groupby_cols).agg(
            newcol=(operation_col, operation)
        ).reset_index()
        st.dataframe(result)

        # ✅ BONUS: Added GroupBy visualization
        st.subheader("GroupBy Visualization", divider='blue')
        fig = px.bar(result, x=groupby_cols[0], y='newcol', title=f'{operation} of {operation_col}')
        st.plotly_chart(fig)