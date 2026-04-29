import pandas as pd
import plotly.express as px
import streamlit as st

# Page config and titles
st.set_page_config(page_title="Consoleflare Analytics Portal", page_icon='📊')
st.title(":rainbow[Data Analytics Portal]")
st.subheader(':grey[Explore Data With Ease]', divider='blue')

# File upload box accepting csv and excel only
file = st.file_uploader("Drop csv or excel file here", type=['csv', 'xlsx'])

# Everything runs only if a file is uploaded
if file is not None:

    # Read file into a DataFrame based on extension
    if file.name.endswith('.csv'):
        data = pd.read_csv(file)
    else:
        data = pd.read_excel(file)

    # Show raw data and confirmation message
    st.dataframe(data)
    st.info('File is successfully uploaded!')

    # Section heading for dataset info
    st.subheader(':rainbow[Basic Information of the dataset]', divider='blue')

    # Four tabs for different data views
    tab1, tab2, tab3, tab4 = st.tabs(['Summary', 'Top & Bottom Rows', 'Data Types', 'Columns'])

    # Tab1: row/column count and statistical summary
    with tab1:
        st.write(f'There are {data.shape[0]} rows and {data.shape[1]} columns in the dataset.')
        st.subheader(':grey[Statistical Summary of the dataset]')
        st.dataframe(data.describe())

    # Tab2: sliders to preview top and bottom rows
    with tab2:
        st.subheader(':grey[Top Rows]')
        top_rows = st.slider('Select number of top rows to display', 1, data.shape[0], key='top_slider')
        st.dataframe(data.head(top_rows))
        st.subheader(':grey[Bottom Rows]')
        bottom_rows = st.slider('Select number of bottom rows to display', 1, data.shape[0], key='bottom_slider')
        st.dataframe(data.tail(bottom_rows))

    # Tab3: data type of each column
    with tab3:
        st.subheader(':grey[Data Types of the dataset]')
        st.dataframe(data.dtypes)

    # Tab4: list of all column names
    with tab4:
        st.subheader(':grey[Columns of the dataset]')
        st.write(list(data.columns))

    # Section for counting unique values in any column
    st.subheader(':rainbow[Column Values To Count]', divider='blue')

    # Expander hides the controls until user opens it
    with st.expander('Value Count'):
        col1, col2 = st.columns(2)
        with col1:
            # Dropdown to pick which column to count
            column = st.selectbox('Choose Column Name', options=list(data.columns))
        with col2:
            # Number input for how many top results to show
            top_n = st.number_input("Top Rows", min_value=1, step=1)

    # Count button triggers the value count and charts
    if st.button('Count'):
        # Count occurrences, reset index, keep top n rows
        result = data[column].value_counts().reset_index().head(top_n)
        # Rename columns to avoid pandas version conflicts
        result.columns = [column, 'count']
        st.dataframe(result)
        st.subheader("Visualization", divider='blue')
        # Bar chart of value counts
        fig = px.bar(data_frame=result, x=column, y='count')
        st.plotly_chart(fig)
        # Line chart of same data
        fig = px.line(data_frame=result, x=column, y='count')
        st.plotly_chart(fig)

    # Section for groupby aggregation analysis
    st.subheader(':rainbow[GroupBy: Simplify Your Data Analysis]', divider='blue')
    st.write('The GroupBy lets you summarise your data by specific categories and groups.')

    with st.expander('GroupBy your Columns'):
        col1, col2, col3 = st.columns(3)
        with col1:
            # Multi-select for groupby columns
            groupby_cols = st.multiselect("Choose Your Columns to GroupBy", options=list(data.columns))
        with col2:
            # Column to apply the math operation on
            operation_col = st.selectbox("Choose Column for Operation", options=list(data.columns))
        with col3:
            # Which aggregation operation to perform
            operation = st.selectbox('Choose Operation To Perform',
                options=['sum', 'mean', 'max', 'min', 'count', 'median', 'std', 'var'])

    # Only runs when user has selected groupby columns
    if groupby_cols:
        # Group data and apply chosen operation
        result = data.groupby(groupby_cols).agg(
            newcol=(operation_col, operation)
        ).reset_index()
        st.dataframe(result)
        st.subheader("GroupBy Visualization", divider='blue')
        # Bar chart of grouped result
        fig = px.bar(result, x=groupby_cols[0], y='newcol', title=f'{operation} of {operation_col}')
        st.plotly_chart(fig)


    # ================================================================
    # DATA CLEANING SECTION
    # ================================================================

    st.subheader(':rainbow[Data Cleaning]', divider='blue')
    st.write('Clean your data by removing duplicates and handling missing values.')

    # session_state keeps cleaned data alive across button clicks
    if 'cleaned_df' not in st.session_state:
        st.session_state.cleaned_df = data.copy()

    # Always work on the session state version
    df = st.session_state.cleaned_df

    # Four metric cards showing current data health
    st.markdown('### 📋 Current Data Health')
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(label='Total Rows',     value=f'{df.shape[0]:,}')
    m2.metric(label='Total Columns',  value=f'{df.shape[1]}')
    # isnull creates True/False table, sum twice gives total empty cells
    m3.metric(label='Missing Cells',  value=f'{df.isnull().sum().sum():,}')
    # duplicated marks identical rows as True, sum counts them
    m4.metric(label='Duplicate Rows', value=f'{df.duplicated().sum():,}')

    # Build summary table of missing values per column
    missing_df = pd.DataFrame({
        'Column'        : df.columns,
        'Missing Count' : df.isnull().sum().values,
        # Percentage of missing values relative to total rows
        'Missing %'     : (df.isnull().sum().values / len(df) * 100).round(2)
    })

    # Keep only columns that actually have missing values
    missing_df = missing_df[missing_df['Missing Count'] > 0]

    # Show success if clean or show table and chart if dirty
    if len(missing_df) == 0:
        st.success('No missing values found in your dataset!')
    else:
        st.dataframe(missing_df, use_container_width=True)
        # Red bar chart showing missing percentage per column
        fig_missing = px.bar(
            missing_df,
            x='Column',
            y='Missing %',
            title='Missing Values % by Column',
            text='Missing %',
            color='Missing %',
            color_continuous_scale='Reds'
        )
        # Format bar labels to one decimal place with percent sign
        fig_missing.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig_missing, use_container_width=True)


    # ── REMOVE DUPLICATES ────────────────────────────────────────
    st.markdown('### 🔁 Remove Duplicate Rows')

    with st.expander('Handle Duplicates'):
        dup_count = df.duplicated().sum()

        if dup_count == 0:
            st.success('No duplicate rows found!')
        else:
            st.warning(f'Found {dup_count} duplicate rows in your dataset')
            # Show preview of duplicate rows only
            st.dataframe(df[df.duplicated()].head(10), use_container_width=True)

            if st.button('Delete All Duplicate Rows', key='del_dupes'):
                # drop_duplicates keeps first occurrence removes rest
                df = df.drop_duplicates()
                st.session_state.cleaned_df = df
                st.success(f'Successfully removed {dup_count} duplicate rows')
                st.dataframe(df.head(10), use_container_width=True)


    # ── MISSING VALUES FIXER ─────────────────────────────────────
    st.markdown('### 🩹 Fix Missing Values')

    # Guide to help user pick the right strategy
    st.info('''
    HOW TO CHOOSE YOUR STRATEGY:
    → Missing % less than 5%     →  Drop rows
    → Numeric, no big outliers   →  Fill with Mean   (age, height)
    → Numeric, has big outliers  →  Fill with Median (salary, price)
    → Text or category column    →  Fill with Mode   (city, gender)
    → You know the exact value   →  Fill with Custom (0, Unknown)
    ''')

    with st.expander('Handle Missing Values'):

        if df.isnull().sum().sum() == 0:
            st.success('No missing values to fix!')
        else:
            # Only columns that have nulls appear as options
            col_with_nulls = missing_df['Column'].tolist()

            fix_col1, fix_col2 = st.columns(2)
            with fix_col1:
                # User picks one column or all columns at once
                selected_col = st.selectbox(
                    'Select Column to Fix',
                    options=['All Columns'] + col_with_nulls
                )
            with fix_col2:
                fill_strategy = st.selectbox(
                    'Choose Fill Strategy',
                    options=[
                        'Drop rows with nulls',
                        'Fill with Mean',
                        'Fill with Median',
                        'Fill with Mode',
                        'Fill with Custom Value'
                    ]
                )

            # Smart recommendation shown based on column properties
            if selected_col != 'All Columns':
                col_dtype = df[selected_col].dtype
                col_missing_pct = df[selected_col].isnull().sum() / len(df) * 100

                if col_missing_pct < 5:
                    st.info(f'{selected_col} has only {col_missing_pct:.1f}% missing. Safe to drop rows.')
                elif str(col_dtype) == 'object':
                    st.info(f'{selected_col} is a text column. Mode is recommended.')
                else:
                    # Skewness above 1 or below -1 means data is lopsided
                    skew = df[selected_col].skew()
                    if abs(skew) > 1:
                        st.info(f'{selected_col} is skewed ({skew:.2f}). Median recommended.')
                    else:
                        st.info(f'{selected_col} is balanced ({skew:.2f} skew). Mean recommended.')

            # Custom value input only appears when that strategy is chosen
            custom_val = None
            if fill_strategy == 'Fill with Custom Value':
                custom_val = st.text_input('Enter custom fill value', placeholder='e.g. 0 or Unknown')

            if st.button('Apply Fix', key='fix_nulls'):
                before_rows = len(df)

                # ── FIXING ALL COLUMNS AT ONCE ────────────────
                if selected_col == 'All Columns':

                    if fill_strategy == 'Drop rows with nulls':
                        # Removes every row that has any empty cell
                        df = df.dropna()
                        st.success(f'Dropped {before_rows - len(df)} rows')

                    elif fill_strategy == 'Fill with Mean':
                        # Mean only works on numeric columns
                        num_cols = df.select_dtypes(include='number').columns
                        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
                        st.success('Filled numeric nulls with column mean')

                    elif fill_strategy == 'Fill with Median':
                        # Median is middle value, unaffected by outliers
                        num_cols = df.select_dtypes(include='number').columns
                        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
                        st.success('Filled numeric nulls with column median')

                    elif fill_strategy == 'Fill with Mode':
                        # Mode is most frequent value, works for text too
                        for col in df.columns:
                            if df[col].isnull().sum() > 0:
                                df[col] = df[col].fillna(df[col].mode()[0])
                        st.success('Filled all nulls with column mode')

                    elif fill_strategy == 'Fill with Custom Value' and custom_val:
                        df = df.fillna(custom_val)
                        st.success(f'Filled all nulls with: {custom_val}')

                # ── FIXING ONE COLUMN ONLY ────────────────────
                else:

                    if fill_strategy == 'Drop rows with nulls':
                        # subset limits dropna to only this column
                        df = df.dropna(subset=[selected_col])
                        st.success(f'Dropped {before_rows - len(df)} rows')

                    elif fill_strategy == 'Fill with Mean':
                        mean_val = df[selected_col].mean()
                        df[selected_col] = df[selected_col].fillna(mean_val)
                        st.success(f'Filled with mean: {mean_val:.2f}')

                    elif fill_strategy == 'Fill with Median':
                        median_val = df[selected_col].median()
                        df[selected_col] = df[selected_col].fillna(median_val)
                        st.success(f'Filled with median: {median_val:.2f}')

                    elif fill_strategy == 'Fill with Mode':
                        mode_val = df[selected_col].mode()[0]
                        df[selected_col] = df[selected_col].fillna(mode_val)
                        st.success(f'Filled with mode: {mode_val}')

                    elif fill_strategy == 'Fill with Custom Value' and custom_val:
                        df[selected_col] = df[selected_col].fillna(custom_val)
                        st.success(f'Filled with: {custom_val}')

                # Save updated df back to session state after every fix
                st.session_state.cleaned_df = df
                st.dataframe(df.head(10), use_container_width=True)


    # ── DROP UNWANTED COLUMNS ─────────────────────────────────────
    st.markdown('### ❌ Drop Unwanted Columns')

    with st.expander('Remove Columns'):
        # Multi-select lets user pick multiple columns to remove
        cols_to_drop = st.multiselect(
            'Select columns to remove',
            options=df.columns.tolist()
        )

        if st.button('Drop Selected Columns', key='drop_cols'):
            if cols_to_drop:
                # drop removes the selected columns from the DataFrame
                df = df.drop(columns=cols_to_drop)
                st.session_state.cleaned_df = df
                st.success(f'Removed columns: {cols_to_drop}')
                st.dataframe(df.head(10), use_container_width=True)
            else:
                st.warning('Please select at least one column first')


    # ── FINAL SUMMARY AFTER ALL CLEANING ─────────────────────────
    st.markdown('### ✅ Final Cleaned Data Summary')

    f1, f2, f3, f4 = st.columns(4)
    # delta shows change from original as green or red number
    f1.metric('Remaining Rows',    f'{df.shape[0]:,}',  delta=f'{df.shape[0] - data.shape[0]:,}')
    f2.metric('Remaining Columns', f'{df.shape[1]}',    delta=f'{df.shape[1] - data.shape[1]}')
    f3.metric('Missing Values',    f'{df.isnull().sum().sum():,}')
    f4.metric('Duplicates Left',   f'{df.duplicated().sum():,}')

    # Show full cleaned dataframe
    st.dataframe(df, use_container_width=True)


    # ── DOWNLOAD CLEANED DATA ─────────────────────────────────────
    st.markdown('### 💾 Download Your Cleaned Data')

    # Convert DataFrame to csv bytes for download
    cleaned_csv = df.to_csv(index=False).encode('utf-8')

    # Download button triggers file save in browser
    st.download_button(
        label='⬇️ Download Cleaned CSV',
        data=cleaned_csv,
        file_name='cleaned_data.csv',
        mime='text/csv',
        key='download_csv'
    )