import pandas as pd
import matplotlib.pyplot as plt 
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Consoleflare Analytics Portal",page_icon='📊')
st.title(":rainbow[Data Analaytics Portal]")
st.subheader(':grey[Explore Data With Ease]',divider='blue')


file = st.file_uploader("Drop csv or excel file here",type=['csv','xlsx'])
if(file!=None):
    if file.name.endswith('.csv'):
        data = pd.read_csv(file)
    else:
        data = pd.read_excel(file)

    st.dataframe(data)
    st.info('File is successfully uploaded!')

    st.subheader(':rainbow[Basic Information of the dataset]',divider='blue')
    tab1,tab2,tab3,tab4=st.tabs(['Summary','Top Rows and Bottom Rows','Data Types','Columns' ])

with tab1:
    st.write(f'There are {data.shape[0]} rows in dataset and {data.shape[1]} columns in dataset')
    st.subheader(':grey[Statstical Summary of the dataset]')
    st.dataframe(data.describe())

with tab2:
    st.subheader(':grey[Top Rows]')
    TopRows = st.slider('Select number of top rows to display',1,data.shape[0])
    st.dataframe(data.head(TopRows))
    st.subheader(':grey[Bottom Rows]')
    BottomRows = st.slider('Select number of bottom rows to display',1,data.shape[0])
    st.dataframe(data.tail(BottomRows))
    
with tab3:
    st.subheader(':grey[Data Types of the dataset]')
    st.dataframe(data.dtypes)

with tab4:
    st.subheader(':grey[Columns of the dataset]')
    st.write(data.columns)


st.subheader(':rainbow[Column Values To Count]',divider='blue')
with st.expander('Value Count'):
    col1,col2=st.columns(2)
    with col1:
        column = st.selectbox('Choose Column Name',options=list(data.columns))
    with col2:
        TopRows = st.number_input("Top Rows",1,step=1)

count = st.button('Count')
if (count==True):
    result = data[column].value_counts().reset_index().head(TopRows)
    st.dataframe(result)
    st.subheader("Visualization",divider='blue')
    fig = px.bar(data_frame=result,x=column,y='count')
    st.plotly_chart(fig)
    fig = px.line(data_frame=result,x=column,y='count')
    st.plotly_chart(fig)


st.subheader(':rainbow[Groupby:Simplyfy Your Data Analysis]',divider='blue')
st.write('The GroupBy lets you summarise your data by specefic categories and groups')
with st.expander('GroupBy your Columns'):
    col1,col2,col3 = st.columns(3)
    with col1:
        groupby_cols= st.multiselect("Choose Your Columns to GroupBy",options=list(data.columns))
    with col2:
        operation_col = st.selectbox("Choose Column for Operation",options=list(data.columns))
    with col3:
        operattion = st.selectbox('Choose Operation To Perform',options=['sum','mean','max','min','count','median','std','var'  ])
if(groupby_cols):
    result = data.groupby(groupby_cols).agg(
        newcol=(operation_col,operattion)
    )
    result = result.reset_index()
    result  