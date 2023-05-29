import pandas as pd
import streamlit as st
import altair as alt
import seaborn as sns
import plotly.express as px


df = pd.read_csv('Breast_Cancer.csv')

# Define the age ranges
bins = [0, 9, 19, 29, 39, 49, 59, 69]

# Define the corresponding labels for each range
labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69']

# Transform the 'Age' column into categorical ranges
df['Age'] = pd.cut(df['Age'], bins=bins, labels=labels)

def build_st_query_for_line_charts(title: str, options: list):
    feature = st.radio(f"Select {title}", options)
    return feature

def build_heatmap():
    st.subheader('Impact of demographic characteristics on the mortality of women with breast cancer in America')

    col1, col2 = st.columns(2)

    with col1:
        options_feature1 = ['Age', 'Race', 'Marital Status']
        feature1 = build_st_query_for_line_charts("first feature", options_feature1)

    with col2:
        options_feature2 = ['Age', 'Race', 'Marital Status']
        options_feature2.remove(feature1)
        feature2 = build_st_query_for_line_charts("second feature", options_feature2)


    # Calculate the mortality rates based on the "Dead" values
    pivot_df = df.pivot_table(index=feature1, columns=feature2, values='Status',
                              aggfunc=lambda x: sum(x == 'Dead') / len(x))

    # Create a heatmap using Plotly Express
    fig = px.imshow(pivot_df, color_continuous_scale='reds', labels=dict(color="Mortality rate (%)"))
    fig.update_xaxes(side="top")
    fig.update_layout(height=600, width=800)
    
    # Display the heatmap in Streamlit
    st.plotly_chart(fig)



st.title('Visualization final project')
build_heatmap()
