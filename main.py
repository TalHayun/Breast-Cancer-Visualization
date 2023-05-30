import pandas as pd
import streamlit as st
import altair as alt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


df = pd.read_csv('Breast_Cancer.csv')

# Define the age ranges
bins = [0, 9, 19, 29, 39, 49, 59, 69]

# Define the corresponding labels for each range
labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69']

# Transform the 'Age' column into categorical ranges
df['Age'] = pd.cut(df['Age'], bins=bins, labels=labels)

agg_df = df.groupby('Race').agg(malignancy_rate=('A Stage', lambda x: sum(x == 'Distant') / len(x) * 100),
                                    avg_tumor_size=('Tumor Size', 'mean')).reset_index()
grouped_df = agg_df.sort_values(by=['Race'], key=lambda x: x.map({v: i for i, v in enumerate(['Other', 'White', 'Black'])}))

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
    
    annotations = []
    for i, row in enumerate(pivot_df.values):
        for j, value in enumerate(row):
            annotations.append(dict(x=j, y=i, text=str(round(value, 2))+'%', showarrow=False, font=dict(color="black")))
    fig.update_layout(annotations=annotations)

    fig.update_xaxes(side="top")
    fig.update_layout(height=600, width=800)
    fig.update_xaxes(side="top")
    fig.update_layout(height=600, width=800)
    
    # Display the heatmap in Streamlit
    st.plotly_chart(fig)
    
def figure2():
  
  fig = go.Figure()

  fig.add_trace(go.Bar(
      x=grouped_df['Race'],
      y=grouped_df['malignancy_rate'],
      name='Malignancy Rate',
      yaxis='y',
      offsetgroup=0,
      width=0.25,
      marker=dict(color='salmon')
  ))

  fig.add_trace(go.Bar(
      x=grouped_df['Race'],
      y=grouped_df['avg_tumor_size'],
      name='Average Tumor Size',
      yaxis='y2',
      offsetgroup=1,
      width=0.25,
      marker=dict(color='lightseagreen')
  ))

  fig.update_layout(
      title=dict(text='Malignancy Rate and Average Tumor Size by Race', x=0.5),
      xaxis=dict(title='Race',  title_font=dict(size=21)),
      yaxis=dict(title='Malignancy Rate (%)'),
      yaxis2=dict(title='Average Tumor Size (mm)', overlaying='y', side='right'),
      barmode='group',
      bargap=0.5  # Adjust the spacing between the bars
  )
  st.plotly_chart(fig)


st.title('Visualization final project')
build_heatmap()
figure2()
