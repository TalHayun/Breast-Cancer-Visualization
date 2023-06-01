import pandas as pd
import streamlit as st
import altair as alt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

image = Image.open('dataset-cover.jpg')
df = pd.read_csv('Breast_Cancer.csv')

# Define the bin edges
bin_edges = [30, 40, 50, 60, 70]

# Define the corresponding bin labels
bin_labels = ['30-39', '40-49', '50-59', '60-69']

# Convert numeric 'Age' to categorical using bins and labels
df['Age'] = pd.cut(df['Age'], bins=bin_edges, labels=bin_labels, right=False)


agg_df = df.groupby('Race').agg(malignancy_rate=('A Stage', lambda x: sum(x == 'Distant') / len(x) * 100),
                                    avg_tumor_size=('Tumor Size', 'mean')).reset_index()
grouped_df = agg_df.sort_values(by=['Race'], key=lambda x: x.map({v: i for i, v in enumerate(['Other', 'White', 'Black'])}))


def get_mortality_rate(feature_name):
    mortality_df = df.groupby(feature_name)['Status'].value_counts().unstack().fillna(0)
    mortality_df['Mortality Rate'] = mortality_df['Dead'] / (mortality_df['Dead'] + mortality_df['Alive'])
    return mortality_df

def build_st_query_for_line_charts(title: str, options: list):
    feature = st.radio(f'Select {title}', options)
    return feature

def build_heatmap():
    st.subheader('Impact of demographic characteristics on the mortality of women with breast cancer in America')

    col1 = st.columns(1)

    with col1[0]:
        options_feature1 = ['Age', 'Race', 'Marital Status']
        feature1 = build_st_query_for_line_charts("main feature", options_feature1)

    mortality_df = get_mortality_rate(feature1).sort_values(by='Mortality Rate')
    bar_fig = go.Figure()

    bar_fig.add_trace(go.Bar(
          x=mortality_df.index,
          y=mortality_df['Mortality Rate'],
          marker=dict(color='salmon')
      ))
    bar_fig.update_layout(
        yaxis=dict(title=dict(text= "Mortality Rate (%)", font=dict(size=20))),
        xaxis=dict(title=dict(text=f'{feature1}', font=dict(size=20))))
    st.plotly_chart(bar_fig)


    col2 = st.columns(1)
    with col2[0]:
        options_feature2 = ['Age', 'Race', 'Marital Status']
        options_feature2.remove(feature1)
        feature2 = build_st_query_for_line_charts("secondary feature", options_feature2)


    # Calculate the mortality rates based on the "Dead" values
    pivot_df = df.pivot_table(index=feature1, columns=feature2, values='Status',
                              aggfunc=lambda x: round(sum(x == 'Dead') / len(x), 2))

    # Create a heatmap using Plotly Express
    fig = px.imshow(pivot_df, text_auto=True, color_continuous_scale='reds', labels=dict(color="Mortality rate (%)"))
    fig.update_xaxes(side="top")
    fig.update_layout(height=600, width=800)
    fig.update_layout(
    yaxis=dict(title=dict(text=f"{feature1}", font=dict(size=24))),
    xaxis=dict(title=dict(text=f"{feature2}", font=dict(size=24))),
    coloraxis_colorbar=dict(title=dict(text='Mortality rate (%)', font=dict(size=22)))
)
    # Display the heatmap in Streamlit
    st.plotly_chart(fig)


def build_two_y_axis_chart():
    st.subheader('Malignancy Rate and Average Tumor Size by Race')
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
        xaxis=dict(title='Race', title_font=dict(size=20)),
        yaxis=dict(title='Malignancy Rate (%)', title_font=dict(size=16)),
        yaxis2=dict(title='Average Tumor Size (mm)', overlaying='y', side='right', title_font=dict(size=16)),
        barmode='group',
        bargap=0.5  # Adjust the spacing between the bars
    )
    st.plotly_chart(fig)

def figure3():
    st.subheader('Women with which characteristics are more likely to have a short recovery from breast cancer?')

    survived = df[df['Status'] == 'Alive']
    survived_avg = survived.groupby(['Marital Status', 'Race', 'Age'])['Survival Months'].mean().reset_index()
    color_scale = px.colors.qualitative.T10

    fig = px.bar(survived_avg, x="Marital Status", y="Survival Months", color="Race",
                 animation_frame="Age", animation_group="Marital Status", facet_col="Race", range_y=[0, 100], color_discrete_sequence=color_scale)
    fig.update_layout(yaxis=dict(title=dict(text="Recovery Time (Months)")),height=600, width=900)
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 1000},  # Frame duration for "Play" button
                                "fromcurrent": True,
                                "transition": {"duration": 500, "easing": "linear"},
                            },
                        ],
                    ),
                    dict(
                        label="Stop",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0},  # Frame duration for "Stop" button
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            ),
        ],
    )
    st.plotly_chart(fig)





st.markdown("""
    <h1 style='text-align: center;'>Visualization Final Project</h1>
    """, unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Breast Cancer</h2>", unsafe_allow_html=True)
st.image(image)
build_heatmap()
build_two_y_axis_chart()
figure3()
