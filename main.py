import pandas as pd
import streamlit as st
import altair as alt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from matplotlib import cm
import numpy as np

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

def build_st_query_for_ridge_charts(title: str, options: list):
    st.write(f"#### {title}")
    checkbox_states = {}
    # Add "Select All" checkbox
    if title == 'Age':
        select_all = st.checkbox("All Ages")
    elif title == 'Race':
        select_all = st.checkbox("All Races")
    else:
        select_all = st.checkbox("All Marital Statuses")
    checkbox_states = {}  # Dictionary to store checkbox states

    if select_all:
        for option in options:
            checkbox_key = f"{option} ({title})"
            checkbox_states[option] = st.checkbox(option, key=checkbox_key, value=True)
    else:
        for option in options:
            checkbox_key = f"{option} ({title})"
            checkbox_states[option] = st.checkbox(option, key=checkbox_key, value=checkbox_states.get(option, False))

    return checkbox_states

def create_virdis(num):
    viridis = cm.get_cmap('viridis', 12)
    virdis_list = viridis(np.linspace(0, 1, num))

    modified_array = []
    for lst in virdis_list:
        modified_list = lst[:-1]  # Remove the last element from the list
        rgb_string = f'rgb({modified_list[0]}, {modified_list[1]}, {modified_list[2]})'
        modified_array.append(rgb_string)

    return modified_array


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

    bar_fig.update_layout(bargap = 0.8)
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

# def figure3():
#     st.subheader('Women with which characteristics are more likely to have a short recovery from breast cancer?')
#
#     survived = df[df['Status'] == 'Alive']
#     survived_avg = survived.groupby(['Marital Status', 'Race', 'Age'])['Survival Months'].mean().reset_index()
#     color_scale = px.colors.qualitative.T10
#
#     fig = px.bar(survived_avg, x="Marital Status", y="Survival Months", color="Race",
#                  animation_frame="Age", animation_group="Marital Status", facet_col="Race", range_y=[0, 100], color_discrete_sequence=color_scale)
#     fig.update_layout(yaxis=dict(title=dict(text="Recovery Time (Months)")),height=600, width=900)
#     fig.update_layout(
#         updatemenus=[
#             dict(
#                 type="buttons",
#                 buttons=[
#                     dict(
#                         label="Play",
#                         method="animate",
#                         args=[
#                             None,
#                             {
#                                 "frame": {"duration": 1000},  # Frame duration for "Play" button
#                                 "fromcurrent": True,
#                                 "transition": {"duration": 500, "easing": "linear"},
#                             },
#                         ],
#                     ),
#                     dict(
#                         label="Stop",
#                         method="animate",
#                         args=[
#                             [None],
#                             {
#                                 "frame": {"duration": 0},  # Frame duration for "Stop" button
#                                 "mode": "immediate",
#                                 "transition": {"duration": 0},
#                             },
#                         ],
#                     ),
#                 ],
#             ),
#         ],
#     )
#     st.plotly_chart(fig)

def figure3():
    st.subheader('Women with which characteristics are more likely to have a short recovery from breast cancer?')
    st.markdown('### Select Characteristics')
    col1, col2, col3 = st.columns(3)


    with col1:
        age_dict = build_st_query_for_ridge_charts(
            "Age", ['30-39', '40-49', '50-59', '60-69']
        )

    with col2:
        race_dict = build_st_query_for_ridge_charts(
            "Race", ['White', 'Black', 'Other']
        )

    with col3:
        marital_dict = build_st_query_for_ridge_charts(
            "Marital Status", ['Married', 'Divorced', 'Single ', 'Widowed', 'Separated']
        )

    fig = go.Figure()

    grouped = df.groupby(['Age', 'Race', 'Marital Status']).size().reset_index(name='count')
    filtered_groups = grouped[grouped['count'] >= 2]
    num_of_colors = len(filtered_groups)
    colors = create_virdis(num_of_colors)
    i = 0

    survived = df[df['Status'] == 'Alive']
    grouped = survived.groupby(['Age', 'Race', 'Marital Status'])['Survival Months'].mean().reset_index()

    # Sort the groups by the mean survival months in descending order
    sorted_groups = grouped.sort_values('Survival Months', ascending=False)

    # Iterate through each group
    for _, row in sorted_groups.iterrows():
        age = row['Age']
        race = row['Race']
        marital_status = row['Marital Status']
        values = survived[
            (survived['Age'] == age) & (survived['Race'] == race) & (survived['Marital Status'] == marital_status)][
            'Survival Months']
        if len(values) > 1:
            fig.add_trace(go.Violin(x=values, line_color=colors[i], name=f'{age}, {race}, {marital_status}',
                                    meanline_visible=True))
            i += 1

    fig.update_traces(orientation='h', side='positive', width=5, points=False, hoverinfo='skip')
    fig.update_layout(legend=dict(traceorder='reversed', itemsizing='constant'))
    fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False, xaxis_title='Survival Months')
    fig.update_layout(violinmode='group', width=800, height=1000, xaxis_range=[0, 145])
    fig.update_layout(yaxis=dict(showticklabels=False))  # Remove y-axis tick labels
    st.plotly_chart(fig)

st.markdown("""
    <h1 style='text-align: center;'>Visualization Final Project</h1>
    """, unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Breast Cancer</h2>", unsafe_allow_html=True)
st.image(image)
build_heatmap()
build_two_y_axis_chart()
figure3()
