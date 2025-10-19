import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# Load model + target encoder
model = joblib.load('models/bird_conservation_model_xgb.pkl')
target_encoder = joblib.load('models/target_encoders.pkl')  # LabelEncoder trained on ['Low', 'Medium', 'High']

# Load dataset
birds_df = pd.read_excel("data/my_data.xlsx", engine='openpyxl')

st.set_page_config(page_title="Bird Conservation Predictor", layout="wide")
st.title("🦜 Bird Conservation Concern Predictor")

# -------- SIDEBAR input form --------
st.sidebar.header("Input Bird Traits")

group = st.sidebar.multiselect("Group", birds_df['group'].unique())
iucn = st.sidebar.selectbox("IUCN Status", birds_df['iucn_status'].unique())
wlpa = st.sidebar.selectbox("WLPA Schedule", birds_df['wlpa_schedule'].unique())

anal_long = st.sidebar.number_input("Analysed Long-Term", min_value=0, max_value=1000, value=0)
anal_curr = st.sidebar.number_input("Analysed Current", min_value=0, max_value=1000, value=0)

long_trend = st.sidebar.slider("Long-Term Trend (%)", -100.0, 200.0, 0.0)
curr_change = st.sidebar.slider("Current Annual Change (%)", -50.0, 200.0, 0.0)

long_term_status = st.sidebar.selectbox("Long-Term Status", birds_df['long_term_status'].unique())
current_status = st.sidebar.selectbox("Current Status", birds_df['current_status'].unique())
distribution_status = st.sidebar.selectbox("Distribution Status", birds_df['distribution_status'].unique())

migration = st.sidebar.multiselect("Migratory Status", birds_df['migratory_status'].unique())
diet = st.sidebar.selectbox("Diet", birds_df['diet'].unique())
habitat = st.sidebar.selectbox("Habitat Type", birds_df['habitat_type'].unique())
endemic = st.sidebar.selectbox("Endemicity Type", birds_df['endemicity_type'].unique())
bird_type = st.sidebar.selectbox("Bird Type", birds_df['bird_type'].unique())

# Input dict
input_dict = {
    "group": ",".join(group) if isinstance(group, list) else group,
    "migratory_status": ",".join(migration) if isinstance(migration, list) else migration,
    "diet": diet,
    "habitat_type": habitat,
    "wlpa_schedule": wlpa,
    "iucn_status": iucn,
    "analysed_long_term": anal_long,
    "analysed_current": anal_curr,
    "long_term_trend": long_trend,
    "current_annual_change": curr_change,
    "long_term_status": long_term_status,
    "current_status": current_status,
    "distribution_status": distribution_status,
    "endemicity_type": endemic,
    "bird_type": bird_type
}

# -------------- Predict --------------
if st.button("Predict Conservation Concern"):
    input_df = pd.DataFrame([input_dict])
    pred_encoded = model.predict(input_df)[0]
    pred_label = target_encoder.inverse_transform([pred_encoded])[0]
    st.success(f"Predicted Conservation Concern: *{pred_label}*")


# -------------- EDA PLOTS --------------
st.header("📊 Exploratory Data Analysis (EDA)")

# Apply filters
df = birds_df.copy()
if group:
    df = df[df['group'].isin(group)]
if migration:
    df = df[df['migratory_status'].isin(migration)]

# Common color palette
pastel_colors = px.colors.qualitative.Pastel

# ----------- ROW 1 -----------
col1, col2 = st.columns(2)
with col1:
    st.subheader("IUCN Status Distribution")
    fig = px.pie(df, names='iucn_status', color='iucn_status',
                 color_discrete_sequence=pastel_colors, hole=0.3)
    fig.update_traces(textinfo='percent+label',
                      marker=dict(line=dict(color='black', width=1)))
    fig.update_layout(
        legend=dict(font=dict(size=14))
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("WLPA Schedule Distribution")
    tab = df['wlpa_schedule'].value_counts().reset_index()
    tab.columns = ['wlpa_schedule', 'count']
    fig = px.bar(tab, x='wlpa_schedule', y='count', text='count',
                 color='wlpa_schedule', color_discrete_sequence=pastel_colors)
    fig.update_traces(marker=dict(line=dict(color='black', width=1)), textposition="outside")
    fig.update_layout(
        xaxis_title="WLPA Schedule", yaxis_title="Count",
        xaxis=dict(title=dict(font=dict(size=14)), tickfont=dict(size=14)),
        yaxis=dict(title=dict(font=dict(size=14)), tickfont=dict(size=14)),
        legend=dict(font=dict(size=14))
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------- ROW 2 -----------
col3, col4 = st.columns(2)
with col3:
    st.subheader("Current Status (Stable/Declining/Increasing)")
    tab = df['current_status'].value_counts().reset_index()
    tab.columns = ['current_status', 'count']
    fig = px.bar(tab, x='current_status', y='count', text='count',
                 color='current_status', color_discrete_sequence=pastel_colors)
    fig.update_traces(marker=dict(line=dict(color='black', width=1)), textposition="outside")
    fig.update_layout(
        xaxis_title="Current Status", yaxis_title="Count",
        xaxis=dict(title=dict(font=dict(size=14)), tickfont=dict(size=14)),
        yaxis=dict(title=dict(font=dict(size=14)), tickfont=dict(size=14)),
        legend=dict(font=dict(size=14))
    )
    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.subheader("Migratory Status Distribution")
    fig = px.pie(df, names='migratory_status', color='migratory_status',
                 color_discrete_sequence=pastel_colors)
    fig.update_traces(textinfo='percent+label',
                      marker=dict(line=dict(color='black', width=1)))
    fig.update_layout(
        legend=dict(font=dict(size=14))
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------- ROW 3 -----------
col5, col6 = st.columns(2)
with col5:
    st.subheader("Habitat Type Distribution")
    tab = df['habitat_type'].value_counts().reset_index()
    tab.columns = ['habitat_type', 'count']
    fig = px.bar(tab, x='habitat_type', y='count', text='count',
                 color='habitat_type', color_discrete_sequence=pastel_colors)
    fig.update_traces(marker=dict(line=dict(color='black', width=1)), textposition="outside")
    fig.update_layout(
        xaxis_title="Habitat Type", yaxis_title="Count",
        xaxis=dict(title=dict(font=dict(size=14)), tickfont=dict(size=14)),
        yaxis=dict(title=dict(font=dict(size=14)), tickfont=dict(size=14)),
        legend=dict(font=dict(size=14))
    )
    st.plotly_chart(fig, use_container_width=True)

with col6:
    st.subheader("Diet Type Distribution")
    tab = df['diet'].value_counts().reset_index()
    tab.columns = ['diet', 'count']
    fig = px.bar(tab, x='diet', y='count', text='count',
                 color='diet', color_discrete_sequence=pastel_colors)
    fig.update_traces(marker=dict(line=dict(color='black', width=1)), textposition="outside")
    fig.update_layout(
        xaxis_title="Diet", yaxis_title="Count",
        xaxis=dict(title=dict(font=dict(size=14)), tickfont=dict(size=14)),
        yaxis=dict(title=dict(font=dict(size=14)), tickfont=dict(size=14)),
        legend=dict(font=dict(size=14))
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------- ROW 4 -----------
col7, col8 = st.columns(2)
with col7:
    st.subheader("Long-term Trend vs Current Annual Change")
    fig = px.scatter(df, x='long_term_trend', y='current_annual_change',
                     color='long_term_status',
                     color_discrete_sequence=pastel_colors,
                     size_max=15, opacity=0.7)
    fig.update_traces(marker=dict(line=dict(color='black', width=0.5)))
    fig.update_layout(
        xaxis_title="Long-term Trend (%)",
        yaxis_title="Current Annual Change (%)",
        xaxis=dict(title=dict(font=dict(size=14)), tickfont=dict(size=14)),
        yaxis=dict(title=dict(font=dict(size=14)), tickfont=dict(size=14)),
        legend=dict(font=dict(size=14))
    )
    st.plotly_chart(fig, use_container_width=True)

with col8:
    st.subheader("Endemic vs Non-endemic Species")
    fig = px.pie(df, names='endemicity_type', color='endemicity_type',
                 color_discrete_sequence=pastel_colors)
    fig.update_traces(textinfo='percent+label',
                      marker=dict(line=dict(color='black', width=1)))
    fig.update_layout(
        legend=dict(font=dict(size=14))
    )
    st.plotly_chart(fig, use_container_width=True)

