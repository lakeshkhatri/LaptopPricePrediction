import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
data = pickle.load(open('data.pkl', 'rb'))

st.title("💻 Laptop Price Prediction App")

# Brand
manufacturer = st.selectbox('Brand', data['Manufacturer'].unique())

# Type
category = st.selectbox('Type', data['Category'].unique())

# RAM
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 32, 64])

# Weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS Display', ['No', 'Yes'])

# Screen size
screen_size = st.number_input('Screen Size (inches)')

# Resolution
resolution = st.selectbox('Resolution', [
    '1920 x 1080', '1366 x 768', '1600 x 900', '2304 x 1440',
    '2560 x 1440', '2560 x 1600', '2880 x 1800', '3000 x 2000',
    '3200 x 1800', '3840 x 2160'
])

# CPU
cpu = st.selectbox('CPU', data['CPU brand'].unique())

# HDD
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])

# SSD
ssd = st.selectbox('SSD (in GB)', [0, 128, 256, 512, 1024])

# GPU Brand
gpu = st.selectbox('GPU', data['GPU Brand'].unique())

# OS
os = st.selectbox('Operating System', data['os'].unique())

# Predict
if st.button('Predict Price'):
    # Convert touchscreen and ips to binary
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Calculate PPI
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    if screen_size == 0:
        st.error("Screen size cannot be zero.")
    else:
        ppi = ((X_res ** 2 + Y_res ** 2) ** 0.5) / screen_size

    #ppi = ((X_res**2 + Y_res**2) ** 0.5) / screen_size

    # Create DataFrame (so ColumnTransformer works properly)
    columns_order = ['Manufacturer', 'Category', 'RAM', 'Weight',
                     'Touchscreen', 'Ips', 'ppi', 'CPU brand',
                     'HDD', 'SSD', 'GPU Brand', 'os']

    input_df = pd.DataFrame([[manufacturer, category, ram, weight,
                              touchscreen, ips, ppi, cpu, hdd, ssd,
                              gpu, os]], columns=columns_order)

    # Make prediction
    prediction = pipe.predict(input_df)[0]
    st.title(f" Predicted Laptop Price: sek {int(np.exp(prediction))}")
