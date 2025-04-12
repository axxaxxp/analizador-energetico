import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

st.title("Analizador Energético")

uploaded_file = st.file_uploader("Sube tu archivo Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("Archivo cargado correctamente")
    st.dataframe(df.head())

    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df.set_index('Fecha', inplace=True)

    st.subheader("Consumo mensual (kWh)")
    st.line_chart(df['kWh'])

    df_prophet = df[['kWh']].reset_index()
    df_prophet.columns = ['ds', 'y']

    modelo = Prophet()
    modelo.fit(df_prophet)

    futuro = modelo.make_future_dataframe(periods=12, freq='M')
    forecast = modelo.predict(futuro)

    st.subheader("Predicción futura (kWh)")
    st.line_chart(forecast.set_index('ds')[['yhat']])
