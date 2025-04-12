import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

st.set_page_config(page_title="Analizador Energético", layout="wide")
st.title("🔌 Analizador Energético para Clientes")

# Subida del archivo Excel
uploaded_file = st.file_uploader("📤 Sube tu archivo Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("✅ Archivo cargado correctamente")

    # Vista previa
    st.subheader("📋 Vista previa de datos")
    st.dataframe(df.head())

    try:
        # Procesamiento de columnas
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        df.set_index('Fecha', inplace=True)

        # Tabs para separar secciones
        tab1, tab2, tab3 = st.tabs(["📊 Visualización", "📈 Predicción", "🔄 Simulación de tarifas"])

        with tab1:
            st.subheader("Consumo mensual (kWh)")
            st.line_chart(df['kWh'])

            if 'Coste (€)' in df.columns:
                st.subheader("Coste mensual (€)")
                st.line_chart(df['Coste (€)'])

            if 'Tarifa' in df.columns:
                st.subheader("Distribución por tarifas")
                tarifa_counts = df['Tarifa'].value_counts()
                st.bar_chart(tarifa_counts)

        with tab2:
            df_prophet = df[['kWh']].reset_index()
            df_prophet.columns = ['ds', 'y']
            modelo = Prophet()
            modelo.fit(df_prophet)

            futuro = modelo.make_future_dataframe(periods=12, freq='M')
            forecast = modelo.predict(futuro)

            st.subheader("Predicción de consumo (kWh)")
            st.line_chart(forecast.set_index('ds')[['yhat']])

        with tab3:
            st.subheader("Simulación de cambio de tarifa")
            descuento = st.slider("Descuento simulado en tarifa actual (%)", 0, 30, 10)
            if 'Coste (€)' in df.columns:
                df['Coste simulado (€)'] = df['Coste (€)'] * (1 - descuento / 100)
                st.line_chart(df[['Coste (€)', 'Coste simulado (€)']])
                ahorro_total = df['Coste (€)'].sum() - df['Coste simulado (€)'].sum()
                st.success(f"💰 Ahorro estimado con tarifa simulada: {ahorro_total:.2f} €")
            else:
                st.warning("❗ Para usar esta función, añade la columna 'Coste (€)' en tu Excel.")

    except Exception as e:
        st.error(f"❌ Error procesando los datos: {e}")
