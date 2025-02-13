import streamlit as st
import importlib_metadata as metadata
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium
from streamlit_folium import folium_static
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# Configurar estilo de gráficos
sns.set_style("whitegrid")
plt.rcParams.update({
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.figsize": (10, 6)
})

# =============================================================================
# Funciones de ayuda para leer y procesar datos desde Excel
# =============================================================================
def leer_y_procesar_excel(file_path, hojas):
    sheets_dict = pd.read_excel(file_path, sheet_name=hojas, engine='openpyxl')
    dfs = {}
    for sheet_name, df in sheets_dict.items():
        df = df.dropna(how='all').ffill().fillna(0)
        print(f"Columnas en la hoja '{sheet_name}': {df.columns.tolist()}")
        if len(df.columns) >= 8:
            df.columns = ['Etiqueta', '2018', '2019', '2020', '2021', '2022', '2023']
        dfs[sheet_name] = df
    return dfs

def extraer_datos_generales(df):
    df = df.copy()
    df.set_index('Etiqueta', inplace=True)
    fd = df.loc['FD'].values
    pe = df.loc['PE'].values
    otros = df.loc['Otros'].values
    gas_vehiculado = df.loc['GasVehiculado'].values
    pto_suministro = df.loc['Pto. Suministro'].values
    caudal = df.loc['Caudal'].values
    anios = df.columns.astype(int)
    return fd, pe, otros, gas_vehiculado, pto_suministro, caudal, anios

def extraer_datos_ips(df):
    df = df.copy()
    df.set_index('Etiqueta', inplace=True)
    
    # Verificar si las columnas existen
    if 'Caudal IP' not in df.columns or 'Nº IP' not in df.columns:
        st.error("Las columnas 'Caudal IP' o 'Nº IP' no se encuentran en la hoja 'IPs'. Verifica el archivo Excel.")
        return None, None
    
    caudal_ips = df.loc['Caudal IP'].values
    numero_ips = df.loc['Nº IP'].values
    return caudal_ips, numero_ips

def extraer_datos_aviso(df):
    df = df.copy()
    df.set_index('Etiqueta', inplace=True)
    
    # Verificar si las columnas existen
    required_columns = ['Caudal AVISOS', 'Nº AVISOS', 'MOP ≤ 0,05', '0,05<MOP ≤ 0,15', '0,15 < MOP ≤ 5', 'MOP ≤ 16']
    for col in required_columns:
        if col not in df.columns:
            st.error(f"La columna '{col}' no se encuentra en la hoja 'Aviso'. Verifica el archivo Excel.")
            return None, None, None, None, None, None, None
    
    caudal_aviso = df.loc['Caudal AVISOS'].values
    numero_aviso = df.loc['Nº AVISOS'].values.astype(float)
    mop_005 = df.loc['MOP ≤ 0,05'].values.astype(float)
    mop_005_015 = df.loc['0,05<MOP ≤ 0,15'].values.astype(float)
    mop_015_5 = df.loc['0,15 < MOP ≤ 5'].values.astype(float)
    mop_16 = df.loc['MOP ≤ 16'].values.astype(float)
    suma_aviso = mop_005 + mop_005_015 + mop_015_5 + mop_16
    mop_over16 = np.where(numero_aviso - suma_aviso > 0, numero_aviso - suma_aviso, 0)
    return caudal_aviso, numero_aviso, mop_005, mop_005_015, mop_015_5, mop_16, mop_over16

def extraer_datos_reseguimiento(df):
    df = df.copy()
    df.set_index('Etiqueta', inplace=True)
    
    # Verificar si las columnas existen
    required_columns = ['Caudal RESEGUIMIENTO', 'Nº RESEGUIMIENTO', '1 año 0,05', '2 años 0,05', '1 año 5', '2 años 5', '2 años 16']
    for col in required_columns:
        if col not in df.columns:
            st.error(f"La columna '{col}' no se encuentra en la hoja 'Reseguimiento'. Verifica el archivo Excel.")
            return None, None, None, None, None, None
    
    caudal_reseg = df.loc['Caudal RESEGUIMIENTO'].values
    numero_reseg = df.loc['Nº RESEGUIMIENTO'].values.astype(float)
    un_ano_005 = df.loc['1 año 0,05'].values.astype(float)
    dos_anos_005 = df.loc['2 años 0,05'].values.astype(float)
    un_ano_5 = df.loc['1 año 5'].values.astype(float)
    dos_anos_5 = df.loc['2 años 5'].values.astype(float)
    dos_anos_16 = df.loc['2 años 16'].values.astype(float)
    
    grupo1 = un_ano_005 + dos_anos_005
    grupo2 = un_ano_5 + dos_anos_5
    grupo3 = dos_anos_16
    grupo4 = np.zeros_like(grupo3)
    
    return caudal_reseg, numero_reseg, grupo1, grupo2, grupo3, grupo4

def entrenar_modelos(X_normalizado, targets, mask=None):
    model = Ridge(alpha=1)
    if mask is not None:
        model.fit(X_normalizado[mask], targets[mask])
    else:
        model.fit(X_normalizado, targets)
    return model

def print_equation(model, name):
    var_names = ["log(IPS)", "FD*0.26", "PE*0.08", "Otros*0.17", "GasVehiculado"]
    equation = f"Ecuación estimada para {name}: f(x) = {model.intercept_:.2f}"
    for coef, var in zip(model.coef_, var_names):
        if coef >= 0:
            equation += f" + {coef:.2f}*{var}"
        else:
            equation += f" - {abs(coef):.2f}*{var}"
    print(equation)
    return equation

def print_general_equation(model_ips, model_aviso, model_reseg):
    var_names = ["log(IPS)", "FD*0.26", "PE*0.08", "Otros*0.17", "GasVehiculado"]
    general_intercept = model_ips.intercept_ + model_aviso.intercept_ + model_reseg.intercept_
    general_coef = model_ips.coef_ + model_aviso.coef_ + model_reseg.coef_
    equation = f"Ecuación General: f(x) = {general_intercept:.2f}"
    for coef, var in zip(general_coef, var_names):
        if coef >= 0:
            equation += f" + {coef:.2f}*{var}"
        else:
            equation += f" - {abs(coef):.2f}*{var}"
    print(equation)
    return equation

# =============================================================================
# Interfaz de usuario con Streamlit
# =============================================================================
with st.sidebar.expander("📂 Carga de archivos", expanded=True):
    archivo = st.file_uploader("Sube tu archivo Excel", type=["xlsx", "xls"])

if archivo is not None:
    hojas = ["General", "IPs", "Aviso", "Reseguimiento"]
    dfs = leer_y_procesar_excel(archivo, hojas)
    
    st.header("Resultados Generales")
    df_general = dfs.get("General")
    if df_general is not None:
        # Extraer datos de cada hoja
        fd_general, pe_general, otros_general, gas_vehiculado_general, pto_suministro_general, caudal_general, anios_general = extraer_datos_generales(df_general)
        if fd_general is None:
            st.stop() # Detener la ejecución si hay un error
        
        # Extraer datos de IPs
        caudal_ips_real, numero_ips_real = extraer_datos_ips(dfs["IPs"])
        if caudal_ips_real is None or numero_ips_real is None:
            st.stop() # Detener la ejecución si hay un error
        
        # Extraer datos de Aviso
        caudal_aviso_real, numero_aviso_real, mop_005, mop_005_015, mop_015_5, mop_16, mop_over16 = extraer_datos_aviso(dfs["Aviso"])
        if caudal_aviso_real is None or numero_aviso_real is None:
            st.stop() # Detener la ejecución si hay un error
        
        # Extraer datos de Reseguimiento
        caudal_reseg_real, numero_reseg_real, grupo1_reseg, grupo2_reseg, grupo3_reseg, grupo4_reseg = extraer_datos_reseguimiento(dfs["Reseguimiento"])
        if caudal_reseg_real is None or numero_reseg_real is None:
            st.stop() # Detener la ejecución si hay un error
        
        # Asegurar consistencia en la longitud de datos
        min_length = min(len(anios_general), len(fd_general), len(numero_ips_real), len(numero_aviso_real), len(numero_reseg_real))
        if min_length == 0:
            st.error("Una o más variables no tienen datos suficientes.")
        else:
            # Recortar vectores para que tengan la misma longitud
            anios_general = anios_general[:min_length]
            fd_general = fd_general[:min_length]
            pe_general = pe_general[:min_length]
            otros_general = otros_general[:min_length]
            gas_vehiculado_general = gas_vehiculado_general[:min_length]
            pto_suministro_general = pto_suministro_general[:min_length]
            caudal_general = caudal_general[:min_length]
            numero_ips_real = numero_ips_real[:min_length]
            caudal_ips_real = caudal_ips_real[:min_length]
            numero_aviso_real = numero_aviso_real[:min_length]
            caudal_aviso_real = caudal_aviso_real[:min_length]
            mop_005 = mop_005[:min_length]
            mop_005_015 = mop_005_015[:min_length]
            mop_015_5 = mop_015_5[:min_length]
            mop_16 = mop_16[:min_length]
            mop_over16 = mop_over16[:min_length]
            numero_reseg_real = numero_reseg_real[:min_length]
            caudal_reseg_real = caudal_reseg_real[:min_length]
            grupo1_reseg = grupo1_reseg[:min_length]
            grupo2_reseg = grupo2_reseg[:min_length]
            grupo3_reseg = grupo3_reseg[:min_length]
            grupo4_reseg = grupo4_reseg[:min_length]
            
            # Estimación de fugas mediante regresión Ridge
            prob_fd, prob_pe, prob_otros = 0.26, 0.08, 0.17
            log_num_ips = np.log(np.where(numero_ips_real > 0, numero_ips_real, 1))
            X = np.column_stack((log_num_ips, fd_general * prob_fd, pe_general * prob_pe, otros_general * prob_otros, gas_vehiculado_general))
            X_normalizado = StandardScaler().fit_transform(X)
            
            # Entrenar modelos para cada categoría
            modelo_ips = entrenar_modelos(X_normalizado, numero_ips_real)
            modelo_aviso = entrenar_modelos(X_normalizado, numero_aviso_real)
            modelo_reseg = entrenar_modelos(X_normalizado, numero_reseg_real)
            
            # Mostrar la ecuación resultante de cada modelo en consola
            eq_ips = print_equation(modelo_ips, "IPs")
            eq_aviso = print_equation(modelo_aviso, "Avisos")
            eq_reseg = print_equation(modelo_reseg, "Reseguimiento")
            
            # Mostrar la ecuación general combinada
            eq_general = print_general_equation(modelo_ips, modelo_aviso, modelo_reseg)
            st.info(eq_general.replace("\\n", "\\n\\n"))
            
            # Predicciones (fugas estimadas)
            fugas_ips_est = modelo_ips.predict(X_normalizado)
            fugas_aviso_est = modelo_aviso.predict(X_normalizado)
            fugas_reseg_est = modelo_reseg.predict(X_normalizado)
            
            # Cálculo del caudal estimado según las reglas definidas
            tiempo = 8760 # horas por año
            
            # Para IPs
            caudal_ips_est = fugas_ips_est * tiempo * 0.005
            
            # Para Avisos
            p_aviso_g1 = np.where(numero_aviso_real != 0, mop_005 / numero_aviso_real, 0)
            p_aviso_g2 = np.where(numero_aviso_real != 0, (mop_005_015 + mop_015_5) / numero_aviso_real, 0)
            p_aviso_g3 = np.where(numero_aviso_real != 0, mop_16 / numero_aviso_real, 0)
            p_aviso_g4 = np.where(numero_aviso_real != 0, mop_over16 / numero_aviso_real, 0)
            leaks_aviso_g1 = fugas_aviso_est * p_aviso_g1
            leaks_aviso_g2 = fugas_aviso_est * p_aviso_g2
            leaks_aviso_g3 = fugas_aviso_est * p_aviso_g3
            leaks_aviso_g4 = fugas_aviso_est * p_aviso_g4
            caudal_aviso_g1 = leaks_aviso_g1 * 4.12 * 0.20
            caudal_aviso_g2 = leaks_aviso_g2 * 0.86 * 1.34
            caudal_aviso_g3 = leaks_aviso_g3 * 1.88 * 5.02
            caudal_aviso_g4 = leaks_aviso_g4 * 2 * 16.49
            caudal_aviso_est = caudal_aviso_g1 + caudal_aviso_g2 + caudal_aviso_g3 + caudal_aviso_g4
            
            # Para Reseguimiento
            total_reseg = np.where(numero_reseg_real != 0, numero_reseg_real, 1)
            p_reseg_g1 = grupo1_reseg / total_reseg
            p_reseg_g2 = grupo2_reseg / total_reseg
            p_reseg_g3 = grupo3_reseg / total_reseg
            p_reseg_g4 = np.where(total_reseg - (grupo1_reseg + grupo2_reseg + grupo3_reseg) > 0, (total_reseg - (grupo1_reseg + grupo2_reseg + grupo3_reseg)) / total_reseg, 0)
            leaks_reseg_g1 = fugas_reseg_est * p_reseg_g1
            leaks_reseg_g2 = fugas_reseg_est * p_reseg_g2
            leaks_reseg_g3 = fugas_reseg_est * p_reseg_g3
            leaks_reseg_g4 = fugas_reseg_est * p_reseg_g4
            caudal_reseg_g1 = leaks_reseg_g1 * tiempo * 0.05
            caudal_reseg_g2 = leaks_reseg_g2 * tiempo * 0.28
            caudal_reseg_g3 = leaks_reseg_g3 * tiempo * 1.14
            caudal_reseg_g4 = leaks_reseg_g4 * tiempo * 1.14
            caudal_reseg_est = caudal_reseg_g1 + caudal_reseg_g2 + caudal_reseg_g3 + caudal_reseg_g4
            mask_reseg_year = anios_general >= 2021
            fugas_reseg_est = np.where(mask_reseg_year, fugas_reseg_est, 0)
            caudal_reseg_est = np.where(mask_reseg_year, caudal_reseg_est, 0)
            
            # Caudal total estimado
            caudal_total_est = caudal_ips_est + caudal_aviso_est + caudal_reseg_est
            
            # Cálculo de errores y tolerancia
            error_caudal = caudal_total_est - caudal_general
            error_caudal_pct = np.where(caudal_general != 0, (error_caudal / caudal_general) * 100, 0)
            media_tolerancia = np.mean(np.abs(error_caudal_pct))
            
            # Gráficos comparativos
            st.subheader("Gráficos Comparativos")
            
            # Fugas Reales vs Estimadas con Plotly
            fig = px.line(x=anios_general, y=[numero_ips_real, fugas_ips_est, numero_aviso_real, fugas_aviso_est, numero_reseg_real, fugas_reseg_est],
                          labels={'x': 'Año', 'value': 'Número de Fugas'},
                          title='Fugas Reales vs Estimadas')
            fig.update_layout(legend_title_text='Categorías')
            fig.for_each_trace(lambda t: t.update(name={'wide_variable_0': 'Reales IPs', 'wide_variable_1': 'Estimadas IPs',
                                                        'wide_variable_2': 'Reales Avisos', 'wide_variable_3': 'Estimadas Avisos',
                                                        'wide_variable_4': 'Reales Reseguimiento', 'wide_variable_5': 'Estimadas Reseguimiento'}[t.name]))
            # Añadir puntos a las líneas
            fig.update_traces(mode='lines+markers')
            st.plotly_chart(fig)
            
            # Caudales Reales vs Estimados con Plotly
            fig = px.line(x=anios_general, y=[caudal_general, caudal_total_est],
                          labels={'x': 'Año', 'value': 'Caudal (m³/h)'},
                          title='Caudales Reales vs Estimados')
            fig.update_layout(legend_title_text='Categorías')
            fig.for_each_trace(lambda t: t.update(name={'wide_variable_0': 'Caudal Real (m³/h)', 'wide_variable_1': 'Caudal Estimado (m³/h)'}[t.name]))
            # Añadir puntos a las líneas
            fig.update_traces(mode='lines+markers')
            st.plotly_chart(fig)
            
            # Creación de tablas resumen
            st.subheader("Tablas Resumen")
            
            df_desglose = pd.DataFrame({
                "Año": anios_general,
                "Fugas IP Real": np.round(numero_ips_real, 0).astype(int),
                "Fugas Aviso Real": np.round(numero_aviso_real, 0).astype(int),
                "Fugas Reseguimiento Real": np.round(numero_reseg_real, 0).astype(int),
                "Fugas IP Est": np.round(fugas_ips_est, 0).astype(int),
                "Fugas Aviso Est": np.round(fugas_aviso_est, 0).astype(int),
                "Fugas Reseguimiento Est": np.round(fugas_reseg_est, 0).astype(int),
                "Caudal IP Est (m³/h)": np.round(caudal_ips_est, 0).astype(int),
                "Caudal Reseguimiento Est (m³/h)": np.round(caudal_reseg_est, 0).astype(int)
            })
            st.dataframe(df_desglose)
            
            df_metricas = pd.DataFrame({
                "Métrica": ["Media Tolerancia (% Error Caudal)"],
                "Valor": [np.round(media_tolerancia, 2)]
            })
            st.dataframe(df_metricas)
            
            # Predicción a Futuro con ARIMA
            st.subheader("Predicción a Futuro con ARIMA")
            modelo_arima = ARIMA(caudal_general, order=(2,1,2))
            modelo_arima_fit = modelo_arima.fit()
            predicciones = modelo_arima_fit.forecast(steps=5)
            anios_futuros = np.arange(anios_general[-1] + 1, anios_general[-1] + 6)
            
            # Crear DataFrame para la predicción
            df_predicciones = pd.DataFrame({
                "Año": np.concatenate((anios_general, anios_futuros)),
                "Caudal": np.concatenate((caudal_general, predicciones)),
                "Tipo": ["Real"] * len(anios_general) + ["Estimado"] * len(anios_futuros)
            })
            
            fig = px.line(df_predicciones, x="Año", y="Caudal", color="Tipo",
                          labels={'Año': 'Año', 'Caudal': 'Caudal (m³/h)'},
                          title='Proyección a Futuro con ARIMA')
            fig.update_traces(mode='lines+markers')
            st.plotly_chart(fig)
    else:
        st.error("No se encontró la hoja 'General' en el archivo.")
