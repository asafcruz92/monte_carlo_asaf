import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# Configurar la pagina principal de Streamlit
st.set_page_config(
    page_title="Simulación Monte Carlo de Asaf Cruz",
    layout="wide"
)


# Configurar estilo general de los graficos
sns.set(style="whitegrid")


@st.cache_data(show_spinner=False)
def ejecutar_simulacion(
    n_simulaciones,
    lambda_frecuencia,
    media_log_severidad,
    sigma_log_severidad,
    nivel_confianza,
    margen_seguridad,
    semilla
):
    # Crear generador aleatorio con semilla para obtener resultados reproducibles
    rng = np.random.RandomState(semilla)

    # Simular la frecuencia de siniestros con distribucion Poisson
    frecuencia_simulada = rng.poisson(
        lam=lambda_frecuencia,
        size=n_simulaciones
    )

    # Calcular estadisticos de frecuencia
    promedio_frecuencia = np.mean(frecuencia_simulada)
    desviacion_frecuencia = np.std(frecuencia_simulada)

    # Simular la severidad individual con distribucion Lognormal
    severidad_simulada = rng.lognormal(
        mean=media_log_severidad,
        sigma=sigma_log_severidad,
        size=n_simulaciones
    )

    # Calcular estadisticos de severidad
    media_severidad = np.mean(severidad_simulada)
    mediana_severidad = np.median(severidad_simulada)
    percentil_95_severidad = np.percentile(severidad_simulada, 95)

    # Crear una lista para guardar la perdida total de cada escenario
    perdida_agregada = []

    # Simular severidades por cada escenario y sumar la perdida total
    for frecuencia in frecuencia_simulada:
        severidades_escenario = rng.lognormal(
            mean=media_log_severidad,
            sigma=sigma_log_severidad,
            size=frecuencia
        )
        perdida_agregada.append(np.sum(severidades_escenario))

    # Convertir las perdidas agregadas en arreglo numerico
    perdida_agregada = np.array(perdida_agregada)

    # Calcular prima pura como perdida promedio esperada
    prima_pura = np.mean(perdida_agregada)

    # Calcular VaR segun el nivel de confianza seleccionado
    var = np.percentile(perdida_agregada, nivel_confianza * 100)

    # Calcular TVaR como promedio de las perdidas que superan el VaR
    tvar = np.mean(perdida_agregada[perdida_agregada >= var])

    # Calcular prima con margen de seguridad
    prima_con_margen = prima_pura * (1 + margen_seguridad)

    # Calcular probabilidad de insuficiencia de prima
    probabilidad_insuficiencia = np.mean(perdida_agregada > prima_con_margen)

    # Crear dataset simulado para visualizar y descargar
    datos_simulados = pd.DataFrame({
        "frecuencia": frecuencia_simulada,
        "perdida_agregada": perdida_agregada
    })

    # Preparar datos para el modelo de regresion lineal simple
    X = datos_simulados[["frecuencia"]]
    y = datos_simulados["perdida_agregada"]

    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=2026
    )

    # Crear y entrenar el modelo de regresion lineal
    modelo_lineal = LinearRegression()
    modelo_lineal.fit(X_train, y_train)

    # Generar predicciones para evaluar el modelo
    y_pred = modelo_lineal.predict(X_test)

    # Calcular metricas de evaluacion del modelo
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Guardar todos los resultados en un diccionario
    return {
        "frecuencia_simulada": frecuencia_simulada,
        "promedio_frecuencia": promedio_frecuencia,
        "desviacion_frecuencia": desviacion_frecuencia,
        "severidad_simulada": severidad_simulada,
        "media_severidad": media_severidad,
        "mediana_severidad": mediana_severidad,
        "percentil_95_severidad": percentil_95_severidad,
        "perdida_agregada": perdida_agregada,
        "prima_pura": prima_pura,
        "var": var,
        "tvar": tvar,
        "prima_con_margen": prima_con_margen,
        "probabilidad_insuficiencia": probabilidad_insuficiencia,
        "datos_simulados": datos_simulados,
        "modelo_lineal": modelo_lineal,
        "X_test": X_test,
        "y_test": y_test,
        "mae": mae,
        "r2": r2
    }


def formato_moneda(valor):
    # Formatear valores monetarios con separadores de miles
    return f"{valor:,.2f}"


def formato_porcentaje(valor):
    # Formatear valores porcentuales
    return f"{valor:.2%}"


# Crear controles de parametros en el panel lateral
st.sidebar.header("Parámetros del modelo")

# Controlar cantidad de escenarios simulados
n_simulaciones = st.sidebar.slider(
    "Número de simulaciones",
    min_value=1000,
    max_value=50000,
    value=10000,
    step=1000
)

# Controlar frecuencia promedio esperada
lambda_frecuencia = st.sidebar.slider(
    "Frecuencia promedio esperada",
    min_value=10,
    max_value=200,
    value=80,
    step=1
)

# Controlar media logaritmica de severidad
media_log_severidad = st.sidebar.slider(
    "Media logarítmica de severidad",
    min_value=4.0,
    max_value=12.0,
    value=8.5,
    step=0.1
)

# Controlar sigma logaritmica de severidad
sigma_log_severidad = st.sidebar.slider(
    "Sigma logarítmica de severidad",
    min_value=0.1,
    max_value=2.0,
    value=0.9,
    step=0.1
)

# Controlar nivel de confianza
nivel_confianza = st.sidebar.slider(
    "Nivel de confianza",
    min_value=0.80,
    max_value=0.99,
    value=0.95,
    step=0.01
)

# Controlar margen de seguridad
margen_seguridad = st.sidebar.slider(
    "Margen de seguridad",
    min_value=0.00,
    max_value=0.50,
    value=0.25,
    step=0.01
)

# Controlar semilla aleatoria
semilla = st.sidebar.number_input(
    "Semilla aleatoria",
    min_value=1,
    value=2026,
    step=1
)


# Ejecutar simulacion con los parametros seleccionados
resultados = ejecutar_simulacion(
    n_simulaciones=n_simulaciones,
    lambda_frecuencia=lambda_frecuencia,
    media_log_severidad=media_log_severidad,
    sigma_log_severidad=sigma_log_severidad,
    nivel_confianza=nivel_confianza,
    margen_seguridad=margen_seguridad,
    semilla=semilla
)


# Mostrar titulo principal de la aplicacion
st.title("Simulación Monte Carlo Actuarial")

# Mostrar descripcion breve de la aplicacion
st.write(
    "Aplicación basada en una simulación Monte Carlo para estimar frecuencia, "
    "severidad, pérdida agregada, prima pura, VaR, TVaR, insuficiencia de prima "
    "y un modelo predictivo simple."
)


# Mostrar indicadores principales
st.subheader("Indicadores principales")

col1, col2, col3, col4 = st.columns(4)

with col1:
    # Mostrar prima pura
    st.metric("Prima pura", formato_moneda(resultados["prima_pura"]))

with col2:
    # Mostrar VaR
    st.metric(f"VaR {int(nivel_confianza * 100)}%", formato_moneda(resultados["var"]))

with col3:
    # Mostrar TVaR
    st.metric(f"TVaR {int(nivel_confianza * 100)}%", formato_moneda(resultados["tvar"]))

with col4:
    # Mostrar prima con margen
    st.metric("Prima con margen", formato_moneda(resultados["prima_con_margen"]))


# Crear tabla resumen de resultados
st.subheader("Tabla resumen")

tabla_resumen = pd.DataFrame({
    "Indicador": [
        "Frecuencia promedio",
        "Desviación estándar frecuencia",
        "Severidad promedio",
        "Severidad mediana",
        "Percentil 95 severidad",
        "Prima pura",
        f"VaR {int(nivel_confianza * 100)}%",
        f"TVaR {int(nivel_confianza * 100)}%",
        "Margen de riesgo TVaR - Prima pura",
        f"Prima con margen {int(margen_seguridad * 100)}%",
        "Probabilidad de insuficiencia",
        "MAE modelo ML",
        "R² modelo ML"
    ],
    "Valor": [
        resultados["promedio_frecuencia"],
        resultados["desviacion_frecuencia"],
        resultados["media_severidad"],
        resultados["mediana_severidad"],
        resultados["percentil_95_severidad"],
        resultados["prima_pura"],
        resultados["var"],
        resultados["tvar"],
        resultados["tvar"] - resultados["prima_pura"],
        resultados["prima_con_margen"],
        resultados["probabilidad_insuficiencia"],
        resultados["mae"],
        resultados["r2"]
    ]
})

# Mostrar tabla resumen
st.dataframe(tabla_resumen, use_container_width=True)


# Mostrar dataset simulado
st.subheader("Dataset simulado")

# Mostrar primeras filas del dataset generado
st.dataframe(resultados["datos_simulados"].head(100), use_container_width=True)

# Permitir descarga del dataset simulado
st.download_button(
    label="Descargar dataset simulado en CSV",
    data=resultados["datos_simulados"].to_csv(index=False).encode("utf-8"),
    file_name="dataset_montecarlo.csv",
    mime="text/csv"
)


# Crear pestañas para visualizar resultados
st.subheader("Visualizaciones")

tab_frecuencia, tab_severidad, tab_perdida, tab_var, tab_relacion = st.tabs([
    "Frecuencia",
    "Severidad",
    "Pérdida agregada",
    "VaR y TVaR",
    "Frecuencia vs pérdida"
])

with tab_frecuencia:
    # Crear histograma de frecuencia simulada
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(resultados["frecuencia_simulada"], bins=30, kde=False, color="steelblue", ax=ax)
    ax.axvline(resultados["promedio_frecuencia"], color="red", linestyle="--", label="Promedio")
    ax.set_title("Histograma de la frecuencia simulada")
    ax.set_xlabel("Número de siniestros")
    ax.set_ylabel("Cantidad de escenarios")
    ax.legend()
    st.pyplot(fig)

    # Mostrar interpretacion actuarial de frecuencia
    st.write(
        "La frecuencia se concentra alrededor del promedio esperado. "
        "Actuarialmente, esto permite estimar cuántos siniestros podría enfrentar "
        "la cartera en un periodo determinado."
    )

with tab_severidad:
    # Crear histograma de severidad simulada
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(resultados["severidad_simulada"], bins=50, kde=True, color="darkorange", ax=ax)
    ax.axvline(resultados["media_severidad"], color="red", linestyle="--", label="Media")
    ax.axvline(resultados["mediana_severidad"], color="green", linestyle="--", label="Mediana")
    ax.axvline(resultados["percentil_95_severidad"], color="purple", linestyle="--", label="Percentil 95")
    ax.set_title("Distribución de la severidad simulada")
    ax.set_xlabel("Monto del siniestro")
    ax.set_ylabel("Cantidad de escenarios")
    ax.legend()
    st.pyplot(fig)

    # Mostrar interpretacion actuarial de severidad
    st.write(
        "La severidad presenta asimetría hacia la derecha. Esto indica que la mayoría "
        "de siniestros son moderados, pero algunos pueden ser muy altos y generar "
        "riesgo extremo."
    )

with tab_perdida:
    # Crear histograma de perdida agregada
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(resultados["perdida_agregada"], bins=50, kde=True, color="crimson", ax=ax)
    ax.set_title("Distribución de la pérdida agregada simulada")
    ax.set_xlabel("Pérdida agregada")
    ax.set_ylabel("Cantidad de escenarios")
    st.pyplot(fig)

    # Mostrar interpretacion actuarial de perdida agregada
    st.write(
        "La pérdida agregada combina la frecuencia y la severidad. Por eso permite "
        "observar la pérdida total esperada de la cartera en miles de escenarios."
    )

with tab_var:
    # Crear histograma de perdida agregada con VaR y TVaR
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(resultados["perdida_agregada"], bins=50, kde=True, color="lightcoral", ax=ax)
    ax.axvline(resultados["var"], color="blue", linestyle="--", label=f"VaR {int(nivel_confianza * 100)}%")
    ax.axvline(resultados["tvar"], color="purple", linestyle="--", label=f"TVaR {int(nivel_confianza * 100)}%")
    ax.set_title("VaR y TVaR sobre la pérdida agregada")
    ax.set_xlabel("Pérdida agregada")
    ax.set_ylabel("Cantidad de escenarios")
    ax.legend()
    st.pyplot(fig)

    # Mostrar interpretacion actuarial de VaR y TVaR
    st.write(
        "El VaR marca el umbral de pérdida para el nivel de confianza seleccionado. "
        "El TVaR es más conservador porque resume el promedio de las pérdidas que "
        "superan ese umbral."
    )

with tab_relacion:
    # Crear grafico de dispersion entre frecuencia y perdida agregada
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(
        resultados["frecuencia_simulada"],
        resultados["perdida_agregada"],
        alpha=0.5,
        color="teal"
    )
    ax.set_title("Relación entre frecuencia y pérdida agregada")
    ax.set_xlabel("Frecuencia de siniestros")
    ax.set_ylabel("Pérdida agregada")
    st.pyplot(fig)

    # Mostrar interpretacion actuarial de la relacion
    st.write(
        "Existe una relación positiva entre frecuencia y pérdida agregada. Sin embargo, "
        "la pérdida total también depende de la severidad, por lo que escenarios con "
        "igual frecuencia pueden tener pérdidas diferentes."
    )


# Mostrar analisis de insuficiencia de prima
st.subheader("Insuficiencia de prima")

col5, col6 = st.columns(2)

with col5:
    # Mostrar margen seleccionado
    st.metric("Margen de seguridad", formato_porcentaje(margen_seguridad))

with col6:
    # Mostrar probabilidad de insuficiencia
    st.metric("Probabilidad de insuficiencia", formato_porcentaje(resultados["probabilidad_insuficiencia"]))

st.write(
    "La insuficiencia ocurre cuando la pérdida agregada supera la prima con margen. "
    "Una probabilidad más alta indica mayor riesgo de que la prima cobrada no alcance "
    "para cubrir los siniestros simulados."
)


# Mostrar modelo predictivo simple
st.subheader("Modelo predictivo simple")

col7, col8 = st.columns(2)

with col7:
    # Mostrar MAE del modelo
    st.metric("MAE", formato_moneda(resultados["mae"]))

with col8:
    # Mostrar R2 del modelo
    st.metric("R²", f"{resultados['r2']:.4f}")

# Crear grafico de regresion lineal
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(
    resultados["X_test"],
    resultados["y_test"],
    alpha=0.5,
    color="teal",
    label="Valores reales"
)

# Crear linea ordenada para representar la regresion lineal
x_linea = np.linspace(
    resultados["datos_simulados"]["frecuencia"].min(),
    resultados["datos_simulados"]["frecuencia"].max(),
    100
)
y_linea = resultados["modelo_lineal"].predict(
    pd.DataFrame({"frecuencia": x_linea})
)

ax.plot(x_linea, y_linea, color="red", linewidth=2, label="Regresión lineal")
ax.set_title("Regresión lineal: frecuencia vs pérdida agregada")
ax.set_xlabel("Frecuencia de siniestros")
ax.set_ylabel("Pérdida agregada")
ax.legend()
st.pyplot(fig)

st.write(
    "El modelo muestra que la frecuencia ayuda a explicar parte de la pérdida agregada, "
    "pero no toda. La severidad también es fundamental para entender el riesgo actuarial."
)


# Mostrar conclusion final de la aplicacion
st.subheader("Conclusión actuarial")

st.write(
    "La simulación Monte Carlo permite visualizar la incertidumbre y evaluar miles de "
    "escenarios posibles. En seguros, ayuda a estimar pérdidas esperadas, medir riesgo "
    "extremo y analizar si una prima puede ser suficiente para cubrir los siniestros."
)
