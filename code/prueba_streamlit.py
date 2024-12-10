import joblib

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config.config_file import (
    R1_BEST_MODEL_RMSE_PATH,
    R1_BEST_MODEL_MAE_PATH,
    R1_BEST_MODEL_R2_PATH,
    R2_BEST_MODEL_RMSE_PATH,
    R2_BEST_MODEL_MAE_PATH,
    R2_BEST_MODEL_R2_PATH,
    R7_BEST_MODEL_RMSE_PATH,
    R7_BEST_MODEL_MAE_PATH,
    R7_BEST_MODEL_R2_PATH,
    R28_BEST_MODEL_RMSE_PATH,
    R28_BEST_MODEL_MAE_PATH,
    R28_BEST_MODEL_R2_PATH
)

def scale_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Escala un dataframe utilizando StandardScaler.

    Args:
        df (pd.DataFrame): DataFrame con los datos a escalar.

    Returns:
        pd.DataFrame: DataFrame escalado con las mismas columnas y estructura que el original.
    """
    # Inicializa el escalador
    scaler = StandardScaler()

    # Realiza el escalamiento
    scaled_data = scaler.fit_transform(df)

    # Crea un nuevo dataframe con los datos escalados y las mismas columnas que el original
    df_scaled = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    return df_scaled


# Título de la aplicación
st.title("Resistencia del Cemento")

@st.cache_resource
def load_model(path):
    return joblib.load(path)


# Título de la aplicación
st.subheader("Ver las predicciones de la resistencia del cemento para los dias 1, 2, 7 y 28 según los datos ingresados")
# Cargar los modelos
@st.cache_resource
def load_model(path):
    return joblib.load(path)

# Cargar los modelos
model_paths = {
    "R1": {
        "RMSE": R1_BEST_MODEL_RMSE_PATH,
        "MAE": R1_BEST_MODEL_MAE_PATH,
        "R2": R1_BEST_MODEL_R2_PATH
    },
    "R2": {
        "RMSE": R2_BEST_MODEL_RMSE_PATH,
        "MAE": R2_BEST_MODEL_MAE_PATH,
        "R2": R2_BEST_MODEL_R2_PATH
    },
    "R7": {
        "RMSE": R7_BEST_MODEL_RMSE_PATH,
        "MAE": R7_BEST_MODEL_MAE_PATH,
        "R2": R7_BEST_MODEL_R2_PATH
    },
    "R28": {
        "RMSE": R28_BEST_MODEL_RMSE_PATH,
        "MAE": R28_BEST_MODEL_MAE_PATH,
        "R2": R28_BEST_MODEL_R2_PATH
    }
}

models = {metric: {key: load_model(path) for key, path in paths.items()} for metric, paths in model_paths.items()}

# Función para predecir con múltiples modelos
def predict_with_models(models, X_input):
    predictions = {}
    for day, models_by_metric in models.items():
        predictions[day] = {}
        for metric, model in models_by_metric.items():
            predictions[day][metric] = model.predict(X_input)
    return predictions

# Sidebar para selección del método de entrada
st.sidebar.header("Opciones de Entrada")
data_input_mode = st.sidebar.radio(
    "Selecciona el método de entrada de datos",
    options=["Manual", "Archivo Plano"]
)

# Entrada manual de datos
if data_input_mode == "Manual":
    st.sidebar.subheader("Ingresa los valores manualmente")
    input_data = {}
    for col in ['g45µ', 'sba', 'pf', 'so3', 'mgo', 'sio2', 'fe2o3', 'caot', 'al2o3', 'na2o', 'k2o']:
        input_data[col] = st.sidebar.number_input(col, value=0.0, min_value=0.0)  # Restricción a valores >= 0
    X_input = pd.DataFrame([input_data])

# Carga de archivo de datos
else:
    st.sidebar.subheader("Carga un archivo de datos")
    uploaded_file = st.sidebar.file_uploader("Sube un archivo archivo plano", type=['csv', 'txt'])
    separator = st.sidebar.text_input("Separador de columnas", value=',', max_chars=1)
    if uploaded_file is not None:
        X_input = pd.read_csv(uploaded_file, encoding='utf-8', sep=separator)

# Realizar predicciones
if st.sidebar.button("Realizar Predicciones"):
    # Validar que X_input no sea nulo
    if "X_input" in locals() and not X_input.empty:
        # Mostrar los datos ingresados
        st.subheader("Datos ingresados")
        st.write("A continuación se muestran los datos cargados para realizar las predicciones:")
        st.dataframe(X_input)

        list_columns = [
            "R1_Minimo",
            "R1_Maximo",
            "R2_Minimo",
            "R2_Maximo",
            "R7_Minimo",
            "R7_Maximo",
            "R28_Minimo",
            "R28_Maximo"
        ]

        res_df = X_input.copy()
        X_input = scale_dataframe(X_input)

        predictions = predict_with_models(models, X_input)

        day_predictions = []
        for row in X_input.itertuples(index=True):
            index = row.Index
            indx_predictions = []
            for day, pred in predictions.items():
                list_values = [
                    pred["RMSE"][index],
                    pred["MAE"][index],
                    pred["R2"][index]
                ]
                indx_predictions.append(min(list_values))
                indx_predictions.append(max(list_values))

            day_predictions.append(indx_predictions)

        new_df = pd.DataFrame(day_predictions, columns=list_columns)
        res_df = pd.concat([res_df, new_df], axis=1)

        st.write("Dataframe con las predicciones realizadas:")
        st.dataframe(res_df)

        st.success("Predicciones realizadas con éxito.")
    else:
        st.error("Por favor, ingresa datos válidos para realizar las predicciones.")
