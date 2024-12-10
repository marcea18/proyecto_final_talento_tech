# proyecto_final_talento_tech

## Pronóstico de la Resistencia del Cemento

### Descripción del Proyecto
Este proyecto es parte del bootcamp de Inteligencia Artificial de Talento Tech IA Intermedio. El objetivo principal es aplicar técnicas de machine learning para predecir la resistencia del cemento, un problema de negocio relevante en la industria de la construcción. La resistencia del cemento es un factor crucial que determina la calidad y durabilidad de las estructuras construidas.

### Objetivos
- Aplicar técnicas de preprocesamiento de datos para limpiar y preparar el conjunto de datos.
- Utilizar algoritmos de machine learning para construir modelos predictivos.
- Evaluar y seleccionar el mejor modelo basado en métricas de desempeño.
- Implementar técnicas de ajuste de hiperparámetros para optimizar el modelo seleccionado.
- Desplegar el modelo final en una aplicación web interactiva utilizando Streamlit.

### Conjunto de Datos
El conjunto de datos utilizado en este proyecto contiene información sobre la composición del cemento y su resistencia. Las características incluyen:
- **Composición química**: Cantidades de diferentes componentes químicos presentes en el cemento.
- **Propiedades físicas**: Características físicas del cemento.
- **Resistencia**: La resistencia del cemento medida en diferentes condiciones en diferentes días, los cuales las variables objetivo a calcular

### Metodología
1. **Exploración de Datos (EDA)**:
   - Análisis exploratorio de datos para entender la distribución y relaciones entre las variables.
   - Visualización de datos usando la librería ydata_profiling para identificar patrones y valores atípicos.

2. **Preprocesamiento de Datos**:
   - Limpieza de datos: Manejo de valores nulos y eliminación de registros irrelevantes.
   - Transformación de datos: Normalización y escalado de características.

3. **Modelado**:
   - Selección de algoritmos de machine learning: Random Forest, Gradient Boosting, XGBoost, CatBoost, Ridge Regression, Lasso Regression, ElasticNet, Support Vector Regression, SGD Regression, Decision Tree Regressor.
   - Entrenamiento de modelos y evaluación utilizando validación cruzada.
   - Ajuste de hiperparámetros para optimizar el rendimiento del modelo.

4. **Evaluación del Modelo**:
   - Métricas de evaluación: RMSE, MAE, R².
   - Selección de los mejores modelo basado en las métricas de desempeño.

5. **Despliegue**:
   - Implementación del modelo final en una aplicación web interactiva utilizando Streamlit.
   - Visualización de resultados y predicciones en tiempo real.

### Resultados
El modelo final seleccionado depende del día que se realice el cálculo y la métrica para predecir la resistencia del cemento, ya que dependiendo de estos factores se selecciona un modelo y otro. Todos los modelos fueron optimizados mediante ajuste de hiperparámetros. Las métricas de desempeño de los modelos son:

#### Resultados del Modelo - Día R1

##### Mejor Modelo por RMSE
- **Modelo**: XGBRegressor
- **Mejor RMSE en el conjunto de validación**: 1.7337
- **Mejores Hiperparámetros (RMSE)**:
  - `learning_rate`: 0.1
  - `max_depth`: 4
  - `n_estimators`: 200
  - `subsample`: 0.9

##### Mejor Modelo por MAE
- **Modelo**: CatBoostRegressor
- **Mejor MAE en el conjunto de validación**: 1.4073
- **Mejores Hiperparámetros (MAE)**:
  - `depth`: 5
  - `iterations`: 100
  - `learning_rate`: 0.2

##### Mejor Modelo por R²
- **Modelo**: XGBRegressor
- **Mejor R² en el conjunto de validación**: 0.2363
- **Mejores Hiperparámetros (R²)**:
  - `learning_rate`: 0.1
  - `max_depth`: 4
  - `n_estimators`: 100
  - `subsample`: 0.9

#### Resultados del Modelo - Día R2

##### Mejor Modelo por RMSE
- **Modelo**: RandomForestRegressor
- **Mejor RMSE en el conjunto de validación**: 1.7388
- **Mejores Hiperparámetros (RMSE)**:
  - `max_depth`: None
  - `min_samples_leaf`: 1
  - `min_samples_split`: 2
  - `n_estimators`: 500

##### Mejor Modelo por MAE
- **Modelo**: RandomForestRegressor
- **Mejor MAE en el conjunto de validación**: 1.3759
- **Mejores Hiperparámetros (MAE)**:
  - `max_depth`: 20
  - `min_samples_leaf`: 1
  - `min_samples_split`: 2
  - `n_estimators`: 500

##### Mejor Modelo por R²
- **Modelo**: [Nombre del modelo]
- **Mejor R² en el conjunto de validación**: 0.2195
- **Mejores Hiperparámetros (R²)**:
  - `max_depth`: 20
  - `min_samples_leaf`: 1
  - `min_samples_split`: 5
  - `n_estimators`: 500

#### Resultados del Modelo - Día R7

##### Mejor Modelo por RMSE
- **Modelo**: CatBoostRegressor
- **Mejor RMSE en el conjunto de validación**: 2.0173
- **Mejores Hiperparámetros (RMSE)**:
  - `depth`: 4
  - `iterations`: 200
  - `learning_rate`: 0.1

##### Mejor Modelo por MAE
- **Modelo**: CatBoostRegressor
- **Mejor MAE en el conjunto de validación**: 1.6237
- **Mejores Hiperparámetros (MAE)**:
  - `depth`: 4
  - `iterations`: 200
  - `learning_rate`: 0.1

##### Mejor Modelo por R²
- **Modelo**: RandomForestRegressor
- **Mejor R² en el conjunto de validación**: 0.2763
- **Mejores Hiperparámetros (R²)**:
  - `max_depth`: None
  - `min_samples_leaf`: 1
  - `min_samples_split`: 5
  - `n_estimators`: 500

#### Resultados del Modelo - Día R28

##### Mejor Modelo por RMSE
- **Modelo**: CatBoostRegressor
- **Mejor RMSE en el conjunto de validación**: 1.7153
- **Mejores Hiperparámetros (RMSE)**:
  - `depth`: 5
  - `iterations`: 100
  - `learning_rate`: 0.1

##### Mejor Modelo por MAE
- **Modelo**: CatBoostRegressor
- **Mejor MAE en el conjunto de validación**: 1.3841
- **Mejores Hiperparámetros (MAE)**:
  - `depth`: 5
  - `iterations`: 100
  - `learning_rate`: 0.1

##### Mejor Modelo por R²
- **Modelo**: CatBoostRegressor
- **Mejor R² en el conjunto de validación**: 0.1530
- **Mejores Hiperparámetros (R²)**:
  - `depth`: 5
  - `iterations`: 100
  - `learning_rate`: 0.1

### Conclusiones
- La aplicación de técnicas de preprocesamiento y ajuste de hiperparámetros fue crucial para mejorar la precisión del modelo.
- Notamos que, al usar los 3 modelos para calcular la resistencia en un día específico, estos nos dan un rango el cual, conforme se acerca a R28, tiende a converger en un valor.
- La implementación en Streamlit permite una interacción fácil y visualización de los resultados de manera efectiva.

### Requisitos
- Python 3.8+
- Bibliotecas: pandas, numpy, scikit-learn, xgboost, catboost, lightgbm, streamlit, matplotlib, seaborn

### Instrucciones de Uso
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/tu_usuario/proyecto_final_talento_tech.git
   ```
2. Ubicarse en la carpeta raíz del proyecto
    ```bash
   cd code/
   ```
3. Crear y activar el ambiente virtual
    ```bash
   python -m venv .venv
   source .venv/Scripts/Activate
   ```
4. Instalar las dependencias
    ```bash
   pip install -r requirements.txt
   ```
5. Ejecutar la aplicación web
    ```bash
   streamlit run prueba_streamlit.py
   ```
