from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import pyodbc
from io import StringIO
import base64

app = FastAPI()

# Listas para almacenar las métricas de todos los archivos
rmse_pls_list = []
mae_pls_list = []
precision_pls_list = []

rmse_rf_list = []
mae_rf_list = []
precision_rf_list = []

rmse_svm_list = []
mae_svm_list = []
precision_svm_list = []

fakedb = []

# Configuración de la conexión a la base de datos
server = 'adminpy.database.windows.net'
database = 'Nirpy'
username = 'adminpy'
password = 'CCoNNa2205**'
driver = 'ODBC Driver 17 for SQL Server'

# Cadena de conexión
conn_str = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'

# Crear la conexión
connection = pyodbc.connect(conn_str)
cursor = connection.cursor()

# Consulta SQL para obtener el último archivo cargado
query = "SELECT TOP 1 FileContent FROM FileUpload ORDER BY DateUploaded DESC"

# Ejecutar la consulta
cursor.execute(query)
row = cursor.fetchone()

# Define an endpoint for your API
@app.get("/nir")
def get_metrics():
    # ... (code for processing data and calculating metrics)
    if row:
    # Obtener el contenido del archivo en formato base64
        file_content_base64 = row.FileContent

        # Decodificar el contenido base64 para obtener el archivo CSV
        decoded_data = base64.b64decode(file_content_base64)

        # Convertir los datos decodificados a una cadena y asignar a la variable CafeAbsorbancia
        CafeAbsorbancia = decoded_data.decode('utf-8')

        # Crear un objeto StringIO y omitir las primeras 28 filas
        cafe_absorbancia_io = StringIO(CafeAbsorbancia)
        for _ in range(28):
            cafe_absorbancia_io.readline()

        # Cargar los datos en un DataFrame de pandas
        df = pd.read_csv(cafe_absorbancia_io)

        # Resto del código para procesar el DataFrame como lo hacías anteriormente
        # Detección y manejo de valores atípicos
        Q1 = df['Absorbance (AU)'].quantile(0.25)
        Q3 = df['Absorbance (AU)'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df['Absorbance (AU)'] < (Q1 - 1.5 * IQR)) | (df['Absorbance (AU)'] > (Q3 + 1.5 * IQR)))

        # Opción 1: Eliminar valores atípicos
        df_sin_atipicos = df[~outliers]

        # Opción 2: Ajustar valores atípicos
        df_ajustado = df.copy()
        df_ajustado.loc[outliers, 'Absorbance (AU)'] = df['Absorbance (AU)'].median()

        # División de datos en conjunto de entrenamiento y prueba (80-20)
        X = df_sin_atipicos['Wavelength (nm)'].values.reshape(-1, 1)
        y = df_sin_atipicos['Absorbance (AU)'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenamiento del modelo PLS con 1 componente
        pls_model = PLSRegression(n_components=1)
        pls_model.fit(X_train, y_train)

        # Predicciones en el conjunto de prueba para PLS
        y_pred_pls = pls_model.predict(X_test)

        # Calcula RMSE y MAE para PLS
        rmse_pls = sqrt(mean_squared_error(y_test, y_pred_pls))
        mae_pls = mean_absolute_error(y_test, y_pred_pls)

        # Calcula la precisión para PLS en porcentaje
        rango_y = np.max(y) - np.min(y)
        precision_pls = (1 - rmse_pls / rango_y) * 100

        # Almacena las métricas en las listas
        rmse_pls_list.append(rmse_pls)
        mae_pls_list.append(mae_pls)
        precision_pls_list.append(precision_pls)

        # Entrenamiento del modelo de Bosques Aleatorios
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Predicciones en el conjunto de prueba para Bosques Aleatorios
        y_pred_rf = rf_model.predict(X_test)

        # Calcula RMSE y MAE para Bosques Aleatorios
        rmse_rf = sqrt(mean_squared_error(y_test, y_pred_rf))
        mae_rf = mean_absolute_error(y_test, y_pred_rf)

        # Calcula la precisión para Bosques Aleatorios
        precision_rf = (1 - rmse_rf / rango_y) * 100

        # Almacena las métricas en las listas
        rmse_rf_list.append(rmse_rf)
        mae_rf_list.append(mae_rf)
        precision_rf_list.append(precision_rf)

        # Entrenamiento del modelo de Máquinas de Vectores de Soporte (SVM)
        svm_model = SVR(kernel='linear')
        svm_model.fit(X_train, y_train)

        # Predicciones en el conjunto de prueba para SVM
        y_pred_svm = svm_model.predict(X_test)

        # Calcula RMSE y MAE para SVM
        rmse_svm = sqrt(mean_squared_error(y_test, y_pred_svm))
        mae_svm = mean_absolute_error(y_test, y_pred_svm)

        # Calcula la precisión para SVM
        precision_svm = (1 - rmse_svm / rango_y) * 100

        # Almacena las métricas en las listas
        rmse_svm_list.append(rmse_svm)
        mae_svm_list.append(mae_svm)
        precision_svm_list.append(precision_svm)

        # Calcula el promedio de las métricas para PLS
        promedio_rmse_pls = sum(rmse_pls_list) / len(rmse_pls_list)
        promedio_mae_pls = sum(mae_pls_list) / len(mae_pls_list)
        promedio_precision_pls = sum(precision_pls_list) / len(precision_pls_list)

        print(f"Promedio de RMSE (PLS): {promedio_rmse_pls}, Promedio de MAE (PLS): {promedio_mae_pls}, Promedio de Precisión (PLS): {promedio_precision_pls}%")

        # Calcula el promedio de las métricas para Bosques Aleatorios
        promedio_rmse_rf = sum(rmse_rf_list) / len(rmse_rf_list)
        promedio_mae_rf = sum(mae_rf_list) / len(mae_rf_list)
        promedio_precision_rf = sum(precision_rf_list) / len(precision_rf_list)

        print(f"Promedio de RMSE (Bosques Aleatorios): {promedio_rmse_rf}, Promedio de MAE (Bosques Aleatorios): {promedio_mae_rf}, Promedio de Precisión (Bosques Aleatorios): {promedio_precision_rf}%")

        # Calcula el promedio de las métricas para SVM
        promedio_rmse_svm = sum(rmse_svm_list) / len(rmse_svm_list)
        promedio_mae_svm = sum(mae_svm_list) / len(mae_svm_list)
        promedio_precision_svm = sum(precision_svm_list) / len(precision_svm_list)

        print(f"Promedio de RMSE (SVM): {promedio_rmse_svm}, Promedio de MAE (SVM): {promedio_mae_svm}, Promedio de Precisión (SVM): {promedio_precision_svm}%")

    # Return the metrics as JSON
    result = {
        "promedio_rmse_pls": promedio_rmse_pls,
        "promedio_mae_pls": promedio_mae_pls,
        "promedio_precision_pls": promedio_precision_pls,
        "promedio_rmse_rf": promedio_rmse_rf,
        "promedio_mae_rf": promedio_mae_rf,
        "promedio_precision_rf": promedio_precision_rf,
        "promedio_rmse_svm": promedio_rmse_svm,
        "promedio_mae_svm": promedio_mae_svm,
        "promedio_precision_svm": promedio_precision_svm,
    }

    return JSONResponse(content=result)

if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI application using uvicorn
    uvicorn.run(app)

