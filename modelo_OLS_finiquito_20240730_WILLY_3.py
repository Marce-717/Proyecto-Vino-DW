# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 16:29:02 2024

@author: marce
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# Cargar el archivo Excel
d = pd.read_excel('C:/Users/marce/Proyecto vino/entrada_modelo_ols_precio7_20240711.xlsx', index_col=0)

data = d[['CANTIDAD_sum', 'precio', 'precio_2', 'precio_7', 'precio_12','GDP_USA', 'precio_30']].copy()

# Seleccionar las columnas relevantes
independent_vars = ['precio_7', 'GDP_USA']
dependent_var = 'CANTIDAD_sum'

# Aplicar logaritmo natural
data_log = data.copy()
data_log[independent_vars] = np.log(data_log[independent_vars] + 1)  # Evitar log(0)
data_log[dependent_var] = np.log(data_log[dependent_var] + 1)

# División de los datos
X = data_log[independent_vars]
y = data_log[dependent_var]

# Repartcicion de la data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo de regresión lineal
X_train_sm = sm.add_constant(X_train)  # Agregar la constante para la intersección
X_test_sm = sm.add_constant(X_test)

model_sm = sm.OLS(y_train, X_train_sm).fit()
y_pred_log_sm = model_sm.predict(X_test_sm)

# Modelos adicionales
models = {
    "CatBoost": CatBoostRegressor(silent=True, random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=3)
}

predictions_log = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions_log[name] = model.predict(X_test)

# Evaluación del modelo
def evaluate_model(y_true, y_pred_log, model_name):
    mse = mean_squared_error(y_true, y_pred_log)
    r2 = r2_score(y_true, y_pred_log)
    mape = mean_absolute_percentage_error(y_true, y_pred_log)
    return {"Model": model_name, "MSE (log)": round(mse, 2), "R2 (log)": round(r2, 2), "MAPE (log)": round(mape, 2)}

# Guardar resultados de evaluación en logaritmo
results_log = [evaluate_model(y_test, y_pred_log_sm, "OLS Regression")]
for name, y_pred_log in predictions_log.items():
    results_log.append(evaluate_model(y_test, y_pred_log, name))

# Crear DataFrame con los resultados en logaritmo
results_log_df = pd.DataFrame(results_log)

# Transformar las predicciones de vuelta a la escala original
y_pred_sm = np.exp(y_pred_log_sm) - 1
predictions = {name: np.exp(y_pred_log) - 1 for name, y_pred_log in predictions_log.items()}

# Evaluación del modelo en escala original
def evaluate_model_original(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"Model": model_name, "MSE (original)": round(mse, 2), "R2 (original)": round(r2, 2), "MAPE (original)": round(mape, 2)}

# Guardar resultados de evaluación en escala original
results_original = [evaluate_model_original(np.exp(y_test) - 1, y_pred_sm, "OLS Regression")]
for name, y_pred in predictions.items():
    results_original.append(evaluate_model_original(np.exp(y_test) - 1, y_pred, name))

# Crear DataFrame con los resultados en escala original
results_original_df = pd.DataFrame(results_original)

# Mostrar resultados en forma de tablas
print("\nDesempeño de modelos con datos de entrenamiento - Resultados en escala original:")
print(results_original_df)
print("")
print("Desempeño de modelos con datos de entrenamiento - Resultados en escala logarítmica:")
print(results_log_df)

# Visualización de los resultados
plt.figure(figsize=(10, 6))
plt.scatter(np.exp(y_test) - 1, y_pred_sm, label='OLS Regression')
for name, y_pred in predictions.items():
    plt.scatter(np.exp(y_test) - 1, y_pred, label=name)
plt.plot([min(np.exp(y_test) - 1), max(np.exp(y_test) - 1)], [min(np.exp(y_test) - 1), max(np.exp(y_test) - 1)], color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Model Comparison')
plt.legend()
plt.show()

# Interpretación de resultados con dos decimales
summary_table = model_sm.summary2().tables[1]
summary_table['Coef.'] = summary_table['Coef.'].apply(lambda x: f"{x:.2f}")
summary_table['Std.Err.'] = summary_table['Std.Err.'].apply(lambda x: f"{x:.2f}")
summary_table['t'] = summary_table['t'].apply(lambda x: f"{x:.2f}")
summary_table['P>|t|'] = summary_table['P>|t|'].apply(lambda x: f"{x:.2f}")

####################### Proyecciones ######################


# Ejemplo de nuevos datos para los que queremos hacer proyecciones
# Reemplaza estos datos con los nuevos datos reales
new_data = pd.DataFrame({
    'precio_7': [175, 174, 176],
    'GDP_USA': [62000, 62800, 62620]
})

# Asegurarse de aplicar la misma transformación (log) que se aplicó a los datos de entrenamiento
new_data_log = np.log(new_data + 1)

# Realizar las proyecciones con el modelo entrenado
proyecciones_log = model.predict(new_data_log)

# Transformar de vuelta desde la escala logarítmica a la escala original
proyecciones = np.exp(proyecciones_log) - 1

print("Proyecciones de cantidad:")
print(proyecciones)


