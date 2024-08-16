# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 19:40:07 2024

@author: marce
"""

import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score 

# Cargar los datos
d = pd.read_excel('C:/Users/marce/Proyecto vino/entrada_modelo_ols_precio7_20240711.xlsx', index_col=0)
data = d[['CANTIDAD_sum', 'precio', 'precio_2', 'precio_7', 'precio_12','GDP_USA', 'precio_30']].copy()

# Seleccionar variables independientes y dependiente
independent_vars = ['precio_7','GDP_USA']
dependent_var = 'CANTIDAD_sum'

# Aplicar logaritmo natural
data_log = data.copy()
data_log[independent_vars] = np.log(data_log[independent_vars] + 1)  # Evitar log(0)
data_log[dependent_var] = np.log(data_log[dependent_var] + 1)

# División de los datos
X = data_log[independent_vars]
y = data_log[dependent_var]

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
    "KNN": KNeighborsRegressor(n_neighbors=5)
}

predictions_log = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions_log[name] = model.predict(X_test)

# Transformar las predicciones de vuelta a la escala original
y_pred_sm = np.exp(y_pred_log_sm) - 1
predictions = {name: np.exp(y_pred_log) - 1 for name, y_pred_log in predictions_log.items()}

# Función para evaluar y formatear los resultados en tabla
def evaluate_models(y_true, predictions, model_names):
    results = []
    for model_name, y_pred in predictions.items():
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        results.append({
            'Model': model_name,
            'MSE': f"{mse:.2f}",
            'R2': f"{r2:.2f}",
            'MAPE': f"{mape:.2f}"
        })
    return pd.DataFrame(results)

# Evaluar todos los modelos
results_df = evaluate_models(np.exp(y_test) - 1, predictions, models.keys())

# Evaluar modelo OLS
ols_results = evaluate_models(np.exp(y_test) - 1, {"OLS Regression": y_pred_sm}, ["OLS Regression"])
results_df = pd.concat([ols_results, results_df])

# Mostrar resultados en tabla
print("RESULTADOS DEL MODELO (Escala Original):")
print(results_df.to_string(index=False))

# Visualización de los resultados
plt.figure(figsize=(10, 6))
plt.scatter(np.exp(y_test) - 1, y_pred_sm, label='OLS Regression')
for name, y_pred in predictions.items():
    plt.scatter(np.exp(y_test) - 1, y_pred, label=name)
plt.plot([min(np.exp(y_test) - 1), max(np.exp(y_test) - 1)], 
         [min(np.exp(y_test) - 1), max(np.exp(y_test) - 1)], 
         color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Model Comparison')
plt.legend()
plt.show()

# Interpretación de resultados del modelo OLS
summary_table = model_sm.summary2().tables[1]
summary_table['Coef.'] = summary_table['Coef.'].apply(lambda x: f"{x:.2f}")
summary_table['Std.Err.'] = summary_table['Std.Err.'].apply(lambda x: f"{x:.2f}")
summary_table['t'] = summary_table['t'].apply(lambda x: f"{x:.2f}")
summary_table['P>|t|'] = summary_table['P>|t|'].apply(lambda x: f"{x:.2f}")

# print("\nRESULTADO MODELO OLS:")
# print(summary_table)

############################### VALIDACION CRUZADA ##########################################
print("")
print("Validación cruzada")

# Definir el modelo con los mejores parámetros encontrados o iniciales
model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)

# Realizar la validación cruzada
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

# Calcular el promedio y la desviación estándar del RMSE
rmse_scores = np.sqrt(-cv_scores)
mean_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)

print(f"RMSE promedio: {mean_rmse}")
print(f"Desviación estándar del RMSE: {std_rmse}")

######################## valacion cruzada grafica #####################################

import matplotlib.pyplot as plt

# Entrenar el modelo si aún no está entrenado
model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
model.fit(X, y)

# Obtener las importancias de las características
feature_importances = model.feature_importances_
features = X.columns

# Crear un DataFrame para facilitar el plot
importances_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importances_df = importances_df.sort_values(by='Importance', ascending=False)

# Gráfico de barras de importancia de características
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importances_df)
plt.title('Importancia de las características')
plt.xlabel('Importancia')
plt.ylabel('Características')
plt.show()

############################# Grafica de puntos estimados y reales #####################################


# Predecir en el conjunto de prueba si aún no lo has hecho
y_pred = model.predict(X_test)

# Crear un gráfico de dispersión de los valores reales vs. predichos
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Predicciones')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Línea ideal')
plt.xlabel('Valores reales (log transformados)')
plt.ylabel('Valores predichos (log transformados)')
plt.title('Valores reales vs. Predicciones del modelo')
plt.legend()
plt.grid(True)
plt.show()


################## Proyecciones ################################################
print("")

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
proyecciones_cajas = np.exp(proyecciones_log) - 1
proyecciones_litros = np.exp(proyecciones_log) - 1
proyecciones_litros = proyecciones_litros * 9

# Redondear los resultados y formatear con separadores de miles
proyecciones_cajas_formatted = [f"{int(round(p)):,}" for p in proyecciones_cajas]
proyecciones_litros_formatted = [f"{int(round(p)):,}" for p in proyecciones_litros]

print("Proyecciones de cantidad numero de cajas:", proyecciones_cajas_formatted)
print("Proyecciones de cantidad numero de litros:", proyecciones_litros_formatted)

