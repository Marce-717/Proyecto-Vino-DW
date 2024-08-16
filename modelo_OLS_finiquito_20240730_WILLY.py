# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""


"""

Don Lalo buen dia!!

Necesitamos colocar Metabisulfito (SO2) al vino Syrah que esta en el pallet de 1000 L y 200 L mas:
    
    - Estanques 1000 litros: aplicar 39 gramos de S02
    - estanques de 200 litros: aplicar 8 gramos de SO2
    
Ademas colocar Ácido Tartarico (AT)a los mismos estanques:
    
    - Estanques 1000 litros: aplicar 960 gramos de AT
    - estanques de 200 litros: aplicar 192 gramos de AT    

Lo llamo por teléfono para conversar sobre este tema.


Gracias!


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

d = pd.read_excel('C:/Users/marce/Proyecto vino/entrada_modelo_ols_precio7_20240711.xlsx', index_col=0)

data = d[['CANTIDAD_sum', 'precio', 'precio_2', 'precio_7', 'precio_12','GDP_USA', 'precio_30']].copy()

# Seleccionar las colum
# Cargar el archivo Excelnas relevantes
independent_vars = ['precio_7', 'GDP_USA']
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
# Quiero dejar en forma de tabla los rsultados de los modelos
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

# Evaluación del modelo
def evaluate_model(y_true, y_pred_log, model_name):
    mse = mean_squared_error(y_true, y_pred_log)
    r2 = r2_score(y_true, y_pred_log)
    mape = mean_absolute_percentage_error(y_true, y_pred_log)
    print(f"{model_name} (log) - MSE: {mse:.2f}, R2: {r2:.2f}, MAPE: {mape:.2f}")  
 


# Transformar las predicciones de vuelta a la escala original
y_pred_sm = np.exp(y_pred_log_sm) - 1
predictions = {name: np.exp(y_pred_log) - 1 for name, y_pred_log in predictions_log.items()}

# Evaluación del modelo en escala original
def evaluate_model_original(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"{model_name} (original) - MSE: {mse:.2f}, R2: {r2:.2f}, MAPE: {mape:.2f}")

evaluate_model_original(np.exp(y_test) - 1, y_pred_sm, "OLS Regression")
for name, y_pred in predictions.items():
    evaluate_model_original(np.exp(y_test) - 1, y_pred, name)

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

print("RESULTADO MODELO OLS ")
print(summary_table)


