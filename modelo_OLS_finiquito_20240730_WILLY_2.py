# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
# Como experto en programación Phyton 3 y matematico estadistico solucionar este Warning
"""

Hipótesis: "El procesos exportador chileno se ve influenciado medianamente por hábitos de consumo de los principales mercados norteamericanos y Europeos."

Modelo de regresión lineal OLS:
    y = Cantidada exportada de vino
    
Variables independientes (x1 a x5):
    x1 = precio: Average Price: Wine, Red and White Table, All Sizes, Any Origin (Cost per 1 Liter/33.8 Ounces) in U.S. City Average.
    All grape based red, white and rose wines for all sizes, packaging typesAverage consumer prices are calculated for household fuel, motor fuel, and food items from prices collected for the Consumer Price Index (CPI). 
    Average prices are best used to measure the price level in a particular month, not to measure price change over time. It is more appropriate to use CPI index values for the particular item categories to measure price change.
    and origins. 
    Excludes sparkling, dessert, and other fruit wines. Excludes shipping costs on wines sold directly from wineries and distributors.
    
    x2 = precio_2: Harmonized Index of Consumer Prices: Wine for Euro area (20 countries) (CP0212MI15EA20M086NEST)
    x3 = precio_12: Harmonized Index of Consumer Prices: Wine for European Economic Area (EEA18-2004, EEA28-2006, EEA30); (CP0212E3CCM086NEST)
    x4 = precio_30: Industrial Production: Manufacturing: Non-Durable Goods: Food, Beverage, and Tobacco (NAICS = 311,2) (IPG311A2SQ)
    x5 = precio_7:  Producer Price Index by Industry: Wineries (PCU312130312130). Not Seasonally Adjusted. https://fred.stlouisfed.org/series/PCU312130312130
    x6 = GDP_USA: Real gross domestic product per capita (A939RX0Q048SBEA)    

N_exportada = 2.045.788
N_codigos arancelarios = 11

"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Cargar el archivo Excel
data = pd.read_excel('C:/Users/marce/Proyecto vino/entrada_modelo_ols_precio7_20240711.xlsx', index_col=0)
k = data.groupby(['FECHA'])['CANTIDAD_sum'].sum().reset_index()
#print(data.info())
print("")
################################################## OLS NORMAL ################################
# Con ambas variables aplicandole logaritmo natural

df = data[['CANTIDAD_sum','precio','precio_2','precio_7','precio_12','precio_30', 'GDP_USA']].copy()
#print(df)
# # Aplicar logaritmo natural a las variables
df_log = np.log(df)

# # Dividir los datos en variables dependientes e independientes
y_log = df_log['CANTIDAD_sum']
X = df_log[['precio_7','GDP_USA']]

# # Agregar una constante al conjunto de variables independientes (intercepto)
X = sm.add_constant(X)

# # Crear el modelo de regresión lineal
modelo_OLS = sm.OLS(y_log, X).fit()

# # Imprimir un resumen del modelo
print('Resumen del Modelo OLS para la variable (y): \n', modelo_OLS.summary())



############################# MODELO OLS SOLAMENTE y LOG CONJUNTO DE ENTRENAMIENTO #########################################

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Entrenar el modelo OLS con el logaritmo natural de la variable dependiente
ols_model_log = sm.OLS(y_train, X_train).fit()

print(ols_model_log.summary())
print("")
####################### PREDICCIONES ######################################################################

# Predicciones OLS con el logaritmo natural de la variable dependiente
y_pred_log = ols_model_log.predict(X_test)

# Transformar las predicciones de vuelta a la escala original
y_pred = np.exp(y_pred_log)
y_test_exp = np.exp(y_test)

# Evaluación del modelo OLS con el logaritmo natural de la variable dependiente
mse_ols_log = mean_squared_error(y_test_exp, y_pred)
r2_ols_log = r2_score(y_test_exp, y_pred)
mape_ols_log = mean_absolute_percentage_error(y_test_exp, y_pred)

print(f"OLS with log-transformed dependent variable R^2: {r2_ols_log}")
print(f"OLS with log-transformed dependent variable MSE: {mse_ols_log}")
print(f"OLS with log-transformed dependent variable MAPE: {mape_ols_log}")
print("")

########################### ENTRENAMIENTO #########################################

print("Evaluación del modelo OLS con el conjunto de prueba y entrenamiento")
# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo OLS
X_train_ols = sm.add_constant(X_train)  # Agregar una constante al modelo
ols_model = sm.OLS(y_train, X_train_ols).fit()

print(ols_model.summary())
print("")

print("Predicciones con el conjunto de prueba")
# Predicciones OLS
X_test_ols = sm.add_constant(X_test)
y_pred_ols = ols_model.predict(X_test_ols)

# Evaluación del modelo OLS
mse_ols = mean_squared_error(y_test, y_pred_ols)
r2_ols = r2_score(y_test, y_pred_ols)
mape_ols = mean_absolute_percentage_error(y_test, y_pred_ols)

print("")
print(f"OLS R-Adjusted: {r2_ols}")
print(f"OLS MSE: {mse_ols}")
print(f"OLS MAPE: {mape_ols}")

# Modelo CatBoost
catboost_model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, silent=True)
catboost_model.fit(X_train, y_train)
y_pred_catboost = catboost_model.predict(X_test)

# Evaluación del modelo CatBoost
mse_catboost = mean_squared_error(y_test, y_pred_catboost)
r2_catboost = r2_score(y_test, y_pred_catboost)
mape_catboost = mean_absolute_percentage_error(y_test, y_pred_catboost)
print("")
print(f"CatBoost R-Adjusted: {r2_catboost}")
print(f"CatBoost MSE: {mse_catboost}")
print(f"CatBoost MAPE: {mape_catboost}")

# Modelo XGBoost
xgboost_model = XGBRegressor(n_estimators=1000, learning_rate=0.1, max_depth=6)
xgboost_model.fit(X_train, y_train)
y_pred_xgboost = xgboost_model.predict(X_test)

# Evaluación del modelo XGBoost
mse_xgboost = mean_squared_error(y_test, y_pred_xgboost)
r2_xgboost = r2_score(y_test, y_pred_xgboost)
mape_xgboost = mean_absolute_percentage_error(y_test, y_pred_xgboost)
print("")
print(f"XGBoost R-Adjusted: {r2_xgboost}")
print(f"XGBoost MSE: {mse_xgboost}")
print(f"XGBoost MAPE: {mape_xgboost}")

# Modelo RandomForest
rf_model = RandomForestRegressor(n_estimators=1000, max_depth=6, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluación del modelo RandomForest
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf)
print("")
print(f"RandomForest R-Adjusted: {r2_rf}")
print(f"RandomForest MSE: {mse_rf}")
print(f"RandomForest MAPE: {mape_rf}")

##############################################

# Valores reales y proyectados
y_test_exp = np.exp(y_test)  # Valores reales (transformados de nuevo a la escala original)
y_pred = np.exp(y_pred_log)  # Valores proyectados (transformados de nuevo a la escala original)


#################################
plt.figure(figsize=(9, 7))
plt.plot(y_test_exp.values, label='Valores Reales', marker='o')
plt.plot(y_pred, label='Valores Proyectados', marker='x')
plt.xlabel('Índice')
plt.ylabel('Cantidad')
plt.title('Gráfico de Líneas de Valores Reales vs. Proyectados')
plt.legend()
plt.show()

###########################################

print("")
# Supongamos que future_data es un DataFrame que contiene las variables independientes para el pronóstico
# Por ejemplo:
future_data = pd.DataFrame({
    #'precio': [12.852, 12.524, 13.000],
    #'precio_30': [95, 100, 105],
    #'precio_2': [102, 106, 112],
    #'precio_12': [90, 95, 100],
    'GDP_USA': [62000, 62800, 62620],
    #'Pcaja_mean': [5, 5.1, 5.2],
    'precio_7': [175, 174, 176]
})

# Agregar una constante a las variables independientes
future_data = sm.add_constant(future_data)

# Hacer el pronóstico usando el modelo de regresión entrenado
future_predictions_log = ols_model_log.predict(future_data)

# Transformar las predicciones de vuelta a la escala original
future_predictions = np.exp(future_predictions_log)
print("Predicciones para el periodo próximo de 3 meses.")
print("")
# Mostrar las predicciones
print(f"Future Predictions: \n {future_predictions}")
print("")
data.drop(['precio_30','precio_12','FECHA','precio','precio_2'], axis= 1, inplace=True)
print(data.describe())

#######################################################

