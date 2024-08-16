# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 20:08:45 2024

@author: marce
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Cargar los datos
d = pd.read_excel('C:/Users/marce/Proyecto vino/entrada_modelo_ols_precio7_20240711.xlsx', index_col=0)
data = d[['CANTIDAD_sum', 'precio', 'precio_2', 'precio_7', 'precio_12','GDP_USA', 'precio_30']].copy()

# Selección de variables
independent_vars = ['precio_7', 'GDP_USA'] 
dependent_var = 'CANTIDAD_sum'

# Aplicar logaritmo natural
data_log = data.copy()
data_log[independent_vars] = np.log(data_log[independent_vars] + 1)
data_log[dependent_var] = np.log(data_log[dependent_var] + 1)

# División de los datos
X = data_log[independent_vars]
y = data_log[dependent_var]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir los modelos
models = {
    "OLS Regression": sm.OLS(y_train, sm.add_constant(X_train)),
    "CatBoost": CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, silent=True),
    "XGBoost": XGBRegressor(n_estimators=1000, learning_rate=0.1, max_depth=6, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=1000, max_depth=6, random_state=42)
}

# Evaluación de los modelos
results = []

for name, model in models.items():
    if name == "OLS Regression":
        model = model.fit()
        y_pred = model.predict(sm.add_constant(X_test))
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    results.append({
        'Model': name,
        'MSE': f"{mse:.2f}",
        'R2': f"{r2:.2f}",
        'MAPE': f"{mape:.2f}"
    })

# Crear un DataFrame con los resultados
results_df = pd.DataFrame(results)

# Mostrar la tabla de resultados
print("Predicciones con el conjunto de prueba:")
print(results_df.to_string(index=False))
