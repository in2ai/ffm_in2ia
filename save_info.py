import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Cargar los modelos entrenados
model_f = joblib.load('model_f.pkl')
model_failure_type = joblib.load('model_failure_type.pkl')
model_cost = joblib.load('model_cost.pkl')

# Cargar el dataset
cleaned_combined_df = pd.read_csv('combined_df_dataset.csv')

# Asegúrate de incluir el nombre del activo en las características
X = cleaned_combined_df[['Asset Number', 'Asset Age', 'Time Since Last Maintenance', 'Total Spare Parts Cost', 'Failure Count', 'Avg Time Between Maintenance', 'Month']]
y = cleaned_combined_df[['Time to Next Failure','Total Spare Parts Cost', 'Failure Code']]

from sklearn.model_selection import train_test_split

# Realizar el train/test split
X_train, X_test, y_train, y_test = train_test_split(X.drop(columns=['Asset Number']), y, test_size=0.3, random_state=42)

# Realizar las predicciones
y_pred_f = model_f.predict(X_test)
y_pred_failure_type_encoded = model_failure_type.predict(X_test)
y_pred_cost = model_cost.predict(X_test)

# Decodificar las predicciones para mostrar los tipos de fallas reales
label_encoder = LabelEncoder()
label_encoder.fit(cleaned_combined_df['Failure Code'])
y_pred_failure_type = label_encoder.inverse_transform(y_pred_failure_type_encoded)


# Save predictions to a DataFrame
df_predictions = pd.DataFrame({
    'Asset Number': X.loc[X_test.index, 'Asset Number'],
    'Predicted Time to Next Failure': y_pred_f,
    'Predicted Failure Type': y_pred_failure_type,
    'Predicted Cost': y_pred_cost
})

# Save predictions to a CSV file
df_predictions.to_csv('predictions.csv', index=False)