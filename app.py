import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Cargar el dataset
cleaned_combined_df = pd.read_csv('combined_df_dataset.csv')

# Asegúrate de incluir el nombre del activo en las características
X = cleaned_combined_df[['Asset Number', 'Asset Age', 'Time Since Last Maintenance', 'Total Spare Parts Cost', 'Failure Count', 'Avg Time Between Maintenance', 'Month']]
y = cleaned_combined_df[['Time to Next Failure','Total Spare Parts Cost', 'Failure Code']]

from sklearn.model_selection import train_test_split

# Realizar el train/test split
X_train, X_test, y_train, y_test = train_test_split(X.drop(columns=['Asset Number']), y, test_size=0.3, random_state=42)

# Load predictions from the CSV file
df_predictions = pd.read_csv('predictions.csv')

# Extract the data and assign to variables
y_pred_f = df_predictions['Predicted Time to Next Failure'].values
y_pred_failure_type = df_predictions['Predicted Failure Type'].values
y_pred_cost = df_predictions['Predicted Cost'].values

# Crear el DataFrame de resultados incluyendo el nombre del activo
df_results = X_test.copy()
df_results['Asset Number'] = X.loc[X_test.index, 'Asset Number']  # Añadir la columna 'Asset Number' al DataFrame de resultados
df_results['Predicted Time to Next Failure'] = y_pred_f
df_results['Predicted Failure Type'] = y_pred_failure_type
df_results['Predicted Cost'] = y_pred_cost

df_results['Actual Time to Next Failure'] = y_test['Time to Next Failure'].values
df_results['Actual Failure Type'] = y_test["Failure Code"].values
df_results['Actual Cost'] = y_test["Total Spare Parts Cost"].values

# Filtrar las máquinas que fallarán en menos de 1 mes
df_fail_1m = df_results[df_results['Predicted Time to Next Failure'] <= 30]

# Filtrar las máquinas que fallarán en menos de 3 meses pero más de 1 mes
df_fail_1_3m = df_results[(df_results['Predicted Time to Next Failure'] > 30) & 
                          (df_results['Predicted Time to Next Failure'] <= 90)]

# Título del Dashboard
st.title("Predictive Maintenance Dashboard")

# Calcular KPIs
total_machines = df_results['Asset Number'].nunique()
machines_at_risk_1m = df_fail_1m['Asset Number'].nunique()
machines_at_risk_3m = df_fail_1_3m['Asset Number'].nunique()
total_predicted_cost = df_results['Predicted Cost'].sum()

# Crear pestañas
tab1, tab2, tab3, tab4 = st.tabs(["Historial y Activos", "Análisis y Tendencias", "Alertas", "Exportar Datos"])

with tab2:

    # Seleccionar un activo específico y mostrar su historial de mantenimiento
    selected_asset = st.selectbox("Selecciona un activo:", df_results['Asset Number'].unique())
    maintenance_history = cleaned_combined_df[cleaned_combined_df['Asset Number'] == selected_asset]

    st.subheader(f"Historial de Mantenimiento para {selected_asset}")
    st.write(maintenance_history[['Creation Date & Time', 'Failure Code', 'Total Spare Parts Cost']])

    # Mostrar la información del activo seleccionado
    st.subheader(f"Información del Activo: {selected_asset}")
    asset_info = df_results[df_results['Asset Number'] == selected_asset]
    st.write(asset_info)

    # Gráfico de barras del tipo de falla predicho para este activo
    st.subheader(f"Tipo de Falla Predicho para {selected_asset}")
    fig, ax = plt.subplots()
    sns.countplot(x='Predicted Failure Type', data=asset_info, ax=ax)
    ax.set_title(f"Predicted Failure Type for {selected_asset}")
    ax.set_xlabel("Failure Type")
    ax.set_ylabel("Count")
    st.pyplot(fig)

with tab1:

    st.header("Indicadores Clave de Rendimiento (KPI)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Máquinas", total_machines)
    col2.metric("Riesgo de Falla en 1 Mes", machines_at_risk_1m)
    col3.metric("Riesgo de Falla en 1-3 Meses", machines_at_risk_3m)
    col4.metric("Costo Total Estimado", f"${total_predicted_cost:,.2f}")

    col1, col2 = st.columns(2)

    with col1:
        # Mostrar los datos en una tabla interactiva
        st.subheader("Machine Predictions")
        st.dataframe(df_results[['Asset Number', 'Predicted Time to Next Failure', 'Actual Time to Next Failure','Predicted Failure Type', 'Actual Failure Type', 'Predicted Cost', 'Actual Cost']])

        # Mostrar los resultados en Streamlit
        st.subheader("Máquinas que fallarán en menos de 1 mes")
        st.dataframe(df_fail_1m[['Asset Number', 'Predicted Time to Next Failure', 'Predicted Failure Type', 'Predicted Cost']])

        st.subheader("Máquinas que fallarán en 1 a 3 meses")
        st.dataframe(df_fail_1_3m[['Asset Number', 'Predicted Time to Next Failure', 'Predicted Failure Type', 'Predicted Cost']])

        # Comparación de Costos Predichos y Reales
        st.subheader("Comparación de Costos Predichos y Reales")
        fig, ax = plt.subplots()
        ax.scatter(df_results['Actual Cost'], df_results['Predicted Cost'])
        ax.plot([df_results['Actual Cost'].min(), df_results['Actual Cost'].max()],
                [df_results['Actual Cost'].min(), df_results['Actual Cost'].max()],
                'r--')  # Línea de referencia
        ax.set_xlabel("Actual Cost")
        ax.set_ylabel("Predicted Cost")
        ax.set_title("Actual vs Predicted Cost")
        st.pyplot(fig)

    with col2:
        # Guardar los resultados en archivos CSV
        df_fail_1m.to_csv('machines_failing_in_1_month.csv', index=False)
        df_fail_1_3m.to_csv('machines_failing_in_1_to_3_months.csv', index=False)

        # Matriz de Confusión para Tipo de Falla
        st.subheader("Matriz de Confusión para Tipo de Falla")
        conf_matrix = confusion_matrix(df_results['Actual Failure Type'], df_results['Predicted Failure Type'])
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
        disp.plot(cmap=plt.cm.Blues)
        st.pyplot(plt)

        # Gráfico de barras de los tipos de fallas predichos
        fig, ax = plt.subplots()
        sns.countplot(x='Predicted Failure Type', data=df_results, ax=ax)
        ax.set_title("Distribution of Predicted Failure Types")
        ax.set_xlabel("Failure Type")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # Análisis de tendencia de costos
        st.subheader("Tendencia de Costos de Mantenimiento a lo Largo del Tiempo")

        # Asegúrate de que 'Creation Date & Time' esté en formato datetime
        cleaned_combined_df['Creation Date & Time'] = pd.to_datetime(cleaned_combined_df['Creation Date & Time'])

        # Crear la columna 'Month-Year' y convertirla a string
        df_results['Month-Year'] = cleaned_combined_df['Creation Date & Time'].dt.to_period('M').astype(str)

        # Agrupar los datos por 'Month-Year' y calcular el costo total predicho por mes
        cost_trend = df_results.groupby('Month-Year')['Predicted Cost'].sum().reset_index()

        # Graficar la tendencia de costos
        fig, ax = plt.subplots()
        sns.lineplot(x='Month-Year', y='Predicted Cost', data=cost_trend, ax=ax)
        ax.set_title("Tendencia de Costos de Mantenimiento")
        ax.set_xlabel("Mes-Año")
        ax.set_ylabel("Costo Estimado")
        ax.tick_params(axis='x', rotation=45)  # Rotar las etiquetas del eje x para mayor claridad
        st.pyplot(fig)

        # Análisis de distribución de tipos de fallas
        st.subheader("Distribución de Tipos de Fallas")
        failure_distribution = df_results['Predicted Failure Type'].value_counts().reset_index()
        failure_distribution.columns = ['Failure Type', 'Count']

        fig, ax = plt.subplots()
        sns.barplot(x='Failure Type', y='Count', data=failure_distribution, ax=ax)
        ax.set_title("Distribución de Tipos de Fallas")
        ax.set_xlabel("Tipo de Falla")
        ax.set_ylabel("Cantidad")
        st.pyplot(fig)

        # Mapa de calor de correlaciones
        st.subheader("Mapa de Calor de Correlaciones")

        # Filtrar solo las columnas numéricas
        numeric_cols = df_results.select_dtypes(include=['float64', 'int64']).columns
        corr_matrix = df_results[numeric_cols].corr()

        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Mapa de Calor de Correlaciones")
        st.pyplot(fig)

with tab3:
    # Sistema de alertas
    st.subheader("Alertas Automáticas")
    critical_alerts = df_results[df_results['Predicted Time to Next Failure'] <= 7]
    if not critical_alerts.empty:
        st.error(f"¡Atención! {len(critical_alerts)} máquinas en riesgo crítico de falla en menos de una semana.")
        st.write(critical_alerts[['Asset Number', 'Predicted Time to Next Failure', 'Predicted Failure Type', 'Predicted Cost']])
    else:
        st.success("No hay máquinas en riesgo crítico de falla en la próxima semana.")

with tab4:
    # Opción para descargar los resultados filtrados
    st.subheader("Exportar Datos")
    st.download_button("Descargar Máquinas en Riesgo (1 Mes)", df_fail_1m.to_csv(index=False).encode('utf-8'), "machines_failing_in_1_month.csv", "text/csv")
    st.download_button("Descargar Máquinas en Riesgo (1-3 Meses)", df_fail_1_3m.to_csv(index=False).encode('utf-8'), "machines_failing_in_1_to_3_months.csv", "text/csv")

# Filtros adicionales para la interactividad
st.sidebar.subheader("Filtros Avanzados")

# Agrega un key único a cada widget
date_range = st.sidebar.date_input("Rango de Fechas", [], key="date_range")
selected_failure_type = st.sidebar.multiselect("Selecciona Tipo de Falla", df_results['Predicted Failure Type'].unique(), key="failure_type_filter")

filtered_results = df_results
if date_range:
    filtered_results = filtered_results[(filtered_results['Creation Date & Time'] >= date_range[0]) & (filtered_results['Creation Date & Time'] <= date_range[1])]
if selected_failure_type:
    filtered_results = filtered_results[filtered_results['Predicted Failure Type'].isin(selected_failure_type)]

st.sidebar.write(filtered_results)
