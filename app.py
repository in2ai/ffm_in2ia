import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Load the Excel files
spare_issued_cm = pd.read_excel('Spare_Issued_CM.xlsx')
spare_issued_pm = pd.read_excel('Spare_Issued_PM.xlsx')
work_order_cm = pd.read_excel('Work_Order_CM.xlsx')
work_order_pm = pd.read_excel('Work_Order_PM.xlsx')

# Convert date columns to datetime
spare_issued_cm['ITEM ISSUE DATE'] = pd.to_datetime(spare_issued_cm['ITEM ISSUE DATE'])
spare_issued_pm['ITEM ISSUE DATE'] = pd.to_datetime(spare_issued_pm['ITEM ISSUE DATE'])
work_order_cm['Creation Date & Time'] = pd.to_datetime(work_order_cm['Creation Date & Time'])
work_order_pm['Creation Date & Time'] = pd.to_datetime(work_order_pm['Creation Date & Time'])


# Load the datasets
cleaned_combined_df = pd.read_csv('combined_df_dataset.csv')
df_predictions = pd.read_csv('predictions.csv')

# Define features and target variables
X = cleaned_combined_df[['Asset Number', 'Asset Age', 'Time Since Last Maintenance', 'Total Spare Parts Cost', 'Failure Count', 'Avg Time Between Maintenance', 'Month']]
y = cleaned_combined_df[['Time to Next Failure', 'Total Spare Parts Cost', 'Failure Code']]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X.drop(columns=['Asset Number']), y, test_size=0.3, random_state=42)

# Extract the data and assign to variables
y_pred_f = df_predictions['Predicted Time to Next Failure'].values
y_pred_failure_type = df_predictions['Predicted Failure Type'].values
y_pred_cost = df_predictions['Predicted Cost'].values

# Create the DataFrame of results including 'Asset Number'
df_results = X_test.copy()
df_results['Asset Number'] = X.loc[X_test.index, 'Asset Number']  # Add the 'Asset Number' column to the results DataFrame
df_results['Predicted Time to Next Failure'] = y_pred_f
df_results['Predicted Failure Type'] = y_pred_failure_type
df_results['Predicted Cost'] = y_pred_cost
df_results['Actual Time to Next Failure'] = y_test['Time to Next Failure'].values
df_results['Actual Failure Type'] = y_test['Failure Code'].values
df_results['Actual Cost'] = y_test['Total Spare Parts Cost'].values

# Ensure 'Creation Date & Time' is in datetime format
cleaned_combined_df['Creation Date & Time'] = pd.to_datetime(cleaned_combined_df['Creation Date & Time'])
df_results['Month-Year'] = cleaned_combined_df['Creation Date & Time'].dt.to_period('M').astype(str)

# Sort the DataFrame by 'Month-Year' and group by 'Asset Number' to take the last entry
df_results_sorted = df_results.sort_values(by='Month-Year')
latest_predictions = df_results_sorted.groupby('Asset Number').tail(1)

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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Historic","General information", "Specific information", "Alerts", "Export Data"])

with tab1:
    # Sidebar filters for interactivity, solo en esta pestaña
    st.header("Historic Data")
    # Date Range filter solo en la pestaña 'Historic'
    # Filtro de rango de fechas solo en la pestaña 'Historic'
    date_range = st.date_input("Date Range", [])

    # Convertir el rango de fechas a una lista si es necesario
    if isinstance(date_range, list) and len(date_range) == 2:
        start_date, end_date = date_range
    elif isinstance(date_range, list) and len(date_range) == 1:
        start_date = end_date = date_range[0]
    elif isinstance(date_range, pd.Timestamp):  # Para una sola fecha
        start_date = end_date = date_range
    else:
        start_date, end_date = None, None

    # Apply filters to the historical data
    filtered_cm_spare = spare_issued_cm
    filtered_pm_spare = spare_issued_pm
    filtered_cm_work = work_order_cm
    filtered_pm_work = work_order_pm

    # Apply date range filter if a range is selected
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_cm_spare = filtered_cm_spare[(filtered_cm_spare['ITEM ISSUE DATE'] >= pd.to_datetime(start_date)) & 
                                            (filtered_cm_spare['ITEM ISSUE DATE'] <= pd.to_datetime(end_date))]
        filtered_pm_spare = filtered_pm_spare[(filtered_pm_spare['ITEM ISSUE DATE'] >= pd.to_datetime(start_date)) & 
                                            (filtered_pm_spare['ITEM ISSUE DATE'] <= pd.to_datetime(end_date))]
        filtered_cm_work = filtered_cm_work[(filtered_cm_work['Creation Date & Time'] >= pd.to_datetime(start_date)) & 
                                            (filtered_cm_work['Creation Date & Time'] <= pd.to_datetime(end_date))]
        filtered_pm_work = filtered_pm_work[(filtered_pm_work['Creation Date & Time'] >= pd.to_datetime(start_date)) & 
                                            (filtered_pm_work['Creation Date & Time'] <= pd.to_datetime(end_date))]
    else:
        filtered_df_results = df_results

    # Aggregate the filtered data for spare issued
    filtered_cm_spare_summary = filtered_cm_spare.groupby(['ITEM NAME', 'MAJOR CATEGORY']).agg({
        'QUANTITY': 'sum',
        'TOTAL VALUE': 'sum'
    }).reset_index()

    filtered_pm_spare_summary = filtered_pm_spare.groupby(['ITEM NAME', 'MAJOR CATEGORY']).agg({
        'QUANTITY': 'sum',
        'TOTAL VALUE': 'sum'
    }).reset_index()

    col1, col2 = st.columns(2)
    figsize = (12, 8)

    with col1:
        st.subheader("Total Value of Items Issued (CM vs PM)")

        # Mostrar solo los 10 ítems principales según el TOTAL VALUE
        top_n = 50
        spare_issued_cm_summary_top = filtered_cm_spare_summary.nlargest(top_n, 'TOTAL VALUE')
        spare_issued_pm_summary_top = filtered_pm_spare_summary.nlargest(top_n, 'TOTAL VALUE')

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(spare_issued_cm_summary_top['ITEM NAME'], spare_issued_cm_summary_top['TOTAL VALUE'], label='CM', alpha=0.7)
        ax.bar(spare_issued_pm_summary_top['ITEM NAME'], spare_issued_pm_summary_top['TOTAL VALUE'], label='PM', alpha=0.7)
        ax.set_xlabel('Item Name', fontsize=14)
        ax.set_ylabel('Total Value', fontsize=14)
        ax.set_title('Total Value of Items Issued (CM vs PM)', fontsize=16)
        ax.legend()

        plt.xticks(rotation=90, ha='right', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Trend Over Time for Issued Items (CM vs PM)")

        def plot_trend_over_time(filtered_cm_spare, filtered_pm_spare):
            filtered_cm_spare['YearMonth'] = filtered_cm_spare['ITEM ISSUE DATE'].dt.to_period('M')
            filtered_pm_spare['YearMonth'] = filtered_pm_spare['ITEM ISSUE DATE'].dt.to_period('M')

            cm_trend = filtered_cm_spare.groupby('YearMonth').agg({'TOTAL VALUE': 'sum'}).reset_index()
            pm_trend = filtered_pm_spare.groupby('YearMonth').agg({'TOTAL VALUE': 'sum'}).reset_index()

            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(cm_trend['YearMonth'].astype(str), cm_trend['TOTAL VALUE'], label='CM', marker='o')
            ax.plot(pm_trend['YearMonth'].astype(str), pm_trend['TOTAL VALUE'], label='PM', marker='o')

            ax.set_xlabel('Year-Month', fontsize=12)
            ax.set_ylabel('Total Value Issued', fontsize=12)
            ax.set_title('Trend Over Time for Issued Items (CM vs PM)', fontsize=14)
            plt.xticks(rotation=45)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Llamar a la función para graficar la tendencia
        plot_trend_over_time(filtered_cm_spare, filtered_pm_spare)

    col1, col2, col3 = st.columns(3)

    with col1:
        # Count occurrences of each ASSET GROUP
        cm_asset_group = filtered_cm_spare['ASSET GROUP'].value_counts().reset_index()
        pm_asset_group = filtered_pm_spare['ASSET GROUP'].value_counts().reset_index()

        # Rename columns for clarity
        cm_asset_group.columns = ['Asset Group', 'Count_CM']
        pm_asset_group.columns = ['Asset Group', 'Count_PM']

        # Merge CM and PM data on 'Asset Group'
        merged_asset_group = pd.merge(cm_asset_group, pm_asset_group, on='Asset Group', how='outer').fillna(0)

        # Plot the distribution of asset groups for CM and PM
        def plot_asset_group_distribution():
            plt.figure(figsize=figsize)
            
            width = 0.35
            indices = range(len(merged_asset_group))
            
            plt.bar(indices, merged_asset_group['Count_CM'], width=width, label='CM', alpha=0.7)
            plt.bar([i + width for i in indices], merged_asset_group['Count_PM'], width=width, label='PM', alpha=0.7)
            
            plt.xlabel('Asset Group')
            plt.ylabel('Count')
            plt.title('Distribution of Asset Groups (CM vs PM)')
            plt.xticks([i + width/2 for i in indices], merged_asset_group['Asset Group'], rotation=90)
            plt.legend()
            plt.tight_layout()
            st.pyplot(plt)
            plt.close(fig)

        st.header("Asset Group Distribution")

        plot_asset_group_distribution()

    with col2:
        # Function to plot time taken to complete work orders
        def plot_time_taken_to_complete_work_orders(filtered_cm_work, filtered_pm_work):
            filtered_cm_work['Time to Complete'] = filtered_cm_work['Target Response Date & Time'] - filtered_cm_work['Creation Date & Time']
            filtered_pm_work['Time to Complete'] = filtered_pm_work['Target Response Date & Time'] - filtered_pm_work['Creation Date & Time']

            plt.figure(figsize=figsize)
            sns.boxplot(data=[filtered_cm_work['Time to Complete'].dt.total_seconds() / 3600, 
                            filtered_pm_work['Time to Complete'].dt.total_seconds() / 3600], 
                        palette="Set2", 
                        notch=True)
            
            plt.xlabel('Work Order Type')
            plt.ylabel('Time to Complete (Hours)')
            plt.title('Time Taken to Complete Work Orders (CM vs PM)')
            plt.xticks([0, 1], ['CM', 'PM'])
            plt.tight_layout()
            st.pyplot(plt)
            plt.close(fig)

        st.header("Time Taken to Complete")
        plot_time_taken_to_complete_work_orders(filtered_cm_work, filtered_pm_work)

    with col3:
        # Count occurrences of each Contractor
        contractor_cm = filtered_cm_work['Contractor'].value_counts().reset_index()
        contractor_pm = filtered_pm_work['Contractor'].value_counts().reset_index()

        # Rename columns for clarity
        contractor_cm.columns = ['Contractor', 'Count_CM']
        contractor_pm.columns = ['Contractor', 'Count_PM']

        # Merge CM and PM data on 'Contractor'
        merged_contractors = pd.merge(contractor_cm, contractor_pm, on='Contractor', how='outer').fillna(0)

        # Plot the performance of contractors for CM and PM
        def plot_contractor_performance():
            plt.figure(figsize=figsize)
            
            width = 0.35
            indices = range(len(merged_contractors))
            
            plt.bar(indices, merged_contractors['Count_CM'], width=width, label='CM', alpha=0.7)
            plt.bar([i + width for i in indices], merged_contractors['Count_PM'], width=width, label='PM', alpha=0.7)
            
            plt.xlabel('Contractor')
            plt.ylabel('Number of Work Orders')
            plt.title('Contractor Performance (CM vs PM)')
            plt.xticks([i + width/2 for i in indices], merged_contractors['Contractor'], rotation=90)
            plt.legend()
            plt.tight_layout()
            st.pyplot(plt)
            plt.close(fig)

        st.header("Contractor Performance")
        plot_contractor_performance()

    # Combine the Work Order PM and CM data for comprehensive analysis
    combined_work_order_df = pd.concat([filtered_pm_work, filtered_cm_work])

    # Count the number of work orders per asset
    work_order_counts = combined_work_order_df['Asset Number'].value_counts()

    # Preparing a dataframe to display the assets requiring the most maintenance
    most_maintenance_assets_df = work_order_counts.reset_index()
    most_maintenance_assets_df.columns = ['Asset Number', 'Work Order Count']

    ##########################################################

    # Create frequency analysis for both CM and PM work orders
    cm_work_order_frequency = filtered_cm_work.groupby(filtered_cm_work['Creation Date & Time'].dt.to_period('M')).size().sort_values(ascending=False)
    pm_work_order_frequency = filtered_pm_work.groupby(filtered_pm_work['Creation Date & Time'].dt.to_period('M')).size().sort_values(ascending=False)

    # Categorize issues in CM and PM by analyzing work order descriptions
    cm_issue_categories = filtered_cm_work['Description'].value_counts().head(10)
    pm_issue_categories = filtered_pm_work['Description'].value_counts().head(10)

    # Compare the number of CM vs. PM work orders
    cm_vs_pm_comparison = pd.DataFrame({
        'Work Order Type': ['Corrective Maintenance', 'Preventive Maintenance'],
        'Total Work Orders': [filtered_cm_work.shape[0], filtered_pm_work.shape[0]]
    })

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    # Display the top assets requiring the most maintenance
    with col1:
        st.dataframe(most_maintenance_assets_df.head(10))
    with col2:
        st.dataframe(cm_work_order_frequency.head())
    with col3:
        st.dataframe(pm_work_order_frequency.head())
    with col4:
        st.dataframe(cm_issue_categories)
    with col5:
        st.dataframe(pm_issue_categories)
    with col6:
        st.dataframe(cm_vs_pm_comparison)

    # Function to plot top 10 most common work order descriptions for CM
    def plot_top_cm_issue_categories(cm_issue_categories):

        plt.figure(figsize=(10, 6))
        cm_issue_categories.plot(kind='bar', color='blue')
        plt.title('Top 10 Most Common Work Order Descriptions (CM)')
        plt.xlabel('Description')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(plt)

    # Function to plot top 10 most common work order descriptions for PM
    def plot_top_pm_issue_categories(pm_issue_categories):

        plt.figure(figsize=(10, 6))
        pm_issue_categories.plot(kind='bar', color='green')
        plt.title('Top 10 Most Common Work Order Descriptions (PM)')
        plt.xlabel('Description')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(plt)

    col1, col2 = st.columns(2)

    with col1:
        st.header("Top 10 Most Common Work Order Descriptions (CM)")
        plot_top_cm_issue_categories(cm_issue_categories)

    with col2:

        st.header("Top 10 Most Common Work Order Descriptions (PM)")
        plot_top_pm_issue_categories(pm_issue_categories)


    #################################################################################

    # Count the frequency of each spare part in the CM dataset
    cm_spare_parts_frequency = filtered_cm_spare['ITEM NAME'].value_counts()

    # Count the frequency of each spare part in the PM dataset
    pm_spare_parts_frequency = filtered_pm_spare['ITEM NAME'].value_counts()

    # Calculate the total cost of each spare part in the CM dataset
    cm_spare_parts_cost = filtered_cm_spare.groupby('ITEM NAME')['TOTAL VALUE'].sum()

    # Calculate the total cost of each spare part in the PM dataset
    pm_spare_parts_cost = filtered_pm_spare.groupby('ITEM NAME')['TOTAL VALUE'].sum()

    # Merge CM spare parts with CM work orders
    cm_spare_asset_correlation = pd.merge(filtered_cm_spare, filtered_cm_work, left_on='WORK ORDER NUMBER', right_on='Work Order')

    # Merge PM spare parts with PM work orders
    pm_spare_asset_correlation = pd.merge(filtered_pm_spare, filtered_pm_work, left_on='WORK ORDER NUMBER', right_on='Work Order')

    # Analyze the correlation by grouping by asset and summing the total cost of spare parts used
    cm_asset_performance = cm_spare_asset_correlation.groupby('Asset Number')['TOTAL VALUE'].sum().sort_values(ascending=False)
    pm_asset_performance = pm_spare_asset_correlation.groupby('Asset Number')['TOTAL VALUE'].sum().sort_values(ascending=False)

    # Aggregate by Asset Number for CM data
    cm_asset_performance = cm_spare_asset_correlation.groupby('Asset Number').agg({
        'TOTAL VALUE': 'sum',
        'ITEM NAME': 'count'
    }).rename(columns={'ITEM NAME': 'Spare Parts Count'})

    # Aggregate by Asset Number for PM data
    pm_asset_performance = pm_spare_asset_correlation.groupby('Asset Number').agg({
        'TOTAL VALUE': 'sum',
        'ITEM NAME': 'count'
    }).rename(columns={'ITEM NAME': 'Spare Parts Count'})

    # Compare with the frequency of work orders for CM
    cm_work_order_frequency = filtered_cm_work['Asset Number'].value_counts()

    # Compare with the frequency of work orders for PM
    pm_work_order_frequency = filtered_pm_work['Asset Number'].value_counts()

    # Merge the asset performance with work order frequency for CM
    cm_correlation = pd.merge(cm_asset_performance, cm_work_order_frequency, left_index=True, right_index=True)
    cm_correlation = cm_correlation.rename(columns={cm_correlation.columns[-1]: 'Work Order Count'})  # Rename the last column to Work Order Count

    # Merge the asset performance with work order frequency for PM
    pm_correlation = pd.merge(pm_asset_performance, pm_work_order_frequency, left_index=True, right_index=True)
    pm_correlation = pm_correlation.rename(columns={pm_correlation.columns[-1]: 'Work Order Count'})  # Rename the last column to Work Order Count

    # Function to plot CM data correlation
    def plot_cm_correlation():
        plt.figure(figsize=(10, 6))
        plt.scatter(cm_correlation['Work Order Count'], cm_correlation['TOTAL VALUE'], alpha=0.7)
        plt.title('Corrective Maintenance: Spare Parts Cost vs. Work Order Frequency')
        plt.xlabel('Work Order Count')
        plt.ylabel('Total Spare Parts Cost')
        plt.grid(True)
        st.pyplot(plt)
        plt.close(fig)

    # Function to plot PM data correlation
    def plot_pm_correlation():
        plt.figure(figsize=(10, 6))
        plt.scatter(pm_correlation['Work Order Count'], pm_correlation['TOTAL VALUE'], alpha=0.7, color='orange')
        plt.title('Preventive Maintenance: Spare Parts Cost vs. Work Order Frequency')
        plt.xlabel('Work Order Count')
        plt.ylabel('Total Spare Parts Cost')
        plt.grid(True)
        st.pyplot(plt)
        plt.close(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.header("Corrective Maintenance: Spare Parts Cost vs. Work Order Frequency")
        plot_cm_correlation()

    with col2:
        st.header("Preventive Maintenance: Spare Parts Cost vs. Work Order Frequency")
        plot_pm_correlation()

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(cm_correlation.head())

    with col2:
        st.dataframe(pm_correlation.head())

########################################################

    # Function to plot the top 50 assets by spare parts count for CM
    def plot_top_50_cm_assets():
        plt.figure(figsize=(12, 14))  # Increase figure size to fit more labels
        cm_correlation['Spare Parts Count'].sort_values(ascending=False).head(50).sort_values().plot(kind='barh', color='blue')
        plt.title('Top 50 Assets by Spare Parts Count (CM)')
        plt.xlabel('Spare Parts Count')
        plt.ylabel('Asset Number')
        plt.xlim(0, cm_correlation['Spare Parts Count'].max() + 10)  # Set the x-axis limit to the max value across the dataset
        plt.xticks(fontsize=10)  # Adjust font size for better readability
        plt.tight_layout()
        st.pyplot(plt)

    # Function to plot the top 50 assets by spare parts count for PM
    def plot_top_50_pm_assets():
        plt.figure(figsize=(12, 14))  # Increase figure size to fit more labels
        pm_correlation['Spare Parts Count'].sort_values(ascending=False).head(50).sort_values().plot(kind='barh', color='orange')
        plt.title('Top 50 Assets by Spare Parts Count (PM)')
        plt.xlabel('Spare Parts Count')
        plt.ylabel('Asset Number')
        plt.xlim(0, pm_correlation['Spare Parts Count'].max() + 10)  # Set the x-axis limit to the max value across the dataset
        plt.xticks(fontsize=10)  # Adjust font size for better readability
        plt.tight_layout()
        st.pyplot(plt)

    col1, col2 = st.columns(2)

    with col1:
        # Display the plots in Streamlit
        st.header("Top 50 Assets by Spare Parts Count (CM)")
        plot_top_50_cm_assets()

    with col2:
        st.header("Top 50 Assets by Spare Parts Count (PM)")
        plot_top_50_pm_assets()


####################################################################################

    # Assuming the datasets are already loaded into work_order_cm_df and work_order_pm_df

    # Calculate time difference between target and actual completion for CM
    filtered_cm_work['Completion Difference'] = pd.to_datetime(filtered_cm_work['Actual Resolution Date & Time']) - pd.to_datetime(filtered_cm_work['Target Resolution Date & Time'])

    # Calculate time difference between target and actual completion for PM
    filtered_pm_work['Completion Difference'] = pd.to_datetime(filtered_pm_work['Actual Resolution Date & Time']) - pd.to_datetime(filtered_pm_work['Target Resolution Date & Time'])

    # Categorize work orders based on efficiency for CM
    filtered_cm_work['Efficiency Category'] = filtered_cm_work['Completion Difference'].apply(
        lambda x: 'On-Time' if x <= pd.Timedelta(0) else 'Late'
    )

    # Categorize work orders based on efficiency for PM
    filtered_pm_work['Efficiency Category'] = filtered_pm_work['Completion Difference'].apply(
        lambda x: 'On-Time' if x <= pd.Timedelta(0) else 'Late'
    )

    # Analyze the efficiency categories for CM
    cm_efficiency_summary = filtered_cm_work['Efficiency Category'].value_counts(normalize=True) * 100

    # Analyze the efficiency categories for PM
    pm_efficiency_summary = filtered_pm_work['Efficiency Category'].value_counts(normalize=True) * 100

####################################################################################

    # Function to plot the top 10 assets with the highest percentage of late work orders for CM
    def plot_top_10_cm_late_assets():
        # Calculate the number of late work orders by asset for CM
        cm_late_work_orders = filtered_cm_work[filtered_cm_work['Efficiency Category'] == 'Late'].groupby('Asset Number').size()

        # Calculate the total number of work orders by asset for CM
        cm_total_work_orders = filtered_cm_work.groupby('Asset Number').size()

        # Calculate the percentage of late work orders for CM
        cm_late_percentage = (cm_late_work_orders / cm_total_work_orders) * 100

        # Sort and select the top 10 assets with the highest percentage of late work orders
        top_cm_late_assets = cm_late_percentage.sort_values(ascending=False).head(10)

        # Plot the top 10 late assets for CM
        plt.figure(figsize=(12, 6))
        top_cm_late_assets.plot(kind='bar', color='red')
        plt.title('Top 10 Assets with the Highest Percentage of Late Work Orders (CM)')
        plt.xlabel('Asset Number')
        plt.ylabel('Percentage of Late Work Orders')
        plt.tight_layout()
        st.pyplot(plt)

    # Function to plot the top 10 assets with the highest percentage of late work orders for PM
    def plot_top_10_pm_late_assets():
        # Calculate the number of late work orders by asset for PM
        pm_late_work_orders = filtered_pm_work[filtered_pm_work['Efficiency Category'] == 'Late'].groupby('Asset Number').size()

        # Calculate the total number of work orders by asset for PM
        pm_total_work_orders = filtered_pm_work.groupby('Asset Number').size()

        # Calculate the percentage of late work orders for PM
        pm_late_percentage = (pm_late_work_orders / pm_total_work_orders) * 100

        # Sort and select the top 10 assets with the highest percentage of late work orders
        top_pm_late_assets = pm_late_percentage.sort_values(ascending=False).head(10)

        # Plot the top 10 late assets for PM
        plt.figure(figsize=(12, 6))
        top_pm_late_assets.plot(kind='bar', color='red')
        plt.title('Top 10 Assets with the Highest Percentage of Late Work Orders (PM)')
        plt.xlabel('Asset Number')
        plt.ylabel('Percentage of Late Work Orders')
        plt.tight_layout()
        st.pyplot(plt)

    col1, col2 = st.columns(2)

    with col1:
        st.header("Top 10 Assets with the Highest Percentage of Late Work Orders (CM)")
        plot_top_10_cm_late_assets()
        st.header("Corrective Maintenance Efficiency Summary:")
        st.dataframe(cm_efficiency_summary)

    with col2:
        st.header("Top 10 Assets with the Highest Percentage of Late Work Orders (PM)")
        plot_top_10_pm_late_assets()
        st.header("Preventive Maintenance Efficiency Summary:")
        st.dataframe(pm_efficiency_summary)

##########################################################################
    import textwrap

    # Wrap text function for long labels
    def wrap_text(text, width=50):
        return textwrap.fill(text, width=width)

    # Function to plot the top 10 descriptions with the highest percentage of late work orders for CM
    def plot_top_10_cm_late_descriptions():
        # Calculate the number of late work orders by description for CM
        cm_late_work_orders_by_description = filtered_cm_work[filtered_cm_work['Efficiency Category'] == 'Late'].groupby('Description').size()

        # Calculate the total number of work orders by description for CM
        cm_total_work_orders_by_description = filtered_cm_work.groupby('Description').size()

        # Calculate the percentage of late work orders for each description in CM
        cm_late_percentage_by_description = (cm_late_work_orders_by_description / cm_total_work_orders_by_description) * 100

        # Sort and select the top 10 descriptions with the highest percentage of late work orders
        top_cm_late_descriptions = cm_late_percentage_by_description.sort_values(ascending=False).head(10).sort_values()

        # Apply wrapping to the descriptions
        wrapped_labels = [wrap_text(label) for label in top_cm_late_descriptions.index]

        # Increase the figure size
        plt.figure(figsize=(14, 10))

        # Plot with wrapped labels
        top_cm_late_descriptions.plot(kind='barh', color='red')

        # Apply the wrapped labels to the y-axis
        plt.yticks(ticks=range(len(wrapped_labels)), labels=wrapped_labels)

        plt.title('Top 10 Descriptions with the Highest Percentage of Late Work Orders (CM)')
        plt.xlabel('Percentage of Late Work Orders')
        plt.ylabel('Description')

        plt.tight_layout()
        st.pyplot(plt)
        plt.close(fig)

    # Function to plot the top 10 descriptions with the highest percentage of late work orders for PM
    def plot_top_10_pm_late_descriptions():
        # Calculate the number of late work orders by description for PM
        pm_late_work_orders_by_description = filtered_pm_work[filtered_pm_work['Efficiency Category'] == 'Late'].groupby('Description').size()

        # Calculate the total number of work orders by description for PM
        pm_total_work_orders_by_description = filtered_pm_work.groupby('Description').size()

        # Calculate the percentage of late work orders for each description in PM
        pm_late_percentage_by_description = (pm_late_work_orders_by_description / pm_total_work_orders_by_description) * 100

        # Sort and select the top 10 descriptions with the highest percentage of late work orders
        top_pm_late_descriptions = pm_late_percentage_by_description.sort_values(ascending=False).head(10).sort_values()

        # Apply wrapping to the descriptions
        wrapped_labels = [wrap_text(label) for label in top_pm_late_descriptions.index]

        # Increase the figure size
        plt.figure(figsize=(14, 10))

        # Plot with wrapped labels
        top_pm_late_descriptions.plot(kind='barh', color='red')

        # Apply the wrapped labels to the y-axis
        plt.yticks(ticks=range(len(wrapped_labels)), labels=wrapped_labels)

        plt.title('Top 10 Descriptions with the Highest Percentage of Late Work Orders (PM)')
        plt.xlabel('Percentage of Late Work Orders')
        plt.ylabel('Description')

        plt.tight_layout()
        st.pyplot(plt)
        plt.close(fig)

    col1, col2 = st.columns(2)
    # Display the efficiency summaries

    with col1:

        st.header("Top 10 Descriptions with the Highest Percentage of Late Work Orders (CM)")
        plot_top_10_cm_late_descriptions()
    with col2:
        st.header("Top 10 Descriptions with the Highest Percentage of Late Work Orders (PM)")
        plot_top_10_pm_late_descriptions()

#############################################################################

    # Aggregate the total maintenance cost by asset for CM
    cm_cost_by_asset = filtered_cm_spare.groupby('ASSET NUMBER')['TOTAL VALUE'].sum().sort_values(ascending=False)

    # Aggregate the total maintenance cost by asset for PM
    pm_cost_by_asset = filtered_pm_spare.groupby('ASSET NUMBER')['TOTAL VALUE'].sum().sort_values(ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.header("Top 10 most expensive assets in CM:")
        st.dataframe(cm_cost_by_asset.head(10))

    with col2:
        st.header("\nTop 10 most expensive assets in PM:")
        st.dataframe(pm_cost_by_asset.head(10))

    # Function to plot the top 10 assets by maintenance cost for CM
    def plot_top_10_cm_cost_assets():
        plt.figure(figsize=(10, 6))
        cm_cost_by_asset.head(10).sort_values(ascending=False).plot(kind='bar', color='blue')
        plt.title('Top 10 Assets by Maintenance Cost (CM)')
        plt.xlabel('Asset Number')
        plt.ylabel('Total Maintenance Cost')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(plt)

    # Function to plot the top 10 assets by maintenance cost for PM
    def plot_top_10_pm_cost_assets():
        plt.figure(figsize=(10, 6))
        pm_cost_by_asset.head(10).sort_values(ascending=False).plot(kind='bar', color='green')
        plt.title('Top 10 Assets by Maintenance Cost (PM)')
        plt.xlabel('Asset Number')
        plt.ylabel('Total Maintenance Cost')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(plt)

    col1, col2 = st.columns(2)
    # Display the efficiency summaries

    with col1:

        st.header("Top 10 Assets by Maintenance Cost (CM)")
        plot_top_10_cm_cost_assets()

    with col2:
        st.header("Top 10 Assets by Maintenance Cost (PM)")
        plot_top_10_pm_cost_assets()

#############################################################################

    # Combine CM and PM costs into a single DataFrame for comparison
    combined_cost_by_asset = pd.DataFrame({
        'CM Total Cost': cm_cost_by_asset,
        'PM Total Cost': pm_cost_by_asset
    }).fillna(0)  # Fill NaN with 0 where an asset has costs in only one of CM or PM

    # Calculate the total cost across both CM and PM
    combined_cost_by_asset['Total Cost'] = combined_cost_by_asset['CM Total Cost'] + combined_cost_by_asset['PM Total Cost']


    # Plot comparison of CM and PM costs for the top 10 most expensive assets
    top_combined_cost = combined_cost_by_asset.sort_values('Total Cost', ascending=False).head(10)

    # Function to plot the comparison of CM and PM costs for the top 10 most expensive assets
    def plot_combined_cost_comparison():
        # Sort and select the top 10 most expensive assets overall
        top_combined_cost = combined_cost_by_asset.sort_values('Total Cost', ascending=False).head(10)

        # Plot comparison of CM and PM costs for the top 10 most expensive assets
        top_combined_cost[['CM Total Cost', 'PM Total Cost']].plot(kind='bar', stacked=True, figsize=(12, 8))
        plt.title('Top 10 Assets by Combined Maintenance Cost (CM and PM)')
        plt.xlabel('Asset Number')
        plt.ylabel('Total Maintenance Cost')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(plt)
        plt.close(fig)

    col1, col2 = st.columns(2)

    with col1:
        # Display the top 10 most expensive assets overall
        st.header("Top 10 most expensive assets overall:")
        st.dataframe(combined_cost_by_asset['Total Cost'].sort_values(ascending=False).head(10))

    with col2:
        st.header("Top 10 Assets by Combined Maintenance Cost (CM and PM)")
        plot_combined_cost_comparison()

with tab2:

    st.header("Key Performance Indicators (KPI)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Machines", total_machines)
    col2.metric("Risk of Failure in 1 Month", machines_at_risk_1m)
    col3.metric("Risk of Failure in 1-3 Months", machines_at_risk_3m)
    col4.metric("Total Estimated Cost", f"${total_predicted_cost:,.2f}")

    col1, col2 = st.columns(2)

    with col1:
        # Display data in an interactive table
        st.subheader("Machine Predictions")
        st.dataframe(df_results[['Asset Number', 'Predicted Time to Next Failure', 'Actual Time to Next Failure', 'Predicted Failure Type', 'Actual Failure Type', 'Predicted Cost', 'Actual Cost']])

        # Display the results in Streamlit
        st.subheader("Machines Predicted to Fail in Less Than 1 Month")
        st.dataframe(df_fail_1m[['Asset Number', 'Predicted Time to Next Failure', 'Predicted Failure Type', 'Predicted Cost']])

        st.subheader("Machines Predicted to Fail in 1 to 3 Months")
        st.dataframe(df_fail_1_3m[['Asset Number', 'Predicted Time to Next Failure', 'Predicted Failure Type', 'Predicted Cost']])

        # Comparison of Predicted and Actual Costs
        st.subheader("Comparison of Predicted and Actual Costs")
        fig, ax = plt.subplots()
        ax.scatter(df_results['Actual Cost'], df_results['Predicted Cost'])
        ax.plot([df_results['Actual Cost'].min(), df_results['Actual Cost'].max()],
                [df_results['Actual Cost'].min(), df_results['Actual Cost'].max()],
                'r--')  # Reference line
        ax.set_xlabel("Actual Cost")
        ax.set_ylabel("Predicted Cost")
        ax.set_title("Actual vs Predicted Cost")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        # Save results to CSV files
        df_fail_1m.to_csv('machines_failing_in_1_month.csv', index=False)
        df_fail_1_3m.to_csv('machines_failing_in_1_to_3_months.csv', index=False)

        # Confusion Matrix for Failure Type
        st.subheader("Confusion Matrix for Failure Type")
        conf_matrix = confusion_matrix(df_results['Actual Failure Type'], df_results['Predicted Failure Type'])
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
        disp.plot(cmap=plt.cm.Blues)
        st.pyplot(plt)

        # Bar chart of predicted failure types
        fig, ax = plt.subplots()
        sns.countplot(x='Predicted Failure Type', data=df_results, ax=ax)
        ax.set_title("Distribution of Predicted Failure Types")
        ax.set_xlabel("Failure Type")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # Cost trend analysis
        st.subheader("Maintenance Cost Trend Over Time")

        # Group data by 'Month-Year' and calculate total predicted cost per month
        cost_trend = df_results.groupby('Month-Year')['Predicted Cost'].sum().reset_index()

        # Plot the cost trend
        fig, ax = plt.subplots()
        sns.lineplot(x='Month-Year', y='Predicted Cost', data=cost_trend, ax=ax)
        ax.set_title("Maintenance Cost Trend")
        ax.set_xlabel("Month-Year")
        ax.set_ylabel("Estimated Cost")
        ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability
        st.pyplot(fig)

        # Analysis of failure type distribution
        st.subheader("Distribution of Failure Types")
        failure_distribution = df_results['Predicted Failure Type'].value_counts().reset_index()
        failure_distribution.columns = ['Failure Type', 'Count']

        fig, ax = plt.subplots()
        sns.barplot(x='Failure Type', y='Count', data=failure_distribution, ax=ax)
        ax.set_title("Distribution of Failure Types")
        ax.set_xlabel("Failure Type")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # Correlation Heatmap
        st.subheader("Correlation Heatmap")

        # Filter only numeric columns
        numeric_cols = df_results.select_dtypes(include=['float64', 'int64']).columns
        corr_matrix = df_results[numeric_cols].corr()

        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

with tab3:

    selected_asset = st.selectbox("Select a Machine:", latest_predictions['Asset Number'].unique())
    asset_info = latest_predictions[latest_predictions['Asset Number'] == selected_asset]

    # Display the most recent prediction for the selected asset
    st.subheader(f"🚨 **Critical Alert for Machine {selected_asset}**")
    col1, col2, col3 = st.columns(3)
    col1.metric(f"Predicted Time to Next Failure:", f"{asset_info['Predicted Time to Next Failure'].values[0]:.2f} days")
    col2.metric(f"Predicted Failure Type:", asset_info['Predicted Failure Type'].values[0])
    col3.metric(f"Estimated Cost of Failure:",f"${asset_info['Predicted Cost'].values[0]:,.2f}")

    # Display the maintenance history for the selected asset
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Maintenance History for Machine {selected_asset}")
        maintenance_history = cleaned_combined_df[cleaned_combined_df['Asset Number'] == selected_asset]
        st.write(maintenance_history[['Creation Date & Time', 'Failure Code', 'Total Spare Parts Cost']])


    with col2:
        st.subheader(f"Predicted Failure Type Distribution for {selected_asset}")
        fig, ax = plt.subplots()
        sns.countplot(x='Predicted Failure Type', data=asset_info, ax=ax)
        ax.set_title(f"Predicted Failure Type for {selected_asset}")
        ax.set_xlabel("Failure Type")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        st.subheader("Latest Prediction vs Actual")
        fig, ax = plt.subplots()
        ax.bar(['Actual Cost', 'Predicted Cost'], 
            [asset_info['Actual Cost'].values[0], asset_info['Predicted Cost'].values[0]],
            color=['blue', 'orange'])
        ax.set_title(f"Cost Comparison for {selected_asset}")
        ax.set_ylabel("Cost")
        st.pyplot(fig)

with tab4:
    # Sistema de alertas
    st.subheader("Alertas Automáticas")
    critical_alerts = df_results[df_results['Predicted Time to Next Failure'] <= 7]
    if not critical_alerts.empty:
        st.error(f"¡Atención! {len(critical_alerts)} máquinas en riesgo crítico de falla en menos de una semana.")
        st.write(critical_alerts[['Asset Number', 'Predicted Time to Next Failure', 'Predicted Failure Type', 'Predicted Cost']])
    else:
        st.success("No hay máquinas en riesgo crítico de falla en la próxima semana.")

with tab5:
    # Opción para descargar los resultados filtrados
    st.subheader("Exportar Datos")
    st.download_button("Descargar Máquinas en Riesgo (1 Mes)", df_fail_1m.to_csv(index=False).encode('utf-8'), "machines_failing_in_1_month.csv", "text/csv")
    st.download_button("Descargar Máquinas en Riesgo (1-3 Meses)", df_fail_1_3m.to_csv(index=False).encode('utf-8'), "machines_failing_in_1_to_3_months.csv", "text/csv")
