import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.statespace.sarimax import SARIMAX
import base64

# Set page configuration as the first Streamlit command
st.set_page_config(
    page_title="WEATHERAnalysis",
    page_icon=":sun_behind_rain_cloud:",
    layout="wide",
)

st.title(':partly_sunny: Weather Trends: Temperature and Humidity Analysis')
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

# Options to upload a file
fl = st.file_uploader(":file_folder: Upload a file", type=["csv", "xlsx", "xls"])
#partitioning the page
col1, col2 = st.columns((2))
with col1:
    if fl is not None:
        # Check file type and read accordingly
        if fl.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            # Excel file
            df = pd.read_excel(fl)
            st.write("Data from uploaded Excel file:")
            st.write(df)
        else:
            # Assume other formats as CSV
            df = pd.read_csv(fl, encoding="ISO-8859-1")
            st.write("Data from uploaded CSV file:")
            st.write(df)
    else:
        # Read data from the default Excel file
        #os.chdir(r"default directory containing the file")
        df = pd.read_excel("/content/Temp_Humid.xlsx")
        st.write("Data from default Excel file:")
        st.write(df)

df["time"]=pd.to_datetime(df["time"])
# Create a new column "month" with the month information
df["month"] = df["time"].dt.strftime('%B')
# Create a new column "year" with the year information
df["year"] = df["time"].dt.year
# Create a new column "day" with the year information
df["day"] = df["time"].dt.day

st.sidebar.header("Weather Data Analysis ")
# Create for Month
month = st.sidebar.multiselect("Pick the Months", df["month"].unique())
if not month:
    df2 = df.copy()
else:
    df2 = df[df["month"].isin(month)].copy()

# Create for Year
year = st.sidebar.multiselect("Pick the Years", df["year"].unique())
if not year:
    df3 = df.copy()
else:
    df3 = df[df["year"].isin(year)].copy()

# Filter the data based on Region, State, and City
if not month and not year:
    filtered_df = df.copy()
elif not year:
    filtered_df = df3
elif not month:
    filtered_df = df2
else:
    filtered_df = df[df2["month"].isin(month) & df3["year"].isin(year)].copy()

# Reset index after filtering
filtered_df.reset_index(drop=True, inplace=True)

with col2:
    st.write("Data After Filtering:")
    st.write(filtered_df)

filter_dataset=filtered_df.copy()

#Overview of the dataset 
with col1:
    # Define the desired order of the months
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

    # Convert the 'month' column to a categorical data type with the desired order
    filter_dataset['month'] = pd.Categorical(filter_dataset['month'], categories=month_order, ordered=True)

    # Check if any months are selected
    if month:
        # Create a DataFrame with the necessary columns
        selected_months_df = filter_dataset[filter_dataset['month'].isin(month)]
        
        # Group by month and calculate Mean, Average, and Maximum temperature
        grouped_df = selected_months_df.groupby('month')['temperature_mean'].agg(['mean', 'median', 'max']).reset_index()

        title = 'Mean, Median, and Maximum Temperature of Selected Months'
    else:
        # Calculate Mean, Median, and Maximum temperature for the entire dataset
        grouped_df = filter_dataset.groupby('month')['temperature_mean'].agg(['mean', 'median', 'max']).reset_index()

        title = 'Mean, Median, and Maximum Temperature of All Months'

    # Create a bar plot using plotly express
    fig = px.bar(grouped_df, x='month', y=['mean', 'median', 'max'],
                labels={'value': 'Temperature'},
                title=title,
                barmode='group',color_discrete_sequence=['coral', 'blue', 'navy'])

    # Update layout
    fig.update_layout(xaxis=dict(title='Month', categoryorder='array', categoryarray=month_order), yaxis=dict(title='Temperature'))

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Check if any months are selected
    if month:
        # Create a DataFrame with the necessary columns
        selected_months_df = filter_dataset[filter_dataset['month'].isin(month)]
        
        # Group by month and calculate Mean, Average, and Maximum temperature
        grouped_df = selected_months_df.groupby('month')['relativehumidity_mean'].agg(['mean', 'median', 'max']).reset_index()

        title = 'Mean, Median, and Maximum Related Humidity of Selected Months'
    else:
        # Calculate Mean, Median, and Maximum temperature for the entire dataset
        grouped_df = filter_dataset.groupby('month')['relativehumidity_mean'].agg(['mean', 'median', 'max']).reset_index()

        title = 'Mean, Median, and Maximum Relative Humidity of all Months'

    # Create a bar plot using plotly express
    fig = px.bar(grouped_df, x='month', y=['mean', 'median', 'max'],
                labels={'value': 'Relative Humidity'},
                title=title,
                barmode='group',
                color_discrete_sequence=['coral', 'blue', 'navy'])

    # Update layout
    fig.update_layout(xaxis=dict(title='Month',categoryorder='array', categoryarray=month_order), yaxis=dict(title='Relative Humidity'))
    # Show the plot
    st.plotly_chart(fig, use_container_width=True)



#Scatter Plot with Trend Line: Temperature vs Relative Humidity
with col1:
    #Scatter plot 
    from scipy.stats import linregress
    import numpy as np
    # Create a scatter plot using Plotly Express
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_df['temperature_mean'], y=filtered_df['relativehumidity_mean'],
                            mode='markers', name='Temperature vs Relative Humidity',
                            marker=dict(color='coral'),showlegend=False))

    # Calculate linear regression
    slope, intercept, _, _, _ = linregress(filtered_df['temperature_mean'], filtered_df['relativehumidity_mean'])
    x_line = np.linspace(filtered_df['temperature_mean'].min(), filtered_df['temperature_mean'].max(), 100)
    y_line = slope * x_line + intercept

    # Add trend line
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Trend Line', line=dict(color='blue', dash='dash'),showlegend=False))

    # Update layout
    fig.update_layout(title='Scatter Plot with Trend Line: Temperature vs Relative Humidity',
                    xaxis=dict(title='Temperature'),
                    yaxis=dict(title='Relative Humidity', showgrid=False))

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)
#Interactive Correlation Plot: Temperature vs Relative Humidit
with col2:
    # Create a DataFrame with temperature and relative humidity
    correlation_df = filtered_df[['temperature_mean', 'relativehumidity_mean']]

    # Calculate the correlation matrix
    correlation_matrix = correlation_df.corr()

    # Create an interactive heatmap using plotly express
    fig = px.imshow(correlation_matrix, labels=dict(color="Correlation"), color_continuous_scale='Turbo')

    # Update layout
    fig.update_layout(title='Interactive Correlation Plot: Temperature vs Relative Humidity',
                    xaxis=dict(title='Variable'),
                    yaxis=dict(title='Variable'))

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)

# Sidebar for filtering data
start_date = st.sidebar.date_input("Start Date", min(df["time"]))
end_date = st.sidebar.date_input("End Date", max(df["time"]))

# Convert selected dates to datetime
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Filter data based on date range
filtered_df = df[(df["time"] >= start_date) & (df["time"] <= end_date)]

# Line plot for temperature mean
st.subheader("Temperature Mean over Time")
fig_temp = px.line(filtered_df, x="time", y="temperature_mean", title="Temperature Mean over Time",color_discrete_sequence=["purple"])
st.plotly_chart(fig_temp)

st.subheader("Relative Humidity Mean over Time")
fig_humidity = px.line(filtered_df, x="time", y="relativehumidity_mean", title="Relative Humidity Mean over Time",color_discrete_sequence=["teal"])
st.plotly_chart(fig_humidity)

# Plot temperature and humidity together
st.subheader("Temperature and Relative Humidity Mean over Time")
fig_combined = px.line(filtered_df, x="time", y=["temperature_mean", "relativehumidity_mean"], title="Temperature and Relative Humidity Mean over Time",color_discrete_sequence=["purple", "teal"])
st.plotly_chart(fig_combined)

# Histogram for relative humidity mean
st.subheader("Relative Humidity Mean Distribution")
fig_humidity = px.histogram(filtered_df, x="relativehumidity_mean", title="Relative Humidity Mean Distribution",color_discrete_sequence=["green"])
st.plotly_chart(fig_humidity)

# Summary statistics
st.subheader("Summary Statistics")
st.write(filtered_df.describe())

# Box plot of temperature by month
st.subheader("Box Plot of Temperature by Month")
df['month'] = pd.to_datetime(df['time']).dt.month
fig_box_temp = px.box(df, x="month", y="temperature_mean", title="Box Plot of Temperature by Month",color_discrete_sequence=["coral"])
fig_box_temp.update_layout(xaxis_title="Month", yaxis_title="Temperature Mean")
st.plotly_chart(fig_box_temp)

#for day Comparison between Temperature and Humidity over Days
# Group by day and calculate the average for each day
grouped_df = filtered_df.groupby('day').agg({'temperature_mean': 'mean', 'relativehumidity_mean': 'mean'}).reset_index()

# Melt the DataFrame to have a single column for values and another for the variable
melted_df = pd.melt(grouped_df, id_vars=['day'], value_vars=['temperature_mean', 'relativehumidity_mean'],
                    var_name='Variable', value_name='Value')

# Create a dual-axis line chart using Plotly Express and make_subplots
fig = make_subplots(specs=[[{'secondary_y': True}]])

# Line colors for temperature and relative humidity
temperature_color = 'teal'
humidity_color = 'hotpink'

# Add traces with specified line colors
fig.add_trace(go.Scatter(x=melted_df['day'], y=melted_df['Value'][melted_df['Variable']=='temperature_mean'],
                         mode='lines', name='Temperature', line=dict(color=temperature_color)))
fig.add_trace(go.Scatter(x=melted_df['day'], y=melted_df['Value'][melted_df['Variable']=='relativehumidity_mean'],
                         mode='lines', name='Relative Humidity', line=dict(color=humidity_color)),
              secondary_y=True)

# Update layout
fig.update_layout(title='Comparison between Temperature and Humidity over Days',
                  xaxis=dict(title='Day'),
                  yaxis=dict(title='Temperature', showgrid=False),
                  yaxis2=dict(title='Relative Humidity', showgrid=False, overlaying='y', side='right'))
# Show the plot
st.plotly_chart(fig,use_container_width=True, height = 200)

#For Months Comparison between Temperature and Humidity over Months
# Group by month and calculate the average for each month
grouped_df = filtered_df.groupby('month').agg({'temperature_mean': 'mean', 'relativehumidity_mean': 'mean'}).reset_index()

# Melt the DataFrame to have a single column for values and another for the variable
melted_df = pd.melt(grouped_df, id_vars=['month'], value_vars=['temperature_mean', 'relativehumidity_mean'],
                    var_name='Variable', value_name='Value')

# Create a plot using Plotly Express and make_subplots
fig = make_subplots(specs=[[{'secondary_y': True}]])

# Check if only one month is selected
if len(filtered_df['month'].unique()) == 1:
    # Show a bar plot for mean values when a single month is selected
    fig.add_trace(go.Bar(x=['Temperature', 'Relative Humidity'], y=[grouped_df['temperature_mean'].iloc[0], grouped_df['relativehumidity_mean'].iloc[0]],
                         marker=dict(color=['purple', 'skyblue'])))
else:
    # Show a line plot when multiple months are selected
    fig.add_trace(go.Scatter(x=melted_df['month'], y=melted_df['Value'][melted_df['Variable']=='temperature_mean'],
                             mode='lines', name='Temperature Mean', line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=melted_df['month'], y=melted_df['Value'][melted_df['Variable']=='relativehumidity_mean'],
                             mode='lines', name='Relative Humidity Mean', line=dict(color='orange')),
                  secondary_y=True)

# Update layout
fig.update_layout(title='Comparison between Temperature and Humidity over Months',
                  xaxis=dict(title='Months/Metric'),
                  yaxis=dict(title='Temperature', showgrid=False),
                  yaxis2=dict(title='Relative Humidity', showgrid=False, overlaying='y', side='right'))

# Show the plot
st.plotly_chart(fig, use_container_width=True, height=200)
   
# SARIMA Forecasting for 15 days
st.header("SARIMA Forecasting for 15 Days")

# Fit SARIMA model
order_sarima = (1, 0, 1)
seasonal_order_sarima = (1, 0, 1, 12)
model_sarima = SARIMAX(filtered_df['temperature_mean'], order=order_sarima, seasonal_order=seasonal_order_sarima)
fitted_sarima = model_sarima.fit()

# Forecast
forecast_steps_sarima = 15  # Forecast horizon for 15 days
forecast_sarima = fitted_sarima.get_forecast(steps=forecast_steps_sarima, alpha=0.05)
forecast_index_sarima = pd.date_range(filtered_df['time'].max(), periods=forecast_steps_sarima + 1, freq='D')[1:]
forecast_values_sarima = forecast_sarima.predicted_mean

# Plot SARIMA Forecast
fig_sarima = go.Figure()
fig_sarima.add_trace(go.Scatter(x=filtered_df['time'], y=filtered_df['temperature_mean'], mode='lines', name='Actual Temperature',line=dict(color='teal')))
fig_sarima.add_trace(go.Scatter(x=forecast_index_sarima, y=forecast_values_sarima, mode='lines', name='SARIMA Forecast',line=dict(color='goldenrod')))

# Update layout
fig_sarima.update_layout(title='SARIMA Forecast vs Actual Temperature for 15 Days',
                         xaxis=dict(title='Time'),
                         yaxis=dict(title='Temperature'))

# Show the plot
st.plotly_chart(fig_sarima, use_container_width=True)








