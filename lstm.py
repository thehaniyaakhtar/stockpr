import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the model
model = load_model(r'C:\Users\theha\Desktop\StockPR\Stock Prediction model.keras')

# Sidebar for Navigation and User Inputs
st.sidebar.title("Stock Market Dashboard")
st.sidebar.header("Settings")

# Define a more extensive dictionary of categories and corresponding stock symbols
category_stock_map = {
    "Machinery": ["CAT", "DE", "GE", "BA", "HON"],
    "Technology": ["GOOG", "AAPL", "MSFT", "NVDA", "IBM"],
    "Medicine": ["PFE", "MRNA", "JNJ", "LLY", "ABT"],
    "Automobile": ["TSLA", "GM", "F", "HMC", "TM"],
    "Finance": ["JPM", "BAC", "GS", "C", "WFC"],
    "Retail": ["WMT", "TGT", "COST", "AMZN", "HD"],
    "Energy": ["XOM", "CVX", "BP", "RDS-A", "COP"],
    "Telecommunications": ["VZ", "T", "TMUS", "CHL", "AMX"],
    "Food & Beverage": ["KO", "PEP", "MDLZ", "MCD", "SBUX"],
    "Real Estate": ["PLD", "AMT", "CCI", "SPG", "O"],
}

# Define risk and capitalization categories
risk_cap_map = {
    "High Cap": ["GOOG", "AAPL", "MSFT", "NVDA", "JPM"],
    "Small Cap": ["GE", "HON", "MRNA", "SPG", "CCI"],
    "Low Risk": ["WMT", "JNJ", "KO", "ABT"],
    "High Risk": ["TSLA", "AMD", "BA", "XOM"],
}

# Select category from dropdown
category = st.sidebar.selectbox('Select Category', options=list(category_stock_map.keys()))

# Select risk/capitalization category from dropdown
cap_risk_category = st.sidebar.selectbox('Select Capitalization/Risk Category', options=list(risk_cap_map.keys()))

# Get available stocks for the selected category
stock_options = category_stock_map[category]

# Filter stocks based on the selected capitalization/risk category
filtered_stock_options = [stock for stock in stock_options if stock in risk_cap_map[cap_risk_category]]

# Allow the user to either select from the dropdown or input a stock symbol directly
if filtered_stock_options:
    stock = st.sidebar.selectbox('Select Stock (Scroll or Enter)', options=filtered_stock_options, index=0)
else:
    st.sidebar.warning(f"No stocks available in the selected '{cap_risk_category}' category for '{category}'. Please try different selections.")
    stock = ""

# Provide an input box for manual entry of the stock symbol
user_stock_input = st.sidebar.text_input("Or enter stock symbol manually", value=stock)

# Check if the manually entered stock symbol belongs to the selected filtered stock options
if user_stock_input and user_stock_input not in filtered_stock_options:
    st.sidebar.warning(f"The stock '{user_stock_input}' is not part of the '{cap_risk_category}' category.")
    if filtered_stock_options:
        stock = filtered_stock_options[0]  # Default to the first stock from the filtered options if available
    else:
        stock = ""
else:
    stock = user_stock_input

# Date range input
start = st.sidebar.date_input("Start Date", pd.to_datetime('2012-01-01'))
end = st.sidebar.date_input("End Date", pd.to_datetime('2022-12-31'))

# Show moving averages option
show_ma50 = st.sidebar.checkbox('Show 50-Day MA', True)
show_ma100 = st.sidebar.checkbox('Show 100-Day MA', True)
show_ma200 = st.sidebar.checkbox('Show 200-Day MA', True)

# Fetch stock data from yfinance
if stock:
    data = yf.download(stock, start, end)

    # Display stock data as a DataFrame
    st.header(f"Stock Data for {stock} ({category})")
    st.write(data)

    # Stock Data Summary
    st.subheader("Data Summary")
    st.write(data.describe())

    # Split data into training and testing sets
    data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    past_100_days = data_train.tail(100)
    data_test = pd.concat([past_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)

    # Moving Averages
    ma_50_days = data.Close.rolling(50).mean()
    ma_100_days = data.Close.rolling(100).mean()
    ma_200_days = data.Close.rolling(200).mean()

    # Stock Price Plot
    st.subheader(f'Price Data for {stock} ({category})')
    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(data.Close, 'g', label='Price')

    # Conditionally plot moving averages
    if show_ma50:
        plt.plot(ma_50_days, 'r', label='50-Day MA')
    if show_ma100:
        plt.plot(ma_100_days, 'b', label='100-Day MA')
    if show_ma200:
        plt.plot(ma_200_days, 'y', label='200-Day MA')

    plt.title(f'{stock} Price with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig1)

    # Create x and y for predictions
    x = []
    y = []

    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i - 100:i])
        y.append(data_test_scale[i, 0])

    x, y = np.array(x), np.array(y)

    # Make predictions
    predict = model.predict(x)

    # Rescale predictions
    scale = 1 / scaler.scale_
    predict = predict * scale
    y = y * scale

    # Price vs Predicted Price
    st.subheader('Price vs Predicted Price')
    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(predict, 'r', label='Predicted Price')
    plt.plot(y, 'g', label='Actual Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)

    # Stock Recommendation Logic
    st.subheader('Stock Recommendation')

    # Check for Buy or Sell signals
    if ma_50_days.iloc[-1] > ma_200_days.iloc[-1]:
        st.success("Recommendation: **Buy** (50-day MA is above 200-day MA)")
    elif ma_50_days.iloc[-1] < ma_200_days.iloc[-1]:
        st.error("Recommendation: **Sell** (50-day MA is below 200-day MA)")
    else:
        st.info("Recommendation: **Hold** (No significant trend detected)")

    # Stock Performance Metrics
    st.subheader("Performance Metrics")

    # Daily Returns
    data['Daily Return'] = data['Close'].pct_change()

    # Cumulative Returns
    cumulative_returns = (1 + data['Daily Return']).cumprod() - 1
    st.write(f"Cumulative Return: {cumulative_returns.iloc[-1]:.2%}")

    # Volatility (annualized standard deviation of daily returns)
    annualized_volatility = data['Daily Return'].std() * np.sqrt(252)
    st.write(f"Annualized Volatility: {annualized_volatility:.2%}")

    # Sharpe Ratio (assuming risk-free rate of 0%)
    sharpe_ratio = (data['Daily Return'].mean() / data['Daily Return'].std()) * np.sqrt(252)
    st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    # Maximum Drawdown
    drawdown = data['Close'] / data['Close'].cummax() - 1
    max_drawdown = drawdown.min()
    st.write(f"Maximum Drawdown: {max_drawdown:.2%}")

    # Relative Strength Index (RSI)
    window_length = 14
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window_length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window_length).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    st.write(f"Relative Strength Index (RSI): {rsi.iloc[-1]:.2f}")

    # Pie Chart for Daily Returns Distribution
    st.subheader("Daily Returns Distribution")
    daily_return_counts = pd.cut(data['Daily Return'], bins=3, labels=['Negative Returns', 'Zero Returns', 'Positive Returns']).value_counts()
    all_labels = ['Negative Returns', 'Zero Returns', 'Positive Returns']
    all_sizes = [daily_return_counts.get(label, 0) for label in all_labels]

    fig3, ax3 = plt.subplots()
    wedges, texts, autotexts = ax3.pie(all_sizes, labels=all_labels, startangle=90, autopct='%1.1f%%', pctdistance=0.85)

    # Improve label appearance
    for text in texts:
        text.set_fontsize(12)  # Set font size for the labels
    for autotext in autotexts:
        autotext.set_color('white')  # Set color for percentage text
        autotext.set_fontsize(12)  # Set font size for percentage text

    ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig3)

    # Conclusion based on metrics
    st.subheader("Conclusion")
    if cumulative_returns.iloc[-1] > 0:
        st.success("The stock has positive cumulative returns.")
    else:
        st.error("The stock has negative cumulative returns.")
