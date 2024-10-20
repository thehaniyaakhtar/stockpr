import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Function to fetch stock data
def fetch_stock_data(ticker):
    try:
        stock_data = yf.download(ticker, period="1d", interval="1m")
        if stock_data.empty:
            raise ValueError("No data found.")
        return stock_data
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

# Function to plot stock data
def plot_stock_data(data, ticker):
    plt.figure(figsize=(10, 6))
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.title(f"{ticker} Stock Price")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

# Streamlit app
def main():
    # Custom CSS for black background and header color
    st.markdown(
        """
        <style>
        .reportview-container {
            background-color: black;  /* Set the background color to black */
            color: white;  /* Set the text color to white for visibility */
        }
        h1 {
            color: #FFD700;  /* Set header color to gold */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Stock Chatbot")

    # Input for stock ticker
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, GOOGL):", "")
    
    if ticker:
        # Fetch and display stock data
        stock_data = fetch_stock_data(ticker)
        if stock_data.empty:
            st.error("No data found for this ticker.")
        else:
            st.write(f"Displaying data for {ticker}:")
            st.dataframe(stock_data)
            plot_stock_data(stock_data, ticker)

if __name__ == '__main__':
    main()
