
from gnews import GNews
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import streamlit as st
import datetime
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
import logging
import os
import shutil

import appdirs as ad
ad.user_cache_dir = lambda *args: "/tmp"
import yfinance as yf

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download and setup NLTK for sentiment analysis
try:
    import nltk
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    logging.warning(f"Error downloading NLTK data: {e}")
# Sidebar Header and Input Section
st.sidebar.title("Stock Market Dashboard")
st.sidebar.info("Navigate and analyze stock data effortlessly.")

# **Input: Stock Symbols**
st.sidebar.header("1. Enter Stock Symbols")
user_input_symbols = st.sidebar.text_input(
    "Enter comma-separated stock symbols (e.g., RELIANCE.NS, TCS.NS):",
    value="RELIANCE.NS, TCS.NS, HDFCBANK.NS"
)

# Convert input to a list and validate
stock_symbols = [symbol.strip() for symbol in user_input_symbols.split(",") if symbol.strip()]
if not stock_symbols:
    st.sidebar.error("⚠️ Please enter at least one valid stock symbol.")

# **Dynamic Stock Selection**
selected_stock = st.sidebar.selectbox(
    "2. Select a Stock Symbol",
    options=stock_symbols,
    help="Choose a stock from the list above."
)

# **Analysis Options**
st.sidebar.header("3. Select Analysis Type")
option = st.sidebar.radio(
    "Choose an analysis type:",
    options=[
        "Overall Market Status",
        "Current Price",
        "Price Between Dates",
        "Stock Comparison",
        "Time Series Analysis",
        "Fundamental Analysis",
        "Prediction",
        "Technical Analysis"
    ],
    help="Select the type of stock analysis you'd like to perform."
)

def fetch_stock_data(symbol, start_date, end_date):
    """Fetch historical stock data for a specific symbol within a date range."""
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        st.write(data)
        if data.empty:
            st.warning(f"No data available for {symbol} between {start_date} and {end_date}.")
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()


def fetch_market_data():
    """Retrieve current data for major market indices."""
    indices = {
        "NIFTY": "^NSEI",
        "SENSEX": "^BSESN",
        "Gold": "GC=F",
        "Silver": "SI=F",
        "Dow Jones": "^DJI"
    }
    market_data = {}
    for name, symbol in indices.items():
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            if not data.empty:
                close_price = data['Close'].iloc[-1]
                change = close_price - data['Open'].iloc[0]
                percent_change = (change / data['Open'].iloc[0]) * 100
                market_data[name] = {
                    "price": close_price,
                    "change": change,
                    "percent_change": percent_change
                }
        except Exception as e:
            st.error(f"Error fetching data for {name}: {str(e)}")
    return market_data

def prediction(symbol, days=120):
    """Predict future stock prices and provide recommendations."""
    try:
        # Fetch historical data
        data = fetch_stock_data(symbol, datetime.date.today() - pd.DateOffset(years=1), datetime.date.today())
        
        # Calculate technical indicators
        data['SMA_50'] = SMAIndicator(data['Close'], window=50).sma_indicator()
        data['SMA_200'] = SMAIndicator(data['Close'], window=200).sma_indicator()
        data['RSI'] = RSIIndicator(data['Close'], window=14).rsi()
        data['MACD'] = MACD(data['Close']).macd_diff()
        
        # Clean data
        data = data.dropna()
        
        # Prepare dataset
        X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'MACD']]
        y = data['Close'].shift(-1).dropna()
        X = X.iloc[:-1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
        test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        train_r2 = r2_score(y_train, model.predict(X_train)) * 100
        test_r2 = r2_score(y_test, model.predict(X_test)) * 100
        
        # Predict future prices
        last_data = X.iloc[-1].values.reshape(1, -1)
        predictions = []
        for _ in range(days):
            pred = model.predict(last_data)[0]
            predictions.append(pred)
            last_data = np.roll(last_data, -1)
            last_data[0, -1] = pred
        
        # Generate future dates
        future_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=days)
        pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predictions})
        
        # Combine historical and predicted data
        combined_data = pd.concat([
            pd.DataFrame({'Date': data.index, 'Price': data['Close']}),
            pred_df.rename(columns={'Predicted Price': 'Price'})
        ])
        combined_data.set_index('Date', inplace=True)
        
        # Generate recommendations
        current_price = data['Close'].iloc[-1]
        avg_predicted_price = np.mean(predictions)
        price_change = ((avg_predicted_price - current_price) / current_price) * 100
        
        if price_change > 5:
            recommendation = "Buy"
            summary = "Strong upward trend predicted. Consider buying."
        elif -5 <= price_change <= 5:
            recommendation = "Hold"
            summary = "Stable trend predicted. Hold your position."
        else:
            recommendation = "Sell"
            summary = "Downward trend predicted. Consider selling."
        
        # Display results
        st.write("### Prediction Summary")
        st.write("--------------------------------------------------")
        st.write(f"**Recommendation**: {recommendation}")
        st.write(f"**Summary**: {summary}")
        st.write(f"**Current Price**: ₹{current_price:.2f}")
        st.write(f"**Average Predicted Price**: ₹{avg_predicted_price:.2f}")
        st.write(f"**Predicted Change**: {price_change:.2f}%")
        st.write(f"**Training R²**: {train_r2:.2f}%")
        st.write(f"**Testing R²**: {test_r2:.2f}%")
        st.write(f"**Training RMSE**: ₹{train_rmse:.2f}")
        st.write(f"**Testing RMSE**: ₹{test_rmse:.2f}")
        
        # Plot combined historical and predicted data
        st.line_chart(combined_data)
        
        return predictions
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

def display_technical_analysis(stock_symbol, historical_data):
    """
    Perform and display technical analysis for a selected stock.

    Args:
    - stock_symbol (str): The stock ticker symbol.
    - historical_data (DataFrame): Historical price data for the stock.
    """
    try:
        # Calculate Technical Indicators
        sma_50 = SMAIndicator(historical_data['Close'], window=50).sma_indicator()
        sma_200 = SMAIndicator(historical_data['Close'], window=200).sma_indicator()
        rsi = RSIIndicator(historical_data['Close'], window=14).rsi()
        macd_diff = MACD(historical_data['Close']).macd_diff()

        # Add indicators to DataFrame for charting
        historical_data['SMA_50'] = sma_50
        historical_data['SMA_200'] = sma_200
        historical_data['RSI'] = rsi
        historical_data['MACD_Diff'] = macd_diff

        # Display summaries and charts
        st.write(f"### Technical Analysis for {stock_symbol}")
        st.write("Below are the calculated indicators with charts and explanations to help interpret market trends.")

        # Simple Moving Averages
        st.write("#### Simple Moving Averages (SMA)")
        st.write(
            "The Simple Moving Average (SMA) smooths out price data to identify trends over a specific period. "
            "- **SMA (50)**: Short-term trend. "
            "- **SMA (200)**: Long-term trend. "
            "When the shorter SMA crosses above the longer SMA, it often signals an upward trend, and vice versa."
        )
        sma_fig = go.Figure()
        sma_fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['Close'], mode='lines', name='Close Price'))
        sma_fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['SMA_50'], mode='lines', name='SMA (50)'))
        sma_fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['SMA_200'], mode='lines', name='SMA (200)'))
        sma_fig.update_layout(title="SMA (50 & 200) vs Close Price", xaxis_title="Date", yaxis_title="Price", template="plotly_dark")
        st.plotly_chart(sma_fig)

        # Relative Strength Index
        st.write("#### Relative Strength Index (RSI)")
        st.write(
            "RSI measures the speed and change of price movements. "
            "- Values above 70: Overbought (possible reversal). "
            "- Values below 30: Oversold (possible upward correction)."
        )
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['RSI'], mode='lines', name='RSI'))
        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        rsi_fig.update_layout(title="Relative Strength Index (RSI)", xaxis_title="Date", yaxis_title="RSI Value", template="plotly_dark")
        st.plotly_chart(rsi_fig)

        # MACD (Moving Average Convergence Divergence)
        st.write("#### Moving Average Convergence Divergence (MACD)")
        st.write(
            "The MACD shows the relationship between two moving averages of prices. "
            "The MACD difference (MACD line - Signal line) highlights momentum. "
            "- Positive values: Upward momentum. "
            "- Negative values: Downward momentum."
        )
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['MACD_Diff'], mode='lines', name='MACD Difference'))
        macd_fig.update_layout(title="MACD Difference", xaxis_title="Date", yaxis_title="MACD Value", template="plotly_dark")
        st.plotly_chart(macd_fig)

    except Exception as error:
        st.error(f"Error performing technical analysis: {error}")

def display_stock_comparison(selected_symbols, start_date, end_date):
    """
    Display stock price comparison for selected symbols between specified dates.

    Args:
    - selected_symbols (list): List of stock symbols selected for comparison.
    - start_date (datetime): Start date for historical data.
    - end_date (datetime): End date for historical data.
    """
    try:
        if len(selected_symbols) < 2:
            st.warning("Please select at least two stocks for comparison.")
            return

        st.write(f"### Stock Price Comparison ({start_date} to {end_date})")
        st.write("This section compares the historical closing prices of the selected stocks.")

        # Fetch data for all selected stocks
        stock_data = {}
        for symbol in selected_symbols:
            data = fetch_stock_data(symbol, start_date, end_date)
            if not data.empty:
                stock_data[symbol] = data['Close']
            else:
                st.warning(f"No data available for {symbol} in the selected date range.")

        if not stock_data:
            st.error("No valid data found for the selected stocks.")
            return

        # Combine data into a single DataFrame for comparison
        combined_data = pd.DataFrame(stock_data)
        combined_data.index = pd.to_datetime(combined_data.index)  # Ensure datetime index

        # Plot stock prices
        st.write("#### Closing Price Trend")
        st.write("The line chart below shows the historical closing prices for the selected stocks.")
        fig = go.Figure()
        for symbol in combined_data.columns:
            fig.add_trace(go.Scatter(x=combined_data.index, y=combined_data[symbol], mode='lines', name=symbol))
        fig.update_layout(
            title="Closing Price Comparison",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            legend_title="Stocks",
            template="plotly_dark"
        )
        st.plotly_chart(fig)

        # Calculate and display percentage changes
        st.write("#### Percentage Change")
        st.write(
            "The table below shows the percentage change in closing prices from the start of the period to the end of the period."
        )
        
        # Safely calculate percentage changes, handling missing data or invalid values
        start_prices = combined_data.iloc[0]
        end_prices = combined_data.iloc[-1]
        percent_change = ((end_prices - start_prices) / start_prices) * 100
        percent_change = percent_change.replace([np.inf, -np.inf], np.nan).fillna(0)  # Handle division by zero or NaN

        # Create a DataFrame for display
        percent_change_df = pd.DataFrame({
            "Stock": percent_change.index,
            "Percentage Change (%)": percent_change.values
        })

        # Style and display the table
        st.dataframe(percent_change_df.style.format({"Percentage Change (%)": "{:.2f}"}))

    except Exception as error:
        st.error(f"Error in stock comparison: {error}")



def display_market_interface(market_data):
    """Display overall market data in a grid layout."""
    cols = st.columns(len(market_data))
    for i, (name, data) in enumerate(market_data.items()):
        with cols[i]:
            st.metric(label=name, value=f"{data['price']:.2f}", delta=f"{data['change']:.2f} ({data['percent_change']:.2f}%)")
    
    # Plot NIFTY Intraday Chart
    nifty_data = yf.download('^NSEI', period='1d', interval='5m')
    if not nifty_data.empty:
        fig = go.Figure(data=go.Scatter(x=nifty_data.index, y=nifty_data['Close'], mode='lines'))
        fig.update_layout(title='NIFTY Intraday Chart', xaxis_title='Time', yaxis_title ='Price', template='plotly_dark')
        st.plotly_chart(fig)


def fetch_news_sentiment(symbol):
    """Fetch news and analyze sentiment for a given stock symbol."""
    try:
        gnews = GNews(language='en', country='IN', max_results=10)
        news = gnews.get_news(symbol)
        sentiment_analyzer = SentimentIntensityAnalyzer()
        sentiments = []
        
        st.write("\nLatest News and Sentiments:")
        st.write("--------------------------------------------------")
        
        for article in news:
            title = article['title']
            sentiment_score = sentiment_analyzer.polarity_scores(title)['compound']
            sentiments.append(sentiment_score)
            sentiment_label = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
            st.write(f"Title: {title}\nSentiment: {sentiment_label} (Score: {sentiment_score:.2f})\n")
        
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        overall_sentiment = "Positive" if avg_sentiment > 0 else "Negative" if avg_sentiment < 0 else "Neutral"
        
        st.write("--------------------------------------------------")
        st.write(f"Overall Sentiment: {overall_sentiment} (Score: {avg_sentiment:.2f})")
    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import LSTM, Dense



if option == "Price Between Dates":
    symbol = st.sidebar.selectbox("Select Stock", stock_symbols, key="price_symbol")
    start_date = st.sidebar.date_input(
        "Start Date", 
        datetime.date.today() - pd.DateOffset(years=1), 
        key="price_start_date"
    )
    end_date = st.sidebar.date_input(
        "End Date", 
        datetime.date.today(), 
        key="price_end_date"
    )
    data = fetch_stock_data(symbol, start_date, end_date)
    if not data.empty:
        st.write(f"Price of {symbol} between {start_date} and {end_date}:")
        st.write(data)
        st.line_chart(data['Close'])

elif option == "Stock Comparison":
    selected_stocks = st.sidebar.multiselect("Select Stocks for Comparison", stock_symbols, key="comparison_symbols")
    start_date = st.sidebar.date_input(
        "Start Date", 
        datetime.date.today() - pd.DateOffset(years=1), 
        key="comparison_start_date"
    )
    end_date = st.sidebar.date_input(
        "End Date", 
        datetime.date.today(), 
        key="comparison_end_date"
    )
    display_stock_comparison(selected_stocks, start_date, end_date)


if option == "Overall Market Status":
    market_data = fetch_market_data()
    display_market_interface(market_data)

elif option == "Current Price":
    symbol = st.sidebar.selectbox("Select Stock", stock_symbols)
    ticker = yf.Ticker(symbol)
    st.write(f"Current Price of {symbol}: ₹{ticker.info.get('currentPrice', 'N/A')}")
    fetch_news_sentiment(symbol)


elif option == "Time Series Analysis":
    symbol = st.sidebar.selectbox("Select Stock", stock_symbols)
    data = fetch_stock_data(symbol, datetime.date.today() - pd.DateOffset(years=1), datetime.date.today())
    if not data.empty:
        st.write(f"Time Series Analysis of {symbol}:")
        st.write(data)
        st.line_chart(data['Close'])

elif option == "Fundamental":
    st.write("Hey")
    symbol = st.sidebar.selectbox("Select Stock", stock_symbols)
    ticker = yf.Ticker(symbol)
    info = ticker.info
    st.write(f"Fundamental Analysis of {symbol}:")
    st.write("--------------------------------------------------")
    st.write(f"Market Cap: ₹{info.get('marketCap', 'N/A')}")
    st.write(f"PE Ratio: {info.get('peRatio', 'N/A')}")
    st.write(f"Dividend Yield: {info.get('dividendYield', 'N/A')}")
    st.write(f"EPS: ₹{info.get('eps', 'N/A')}")
    st.write(f"52-Week High: ₹{info.get('fiftyTwoWeekHigh', 'N/A')}")
    st.write(f"52-Week Low: ₹{info.get('fiftyTwoWeekLow', 'N/A')}")

elif option == "Prediction":
    symbol = st.sidebar.selectbox("Select Stock for Prediction", stock_symbols)
    days = st.sidebar.slider("Days to Predict", 1, 120)
    predictions = prediction(symbol, days)
    fetch_news_sentiment(symbol)
elif option == "Technical Analysis":
    selected_stock = st.sidebar.selectbox("Select Stock", stock_symbols)
    stock_data = fetch_stock_data(selected_stock, datetime.date.today() - pd.DateOffset(years=1), datetime.date.today())
    if not stock_data.empty:
        display_technical_analysis(selected_stock, stock_data)
