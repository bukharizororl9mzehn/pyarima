from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pandas import date_range
from io import BytesIO
import base64
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
    


@app.route('/forecast/', methods=['POST'])
def forecast():
    # get user input
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    frequency = request.form['frequency']

    # load data from csv file
    df = pd.read_csv('Gold Price (2013-2023).csv', parse_dates=['Date'])
    df.drop(columns=['Open', 'High', 'Low', 'Vol.', 'Change %'], inplace=True)
    df['Price'] = pd.to_numeric(df['Price'].str.replace(',', ''))
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df["Price"] = df["Price"].round(2)
    df.set_index('Date', inplace=True)

    # generate forecast
    df_forecast, forecast = generate_forecast(start_date, end_date, frequency)

    # create date range for x-axis labels
    if frequency == 'W':
        date_range = pd.date_range(start=df_forecast.index[0], end=df_forecast.index[-1], freq='W')
        date_labels = date_range.strftime('%b %d, %Y')
    elif frequency == 'M':
        date_range = pd.date_range(start=df_forecast.index[0], end=df_forecast.index[-1], freq='M')
        date_labels = date_range.strftime('%b %Y')
    elif frequency == 'Y':
        date_range = pd.date_range(start=df_forecast.index[0], end=df_forecast.index[-1], freq='Y')
        date_labels = date_range.strftime('%Y')
    else:
        date_labels = None

    # plot actual data and forecast
    plt.plot(df_forecast.index, df_forecast['Price'], label='Actual')
    plt.plot(df_forecast.index, forecast, label='Forecast')
    plt.legend()

    # convert plot to HTML string
    plot_html = mpl_to_html(plt)

    # render forecast.html with plot and labels
    return render_template('forecast.html', plot_html=plot_html, date_labels=date_labels)

def generate_forecast(start_date, end_date, frequency):
    # load data from csv file
    df = pd.read_csv('Gold Price (2013-2023).csv', parse_dates=['Date'])
    df.drop(columns=['Open', 'High', 'Low', 'Vol.', 'Change %'], inplace=True)
    df['Price'] = pd.to_numeric(df['Price'].str.replace(',', ''))
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df["Price"] = df["Price"].round(2)
    df.set_index('Date', inplace=True)

    # fit ARIMA model to data
    model = ARIMA(df, order=(1, 1, 1))
    model_fit = model.fit()

    # generate forecast
    forecast = model_fit.forecast(steps=get_forecast_steps(start_date, end_date, frequency))[0]
    forecast = pd.Series(forecast, index=get_forecast_dates(start_date, end_date, frequency))

    # combine actual data and forecast into single DataFrame
    df_forecast = pd.concat([df, forecast.to_frame('Price')], axis=0)

    return df_forecast, forecast

def get_forecast_steps(start_date, end_date, frequency):
    # calculate number of steps required for forecast
    if frequency == 'W':
        steps = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days // 7
    elif frequency == 'M':
        steps = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days // 30
    elif frequency == 'Y':
        steps = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days // 365
    else:
        steps = None
    return steps


def get_forecast_dates(start_date, end_date, frequency):
    # create date range for forecast
    if frequency == 'W':
        forecast_dates = pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq='W')
    elif frequency == 'M':
        forecast_dates = pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq='M')
    elif frequency == 'Y':
        forecast_dates = pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq='Y')
    else:
        forecast_dates = None
    return forecast_dates


def mpl_to_html(figure):
    # convert Matplotlib plot to HTML image tag
    buffer = BytesIO()
    figure.savefig(buffer, format='png')
    buffer.seek(0)
    image_data = buffer.getvalue()
    buffer.close()
    return '<img src="data:image/png;base64,' + base64.b64encode(image_data).decode() + '">'

if __name__ == '__main__':
    app.run(debug=True)





