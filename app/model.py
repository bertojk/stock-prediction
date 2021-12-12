from datetime import datetime
from time import mktime

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


class ModelGenerator:
    def train_model_for_ticker(self, ticker, date=datetime.today(), years=10):
        try:
            data = self.__fetch_data(ticker, date, years)
            smooth_data = self.__smoothing_data(data)

            alphas = [.0001, .001, .01, .1, 1]
            scores = [self.__train_model(data=smooth_data, alpha=a)[1] for a in alphas]
            performance = pd.DataFrame({
                'alpha': alphas,
                'score': scores
            })
            best = performance.sort_values(by='score', ascending=False).iloc[0, 0]
            return self.__train_model(data, alpha=best, final=True)
        except:
            return None

    def __get_url(self, ticker, date=datetime.today(), years=1):
        period1 = self.__get_timestamp(date.replace(year=date.year - years).strftime("%Y-%m-%d"))
        period2 = self.__get_timestamp(date.strftime("%Y-%m-%d"))
        interval = '1d'
        url = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}.JK?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
        return url

    def check_url(self, ticker):
        try:
            url = self.__get_url(ticker)
            pd.read_csv(url)
            return True
        except:
            return False

    def predict_price(self, rows, n_years):
        model = rows.model
        prices = [rows.Target]
        days = n_years * 52 * 5
        for i in range(days):
            current = prices[i]
            predict = model.predict(np.array([current]).reshape(-1, 1))
            prices.append(predict[0])
        return prices[1:]

    def get_cagr(self, rows):
        prices = self.predict_price(rows, 1)
        return prices[-1] / prices[0] - 1

    def __fetch_data(self, ticker, date, years):
        url = self.__get_url(ticker, date, years)
        raw_data = pd.read_csv(url)
        data = raw_data[['Date', 'Close']]
        data = data.fillna(method='ffill')
        return data

    @staticmethod
    def __get_timestamp(dt):
        format = "%Y-%m-%d"
        return int(mktime(datetime.strptime(dt, format).timetuple()))

    @staticmethod
    def __smoothing_data(data, window_size=100):
        smooth_data = [data.Close[i - window_size:i].mean() for i in range(window_size, data.shape[0])]
        return pd.DataFrame({
            'Close': smooth_data
        })

    @staticmethod
    def __train_model(data, n_shift=1, alpha=1, final=False):
        last_data = data.iloc[-2:, :]

        data['Target'] = data['Close'].shift(-n_shift)
        data = data.iloc[:-n_shift]
        X = data[['Close']]
        y = data['Target']
        model = Ridge(alpha)
        if not final:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=52, shuffle=False)
            model.fit(X_train, y_train)
            return model, model.score(X_test, y_test)
        else:
            model.fit(X, y)
            return {
                "model": model,
                "last_date": data.iloc[-1, 0],
                "Close": last_data.iloc[-2, 1],
                "Target": last_data.iloc[-1, 1]
            }
