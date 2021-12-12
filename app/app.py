from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import warnings

from model import ModelGenerator

warnings.filterwarnings('ignore')

st.title("Prediksi Portfolio Saham")
st.write("Note: Hingga 5 saham. Pisahkan masing-masing ticker dengan spasi")
st.write("Example: BBCA BBRI BMRI BJTM PTBA")
amount = st.number_input("Modal", min_value=100000, value=1000000, step=100000)
year = st.number_input("Tahun", 10)
ticker_input = st.text_input("Kode saham")


def format_number(number):
    return f"Rp. {int(number):,}".replace(",", ".")


if ticker_input != "":
    limit = 5
    generator = ModelGenerator()
    tickers = [t.upper() for t in ticker_input.strip().split(' ')]

    valids = np.array([generator.check_url(ticker) for ticker in tickers])
    if not np.all(valids):
        invalid_indexes = np.where(valids == False)[0]
        st.write("Invalid ticker: ", ", ".join([tickers[i] for i in invalid_indexes]))
    tickers = [tickers[i] for i in range(tickers.__len__()) if valids[i]][:limit]
    if tickers.__len__() > 0:
        now = datetime.strptime('2021-11-28', "%Y-%m-%d")
        results = pd.DataFrame({ticker: generator.train_model_for_ticker(ticker, date=now) for ticker in tickers}).T
        rate = results.apply(lambda x: generator.get_cagr(x), axis=1).to_numpy()
        investment_df = pd.DataFrame({
            "a": rate,
            "Kode Saham": tickers,
            "Return per tahun": [str(a) + "%" for a in np.round(rate * 100, 2)]
        })
        investment_df['Hasil'] = [format_number(a) for a in np.round(amount * (1 + rate) ** year, 0)]
        st.write(f"Modal Investasi: {format_number(amount)}")
        st.dataframe(investment_df.iloc[:,1:])
