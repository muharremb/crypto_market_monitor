import pandas as pd
import requests
from pycoingecko import CoinGeckoAPI
import yfinance as yf


def get_DataBaseDf():
    database = pd.read_csv("swapUSData.csv")
    database.drop(columns=["Unnamed: 0"], inplace=True)
    databasedf = database
    return databasedf


def getPricesCoinGecko(tickerList, CCY="usd", lastNDay=130, interval="daily"):
    # cg.get_price(ids=['bitcoin', 'litecoin', 'ethereum'], vs_currencies='usd')
    # 27oct21 close, 27oct21 03.00 Ist Time close
    cg = CoinGeckoAPI()
    apiData = {}
    for coin in tickerList:
        try:
            apiList = cg.get_coin_market_chart_by_id(
                id=coin, vs_currency=CCY, days=lastNDay
            )["prices"]
            apiData[coin] = {}
            apiData[coin]["timestamps"], apiData[coin]["price"] = zip(*apiList)
        except Exception as e:
            print(e)
            print("coin: " + coin)

    apiDataFrameList = [
        pd.DataFrame(apiData[coin]["price"], columns=[coin]) for coin in tickerList
    ]
    apiDataFrame = pd.concat(apiDataFrameList, axis=1).sort_index()
    apiDataFrame["TimeStamp"] = pd.to_datetime(
        apiData[tickerList[0]]["timestamps"], unit="ms"
    )
    apiDataFrame["TimeStamp"] = apiDataFrame["TimeStamp"].dt.date.astype(str)
    apiDataFrame.set_index("TimeStamp", inplace=True)
    # apiDataFrame.drop("timestamps", inplace=True, axis=1)
    apiDataFrame = apiDataFrame.head(-1)
    return apiDataFrame


def getLastPricesCoinGecko(tickerList, CCY="usd"):
    cg = CoinGeckoAPI()
    lastPriceList = cg.get_price(tickerList, vs_currencies=CCY)
    df = pd.DataFrame(lastPriceList)
    lastPriceDataFrame = df.rename(index={"usd": "LastPrice"})
    lastPriceSeries = pd.Series(lastPriceDataFrame.iloc[-1], name="LastPrice")
    return lastPriceSeries


def getYahooApiPrice():
    btcUsd_df = yf.download(
        "BTC-USD", start="2021-01-01", end="2021-10-27", progress=False
    )
    return btcUsd_df


if __name__ == "__main__":
    tickerlist = [
        "bitcoin",
        "ethereum",
        "binancecoin",
        "solana",
        "terra-luna",
        "avalanche-2",
        "matic-network",
        "fantom",
    ]
    print(getPricesCoinGecko(tickerlist))
