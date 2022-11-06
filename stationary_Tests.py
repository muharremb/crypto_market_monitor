"""
    thanks to Quantopian_Pairs_Trader for detailed object structure
    https://github.com/bartchr808/Quantopian_Pairs_Trader/blob/master/algo.py
"""

import numpy as np
from numpy.random import normal
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import pandas as pd
from swapPcaEngine import pcaSwapBuilder
from helpful_scripts import getLastPricesCoinGecko, getPricesCoinGecko


class ADF(object):
    """
    Augmented Dickey–Fuller (ADF) unit root test
    Source: http://www.pythonforfinance.net/2016/05/09/python-backtesting-mean-reversion-part-2/
    Return true or false respect to either P or critical value
    """

    def __init__(self):
        self.p_value = None
        self.five_perc_stat = None
        self.perc_stat = None
        self.p_min = 0.0
        self.p_max = 0.05
        # self.look_back = 63

    def apply_adf(self, time_series):
        model = ts.adfuller(time_series, 1)
        self.p_value = model[1]
        self.five_perc_stat = model[4]["5%"]
        self.perc_stat = model[0]

    def use_P(self):
        return (self.p_value > self.p_min) and (self.p_value < self.p_max)

    def use_critical(self):
        return abs(self.perc_stat) > abs(self.five_perc_stat)


class Half_Life(object):
    """
    Half Life test from the Ornstein-Uhlenbeck process 
    Source: http://www.pythonforfinance.net/2016/05/09/python-backtesting-mean-reversion-part-2/
    hl_max is 14, hl-min is 1
    """

    def __init__(self):
        self.hl_min = 1.0
        self.hl_max = 14.0
        # self.look_back = 43
        self.half_life = None

    def apply_half_life(self, time_series):
        lag = np.roll(time_series, 1)
        lag[0] = 0
        ret = time_series - lag
        ret[0] = 0

        # adds intercept terms to X variable for regression
        lag2 = sm.add_constant(lag)

        model = sm.OLS(ret, lag2)
        res = model.fit()

        self.half_life = -np.log(2) / res.params[1]

    def use(self):
        return (self.half_life < self.hl_max) and (self.half_life > self.hl_min)

    def get_halfLife(self):
        return self.half_life


class Hurst(object):
    """
    If Hurst Exponent is under the 0.5 value of a random walk, then the series is mean reverting
    Source: https://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing
    lag_max = 20
    input should be dataframe.values
    https://towardsdatascience.com/introduction-to-the-hurst-exponent-with-code-in-python-4da0414ca52e
    

    """

    def __init__(self):
        self.h_min = 0.0
        self.h_max = 0.4
        # self.look_back = 126
        self.lag_max = 20
        self.h_value = None

    def apply_hurst(self, time_series):
        """Returns the Hurst Exponent of the time series vector ts"""
        # Create the range of lag values
        lags = range(2, self.lag_max)

        # Calculate the array of the variances of the lagged differences
        tau = [
            np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag])))
            for lag in lags
        ]
        # Use a linear fit to estimate the Hurst Exponent
        poly = np.polyfit(np.log10(lags), np.log10(tau), 1)

        # Return the Hurst exponent from the polyfit output
        self.h_value = poly[0] * 2.0

    def use(self):
        return (self.h_value < self.h_max) and (self.h_value > self.h_min)

    def get_hurst(self):
        return self.h_value


def get_StationaryTable(modelPca):
    normalizedResidualData = modelPca.get_Normalized_Residual_Data()
    adfSeries = pd.Series(
        index=normalizedResidualData.columns, dtype=object, name="adf"
    )
    halfLifeSeries = pd.Series(
        index=normalizedResidualData.columns, dtype=object, name="half_life"
    )
    for column in normalizedResidualData.columns:
        adfObject = ADF()
        halfLifeObject = Half_Life()
        adfObject.apply_adf(normalizedResidualData[column])
        adfSeries[column] = adfObject.use_P()
        halfLifeObject.apply_half_life(normalizedResidualData[column])
        halfLifeSeries[column] = halfLifeObject.get_halfLife()
    adfSeries = adfSeries * 1
    lastResidualRow = normalizedResidualData.tail(1)
    stationaryDataTable = pd.concat([adfSeries, halfLifeSeries], axis=1)
    return stationaryDataTable.T


def get_TwoSDTable(modelPca):
    modelPredictedLastRow = modelPca.predicted.iloc[-1]
    standardDeviationResidual = modelPca.residual.std()
    twoSdHigh = pd.Series(
        standardDeviationResidual * 2 + modelPredictedLastRow, name="2SD_High"
    )
    twoSdLow = pd.Series(
        modelPredictedLastRow - 2 * standardDeviationResidual, name="2SD_Low"
    )
    lastUpdatedLevel = pd.Series(modelPca.get_dataSet().iloc[-1], name="EOD price")
    twoSDTable = pd.concat([lastUpdatedLevel, twoSdHigh, twoSdLow], axis=1)
    return twoSDTable.T


def get_LastResidualNormalizedDataCoins(modelPca):
    standardDeviationResidualSeries = pd.Series(modelPca.residual.std(), name="ModelSD")
    predictedLastSeries = pd.Series(modelPca.predicted.iloc[-1], name="LastPredicted")
    tickers = modelPca.get_dataSet().columns.tolist()
    lastPricesSeries = getLastPricesCoinGecko(tickers)
    datatable = {}
    for ticker in tickers:
        datatable[ticker] = lastPricesSeries[ticker]
    lastPricesSeriesCorrect = pd.Series(datatable, name="LivePrices")
    lastResidualData = pd.Series(
        (lastPricesSeriesCorrect - predictedLastSeries)
        / standardDeviationResidualSeries,
        name="LiveResidual",
    )
    quantile25 = {}
    quantile75 = {}
    for ticker in tickers:
        quantile25[ticker] = modelPca.nResidual[ticker].quantile(0.10)
        quantile75[ticker] = modelPca.nResidual[ticker].quantile(0.90)
    quantile25Series = pd.Series(quantile25, name="10%Quantile")
    quantile75Series = pd.Series(quantile75, name="90%Quantile")
    getUpdatedTable = pd.concat(
        [
            lastResidualData,
            lastPricesSeriesCorrect,
            standardDeviationResidualSeries,
            quantile25Series,
            quantile75Series,
            predictedLastSeries,
        ],
        axis=1,
    )
    stationaryTable = get_StationaryTable(modelPca)
    summaryTable = pd.concat([getUpdatedTable, stationaryTable.T], axis=1)
    return summaryTable


def get_MeanRevertTickers(modelPca):
    df = get_StationaryTable(modelPca)
    subDf = df.T
    adfPositive = subDf.loc[subDf["adf"] == 1]
    return adfPositive.index.tolist()


def get_IdeaTableForMeanRevertedTickers(modelPca, nQA=0.25):
    ideaTable = {}
    BestDataTable = get_LastResidualNormalizedDataCoins(modelPca)
    for ticker in get_MeanRevertTickers(modelPca):
        lastPriceInDataSet = modelPca.get_dataSet().tail(1)[ticker].values
        if modelPca.nResidual.tail(1)[ticker].values > modelPca.nResidual[
            ticker
        ].quantile(1 - nQA):
            ideaTable[ticker] = ["Sell", lastPriceInDataSet]

        elif modelPca.nResidual.tail(1)[ticker].values < modelPca.nResidual[
            ticker
        ].quantile(nQA):
            ideaTable[ticker] = ["Buy", lastPriceInDataSet]
        else:
            ideaTable[ticker] = "No Position"
        lastPriceInDataSet = 0
    return ideaTable


if __name__ == "__main__":
    tickerList = [
        "bitcoin",
        "ethereum",
        "binancecoin",
        "solana",
        "terra-luna",
        "avalanche-2",
        "matic-network",
        "fantom",
    ]
    database = getPricesCoinGecko(tickerList)
    myModelSwap = pcaSwapBuilder(database)
    myModelSwap.pcaModelBuild()
    print("****" * 13)
    print(get_LastResidualNormalizedDataCoins(myModelSwap))
