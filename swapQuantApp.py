from helpful_scripts import get_DataBaseDf, getPricesCoinGecko, getLastPricesCoinGecko
import pandas as pd
from swapPcaEngine import pcaSwapBuilder
from stationary_Tests import (
    get_IdeaTableForMeanRevertedTickers,
    get_MeanRevertTickers,
    get_StationaryTable,
    get_LastResidualNormalizedDataCoins,
)
import backtrader as bt
from datetime import datetime
import plotly.express as px
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Crypto PCA Analysis")

st.title("Relative Value Analysis Using Unsupervised Learning")
st.subheader("Bitcoin, Ethereum, Binance Coin, Solana, Luna, Avax, Polygon, Fantom")
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

st.write("Building database now. Please wait around 30 seconds.")
st.caption(
    "Website might not work well on mobile due to resizing plots. Desktop view is recommended. "
)

dataBaseDf = getPricesCoinGecko(tickerList, CCY="usd", lastNDay=91, interval="daily")

databaseStartFrom1 = dataBaseDf.div(dataBaseDf.iloc[0])

fig2 = px.line(
    databaseStartFrom1,
    y=databaseStartFrom1.columns,
    # title="Historical Price Graph",
    width=500,
    height=500,
)
fig2.update_xaxes(
    rangeselector=dict(
        buttons=list(
            [
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(step="all"),
            ]
        )
    ),
)

fig2.update_layout(
    xaxis_title="Dates", yaxis_title="", margin=dict(l=0, r=5, b=5, t=0, pad=4),
)
# fig2.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
st.plotly_chart(fig2, use_container_width=False)

st.write(
    "Model measures the extend to which crypto currencies deviate from a constructed fair value based on principle component analysis. Depending on residuals, model could highlight relative value trade ideas."
)
# Building PCA model. Input: DataFrame, ModelOutputs: n-array

myModelSwap = pcaSwapBuilder(dataBaseDf)
myModelSwap.pcaModelBuild()

# plotly library usage
df = myModelSwap.nResidual
yesterdayResidualData = df.iloc[-1]
oneWeekAgoResidualData = df.iloc[-8]
oneMonthAgoResidualData = df.iloc[-31]
adfTest = get_StationaryTable(myModelSwap).iloc[0]
frame = {
    "1d Ago": yesterdayResidualData,
    "1w Ago": oneWeekAgoResidualData,
    "1m Ago": oneMonthAgoResidualData,
    "adfTest": adfTest,
}
dataf = pd.DataFrame(frame)
dataf.reset_index(inplace=True)
dataf = dataf.rename(columns={"index": "Coins"})

# Coins column name change
dataf["Coins"].replace(
    {
        "binancecoin": "binance-coin",
        "avalanche-2": "avax",
        "terra-luna": "terra",
        "matic-network": "polygon",
    },
    inplace=True,
)
dataf = dataf.round(2)
meanRevertedTickers = get_MeanRevertTickers(myModelSwap)

highestResidual = dataf.loc[dataf["1d Ago"].idxmax()][0]
lowestResidual = dataf.loc[dataf["1d Ago"].idxmin()][0]

fig = px.bar(
    dataf,
    x="Coins",
    y="1d Ago",
    hover_data=["1w Ago", "1m Ago", "adfTest"],
    color="1d Ago",
    color_continuous_scale=px.colors.diverging.Portland,
    width=400,
    height=400,
)
fig.update_layout(
    xaxis_title="", yaxis_title="EOD Residual", margin=dict(l=0, r=5, b=5, t=40, pad=4),
)
st.plotly_chart(fig, use_container_width=False)
st.write("Bar Chart shows 1d/1w/1m ago residual values with adf test in hover labels.")

st.write(
    "Model uses Augmented Dickey–Fuller(adf) to test mean reversion. Summary table below highlights a trade idea based on mean reverting property and residual value."
)

adfDf = get_LastResidualNormalizedDataCoins(myModelSwap)
# adfDf = df.loc[df["adf"] == 1]
adfDf = adfDf.round(2)
conditions = [
    (adfDf["LiveResidual"] > adfDf["90%Quantile"]) & adfDf["adf"] == 1,
    (adfDf["LiveResidual"] < adfDf["10%Quantile"]) & adfDf["adf"] == 1,
    (
        (adfDf["LiveResidual"] < adfDf["90%Quantile"])
        | (adfDf["LiveResidual"] > adfDf["10%Quantile"])
    ),
]
choices = ["Sell", "Buy", "No Pos"]
adfDf["TradeIdea"] = np.select(conditions, choices, default="No Pos")
adfDf.reset_index(inplace=True)
adfDf = adfDf.rename(columns={"index": "Coins"})
adfDf["Coins"].replace(
    {
        "binancecoin": "binance-coin",
        "avalanche-2": "avax",
        "terra-luna": "terra",
        "matic-network": "polygon",
    },
    inplace=True,
)
adfDfUpdated = adfDf[
    [
        "Coins",
        "TradeIdea",
        "LivePrices",
        "LiveResidual",
        "adf",
        "LastPredicted",
        "10%Quantile",
        "90%Quantile",
        "ModelSD",
    ]
]
adfDfUpdated = adfDfUpdated.round(2)
adfDfUpdated.sort_values(by=["LiveResidual"], ascending=False, inplace=True)
fig3 = go.Figure(
    data=[
        go.Table(
            header=dict(
                values=list(adfDfUpdated.columns),
                fill_color="paleturquoise",
                align="center",
                font_size=12,
            ),
            cells=dict(
                values=[
                    adfDfUpdated.Coins,
                    adfDfUpdated.TradeIdea,
                    adfDfUpdated.LivePrices,
                    adfDfUpdated.LiveResidual,
                    adfDfUpdated.adf,
                    adfDfUpdated.LastPredicted,
                    adfDfUpdated["10%Quantile"],
                    adfDfUpdated["90%Quantile"],
                    adfDfUpdated.ModelSD,
                ],
                fill_color="lavender",
                align="center",
                font_size=12,
            ),
        )
    ]
)
fig3.update_layout(
    xaxis_title="",
    yaxis_title="",
    margin=dict(l=0, r=0, b=0, t=0),
    width=620,
    height=225,
)
st.plotly_chart(fig3, use_container_width=False)

st.subheader("PCA Generated Trade Ideas")
tradeIdeaSummary = adfDfUpdated.loc[adfDfUpdated["TradeIdea"] != "No Position"]
(rw, clm) = tradeIdeaSummary.shape

if rw == 0:
    st.write("No trade idea at current market levels")

else:
    for index, row in tradeIdeaSummary.iterrows():
        st.write(
            row["Coins"],
            ": ",
            row["TradeIdea"],
            "at ",
            str(row["LivePrices"]),
            ", model fair value is ",
            str(row["LastPredicted"]),
        )
st.header("About me")
st.write(
    "Relative value analysis is a must for swap traders. And, if you trade emerging market rates; you just have to love it. I have traded a widow-maker swap market for a while, easy to guess which one I m talking about. My obsession about risk and relative value analysis help me a lot."
)
st.write(
    "Crypto markets took my attention lately and I'm looking for ways to fill the gap between tradFi and crypto. I decided to build a relative value tool for crypto currencies using unsupervised learning. Depending on the residuals, the model could signal relative value trade opportunities. Also it could be used to i) signal timing of profit taking, ii) manage portfolios with creating hedge ratios for specific benchmark assets. "
)
st.write(
    "I used Coingecko API for historical prices and Streamlit for front end. I wrote scripts in Python using scikit-learn and deployed to heroku. For all other details, either technical or related to model building process, you can always reach me at **boztepe.muharrem@gmail.com**"
)
st.write("Muharrem")
st.header("Disclaimer")
st.write(
    "This material is solely for information purposes and not an investment advice."
)

