#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 15:18:40 2018

@author: anantavinashi
"""

import pandas as pd

from pandas_datareader import data, wb

import numpy.random as npr

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import matplotlib as mpl

#Bitcoin vs Transaction Volume and Transaction Count
Bitcoin = pd.read_csv('/Users/anantavinashi/Downloads/Projects/Data/Bitcoin Price Txn Count.csv')

BTC = Bitcoin.set_index('Date')

BTC.head()

BTC.tail()

BTC.corr()

#Bitcoin Price vs Transaction per day
BitcoinPTpD = pd.read_csv('/Users/anantavinashi/Downloads/Projects/Data/BitcoinPTpD.csv')

BitPTpd = BitcoinPTpD.set_index('Date')

BitPTpd.head()

BitPTpd.corr()

#Bitcoin Price vs Hash Rate
BitcoinPvHR = pd.read_csv('/Users/anantavinashi/Downloads/Projects/Data/BitcoinPvHR.csv')

BitcoinPvHR.head()

BitPvHR = BitcoinPvHR.set_index('Date')

BitPvHR.head()

BitPvHR.tail()

BitPvHR.corr()

#Bitcoin Price vs Transaction Volume
PvTV = pd.read_csv('/Users/anantavinashi/Downloads/Projects/CryptoCurrencies/PvTV.csv')

PvV = PvTV.set_index('date')

PvV.corr()

#Bitcoin Price vs Transaction Volume and Hash Rate
Bitcoin_Correlation = pd.read_csv('/Users/anantavinashi/Downloads/Projects/Data/Bitcoin Correlations.csv')

Bitcoin_Correlation.head()

Bitcoin_Correlation.tail()

Bitcor = Bitcoin_Correlation.set_index('Date')

Bitcor.head()

Bitcor.tail()

Bitcor.describe()

Bitcoin_Correlation.describe()

Bitcor.corr()
Out[36]: 

#Bitcoin Price vs Transction per Block
BitcoinPricevsTxnBlock = pd.read_csv('/Users/anantavinashi/Downloads/Projects/Data/BitcoinPricevsTxnBlock.csv')

BitcoinPricevsTxnBlock.head()

bitprvstxnb = BitcoinPricevsTxnBlock.set_index('Date')

bitprvstxnb.head()

bitprvstxnb.tail()

bitprvstxnb.corr()

#Cryptocurrency
cryptocurrency = pd.read_csv('/Users/anantavinashi/Downloads/Projects/Data/Cryptoanant.csv')

crant = cryptocurrency.set_index('Date')

crant.describe()

crant.corr()

#Ethereum Price and Transaction Volume and Count
Ethereum2 = pd.read_csv('/Users/anantavinashi/Downloads/Projects/Data/Ethereum2.csv')

Ethereum2.head()

eth = Ethereum2.set_index('Date')

eth.head()

eth.tail()

eth.corr()

#Ethereum Classic Price and Transaction Volume and Count, Block Size and Count

EthereumClassic = pd.read_csv('/Users/anantavinashi/Downloads/Projects/Data/ETC.csv')

etc = EthereumClassic.set_index('Date')

etc.head()

etc.tail()

etc.corr()

#Litecoin Price and Transaction Volume, Count and Block Size
ltc = Litecoin.set_index('Date')

ltc.head()

ltc.tail()
Out[84]: 

ltc.corr()

#Monero Price and Transaction Volume, Count and Block Size
Monero = pd.read_csv('/Users/anantavinashi/Downloads/Projects/Data/XMR.csv')

Monero.head()

xmr = Monero.set_index('Date')

xmr.tail()

xmr.head()

xmr.corr()

#Dash Price and Transaction Volume, Count and Block Size
Dash1 = pd.read_csv('/Users/anantavinashi/Downloads/Projects/Data/Dash1.csv')

Dash1.head()

Dash1.tail()

dash = Dash1.set_index('Date')

dash.head()

dash.tail()

dash.corr()

#Bitcoin Cash Price and Transaction Volume, Count, Block Size and Block Count
BCH = pd.read_csv('/Users/anantavinashi/Downloads/Projects/Data/BCH.csv')

BCH.head()

bch = BCH.set_index('Date')

bch.head()

bch.tail()

bch.corr()

#ZCash Price and Transaction Volume, Count, Block Size and Block Count
ZEC = pd.read_csv('/Users/anantavinashi/Downloads/Projects/Data/zec.csv')

ZEC.head()

zec = ZEC.set_index('Date')

zec.head()

zec.tail()

zec.corr()

#EOS Price and Transaction Volume, Count
EOS = pd.read_csv('/Users/anantavinashi/Downloads/Projects/Data/EOS.csv')

EOS.head()

eos = EOS.set_index('Date')

eos.head()

eos.tail()

eos.corr()

#Ripple Price vs Transaction olumeand Count
Ripple = pd.read_csv('/Users/anantavinashi/Downloads/Projects/Data/XRP.csv')

Ripple.head()

xrp = Ripple.set_index('Date')

xrp.head()

xrp.tail()

xrp.corr()

#Bitcoin Price with Hash Rate, Txn Vol, Txn/Block, Block Size & Block Count

bitcoin1 = pd.read_csv('/Users/anantavinashi/Downloads/Projects/Data/Bitcoin1.csv')

bitcoin1.head()

xbt = bitcoin1.set_index('Date')

xbt.head()

xbt.tail()

xbt.corr()

#Bitcoin till 2016
bitcoin2016 = pd.read_csv('/Users/anantavinashi/Downloads/Projects/Data/Bitcoin2016.csv')

bitcoin2016.head()

xbt16 = bitcoin2016.set_index('Date')

xbt16.head()

xbt16.tail()

xbt16.corr()

#Bitcoin in 2017
bitcoin2017 = pd.read_csv('/Users/anantavinashi/Downloads/Projects/Data/Bitcoin2017.csv')

xbt17 = bitcoin2017.set_index('Date')

 xbt17.head()

xbt17.tail()

xbt17.corr()

#Bitcoin in 2018
bitcoin2018 = pd.read_csv('/Users/anantavinashi/Downloads/Projects/Data/Bitcoin2018.csv')

xbt18 = bitcoin2018.set_index('Date')

xbt18.head()

xbt18.tail()

xbt18.corr()

#Cryptocurrency vs Index
CryInd = pd.read_csv('/Users/anantavinashi/Downloads/Projects/Data/CryptovIndex.csv')

CryInd.head()

crnd = CryInd.set_index('Date')

crnd.head()

crnd.tail()

crnd.corr()

#Cryptocurrency vs Equity
CryEqt = pd.read_csv('/Users/anantavinashi/Downloads/Projects/Data/CryptovEquity.csv')

CryEqt.head()

cret = CryEqt.set_index('Date')

cret.head()

cret.tail()

cret.corr()

#Cryptocurrency vs Commodity
CryComm = pd.read_csv('/Users/anantavinashi/Downloads/Projects/Data/CryptovCommodity.csv')

crcm = CryComm.set_index('Date')

crcm.head()

crcm.tail()

crcm.corr()

#Cryptocurrency vs Currency
CryCurr = pd.read_csv('/Users/anantavinashi/Downloads/Projects/Data/CryptovCurrency.csv')

crcur = CryCurr.set_index('Date')

crcur.head()

crcur.tail()

crcur.corr()

#Heatmap of Correlations of Cryptocurrencies
corr = crant.corr()
print (corr)
plt.title('Heatmap of Correlation Matrix of Cryptocurrencies')
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, annot=False, cmap="Blues")
plt.show()


xbt16.describe()

xbt17.describe()

xbt18.describe()

#Data Visualisation Price Chart for Bitcoin
bitcoin1.plot(x='Date', y='Price ($)', kind='line')
plt.show()

cryptoascend.plot(figsize=(8, 5))

cryptoascend.plot(kind= 'box' , subplots=True, layout=(3,3), figsize=(13,11))

Bitcoin18 = pd.read_csv('/Users/anantavinashi/Downloads/Projects/Data/Bitcoin18.csv')

Bitcoin18.plot(title='Bitcoin Price 2018', figsize=(8,5), x='Date', y='Price ($)', kind='line', \
colormap='viridis', fontsize='12')
Out[13]: <matplotlib.axes._subplots.AxesSubplot at 0x11a28b390>
￼

Bitcoin17 = pd.read_csv('/Users/anantavinashi/Downloads/Projects/Data/Bitcoin17.csv')

Bitcoin16 = pd.read_csv('/Users/anantavinashi/Downloads/Projects/Data/Bitcoin16.csv')

Bitcoin17.plot(title='Bitcoin Price 2017', figsize=(8,5), x='Date', y='Price ($)', kind='line', \
colormap='viridis', fontsize='12')
Out[16]: <matplotlib.axes._subplots.AxesSubplot at 0x11a28aba8>
￼

Bitcoin16.plot(title='Bitcoin Price till December 2016', figsize=(8,5), x='Date', y='Price ($)', kind='line', \
colormap='viridis', fontsize='12')
Out[17]: <matplotlib.axes._subplots.AxesSubplot at 0x11a3d8518>

#PIE CHARTS

market = pd.read_csv('/Users/anantavinashi/Downloads/Projects/Data/Market.csv')

market

Cryptocurrency = []

MarketCap = []


for row in market:
    Cryptocurrency.append(row[0])
    Marketcap.append(row[1])

explode = [0.1,0,0,0,0,0,0,0,0,0]

plt.title('Market Cap of Cryptocurrencies')
plt.pie(Marketcap, labels=Cryptocurrency,explode=explode,labeldistance=1.1)
plt.axis('equal') # make the pie chart circular
plt.show()

#Linear regression

import math

from sklearn import metrics

from sklearn import cross_validation
/Applications/Anaconda/anaconda/lib/python3.5/site-packages/sklearn/cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
  "This module will be removed in 0.20.", DeprecationWarning)

from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LinearRegression

Bitcoin_Correlation = pd.read_csv('/Users/anantavinashi/Downloads/Projects/Data/Bitcoin Correlations.csv')

Bitcor = Bitcoin_Correlation.set_index('Date')

Bitcor.columns

x = Bitcor[:1]

x

x = Bitcor.values[:, 1:]

x

y = Bitcor['Price ($)']

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=.20)

lm = LinearRegression()

lm.fit(X_train, Y_train)
Out[29]: LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

print("lm.intercept_", lm.intercept_)
lm.intercept_ 112.981373777

print("lm.coef_", lm.coef_)
lm.coef_ [  1.83681168e-04   3.86825305e-07]

predict = lm.predict(X_test)

predict
Out[33]: 
array([   431.26333553,   3824.07612744,    197.65307687, ...,
          234.56235646,   1067.6139936 ,  14163.21182874])
print("MEAN SQUARED ERROR: ", metrics.mean_squared_error(Y_test, predict))
MEAN SQUARED ERROR:  826399.678313

print("R2 SCORE: ", metrics.r2_score(Y_test, predict))
R2 SCORE:  0.925190073957

predict.var()
Out[36]: 9794180.2600216828
import statsmodels.api as sm
/Applications/Anaconda/anaconda/lib/python3.5/site-packages/statsmodels/compat/pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
  from pandas.core import datetools

x_train = sm.add_constant(X_train)

model = sm.OLS(Y_train, X_train)

results = model.fit()

print(results.summary())
                            OLS Regression Results                            
==============================================================================
Dep. Variable:              Price ($)   R-squared:                       0.933
Model:                            OLS   Adj. R-squared:                  0.932
Method:                 Least Squares   F-statistic:                     5040.
Date:                Fri, 10 Aug 2018   Prob (F-statistic):               0.00
Time:                        17:43:26   Log-Likelihood:                -6116.4
No. Observations:                 730   AIC:                         1.224e+04
Df Residuals:                     728   BIC:                         1.225e+04
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1             0.0002   4.38e-06     42.713      0.000       0.000       0.000
x2          3.919e-07   7.57e-09     51.760      0.000    3.77e-07    4.07e-07
==============================================================================
Omnibus:                      156.799   Durbin-Watson:                   1.983
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             4451.027
Skew:                           0.133   Prob(JB):                         0.00
Kurtosis:                      15.094   Cond. No.                         696.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



￼
