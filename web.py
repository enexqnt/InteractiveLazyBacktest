import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import seaborn as sns

st.title('Lazy portfolio backtester')


n_stocks=st.slider('Select the number of assets: ',min_value=2,max_value=12)


##############################
# DICTIONARY FOR INPUT
##############################
tickers={}
for i in range(0,n_stocks):
     tickers[i] = 0

weights={}
for i in range(0,n_stocks):
     weights[i] = 0

##############################
# TICKERS OF ASSETS
##############################
for k, v in tickers.items():
    tickers[k] = st.text_input('Ticker of asset number: ' +str(k), help='Insert the ticker in yahoo finance format',placeholder='WLD.MI')
    st.write(tickers[k])

##############################
# YAHOO FINANCE DATA DOWNLOAD
##############################
tickers=list(tickers.values())
tickers.sort()

data=yf.download(tickers,interval='1d')['Adj Close']
data.dropna(inplace=True)


##############################
# RETURNS COMPUTATION & PLOTTING
##############################
rets=data.pct_change().dropna()
cumrets=(rets+1).cumprod()

import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(cumrets)
plt.legend(loc="upper left")
st.pyplot(fig)


##############################
## WEIGHTS 
##############################
st.subheader('Define weights of the portfolio')

for k, v in weights.items():
    weights[k] = st.text_input(tickers[k]+' weight:',placeholder=0.5)
    st.write(weights[k])

weights=list(weights.values())
weights=[float(i) for i in weights]


##############################
## HIST PORTFOLIO ANALYSIS 
##############################
st.header('Analisi storica')

# Rendimento cumulato
hist=((rets@weights)+1).cumprod()
fig = plt.figure()
plt.plot(hist)
plt.title('Rendimento cumulato storico del portafoglio')
st.pyplot(fig)

##############################
## DRAWDOWN & other metrics
##############################
dd=(hist-hist.cummax())
fig = plt.figure()
plt.plot(dd)
plt.title('Drawdown storico del portafoglio')
st.pyplot(fig)

cagr=hist.mean()**(252/len(hist))-1
std=hist.pct_change().dropna().std()*(252**0.5)
sharpe=cagr/std
col1, col2, col3, col4 = st.columns(4)
col1.metric('CAGR',str("{:.2%}".format(cagr)))
col2.metric('STD',str("{:.2%}".format(std)))
col3.metric('Sharpe ratio',str("{:.2f}".format(sharpe)))
col4.metric('Max Drawdown',str("{:.2%}".format(dd.min())))

##############################
## HIST MAX LOSS AFTER X DAYS
##############################
st.subheader('Minimum days to avoid losses')

# create data
a=[]
for i in range(100, int(len(hist)/2)):
    a.append((((hist.shift(-i)/hist).dropna()-1).cummin().min()))
a=pd.DataFrame(a)

# write min. days
if a[a<0].index[-1]>= int(len(hist)/2)-101:
    st.metric('Min. days to avoid losses', '+∞')
else:
    st.metric('Min. days to avoid losses', len(a[a<0].dropna()))

# Plot
fig = plt.figure()
plt.plot(a,color='black')
plt.axhline(0)
plt.title('Max. loss after x days - Historical analysis')
st.pyplot(fig)




##############################
## MONTE CARLO
##############################
st.header('Monte Carlo Simulation')

#Input
n_mc=st.slider('Define the number of simulations: ',min_value=100,max_value=1000)
n_t=st.slider('Define the number of days for the simulations: ',min_value=5,max_value=250)

# Mean and cov computation
mu_df=pd.DataFrame(rets.mean())
portf_returns = np.full((n_t,n_mc),0.)
cov=rets.cov()

# Loop for simulations
for i in range(0,n_mc):
    Z = np.random.standard_t(n_t,size=len(tickers)*n_t)
    Z = Z.reshape((len(tickers),n_t))
    L = np.linalg.cholesky(cov)
    wkrets=np.inner(L,np.transpose(Z))+np.array(mu_df)
    portf_r = np.cumprod(np.inner(weights,np.transpose(wkrets)) + 1)
    future_dates = [rets.index[-1] + timedelta(days=x) for x in range(0,n_t+1)]
    portf_returns[:,i] = portf_r


##############################
## PLOTTING AUX. FUNCTIONS
##############################

def trend(r,n_t=n_t):
    L= np.zeros([n_t + 1, 1])
    L[0]=1
    for t in range(1, int(n_t)+1):
      L[t] = L[t-1]*(r**(1/n_t))
    L_df=pd.DataFrame(L,index=future_dates)
    return L_df

##############################
## PLOT MONTE CARLO SIM.
##############################

#insert portfolio's start price
ptf_returns=np.insert(portf_returns, 0, 1, axis=0)

f = plt.figure()
f.set_figwidth(15)
f.set_figheight(10)

# Plot mc simulations
with sns.color_palette("winter"):
    plt.plot(hist[-1]*pd.DataFrame(ptf_returns,index=future_dates))

# Plot hist cum returns
plt.plot(hist['2022':])

# Plot trend lines
plt.plot(hist[-1]*trend(np.quantile(portf_returns[-1],q=0.99)),color='black', label='99% probability')
plt.plot(hist[-1]*trend(np.quantile(portf_returns[-1],q=0.01)),color='black',)
plt.plot(hist[-1]*trend(np.quantile(portf_returns[-1],q=0.5)),color='white')

# Display the chart
st.pyplot(f)

# Write some stats
st.write('Forecasted returns')
col1, col2, col3 = st.columns(3)
col1.metric('Maximum loss',str("{:.2%}".format((np.quantile(portf_returns[-1],q=0.01)-1))))
col2.metric('Maximum profit', str("{:.2%}".format((np.quantile(portf_returns[-1],q=0.99)-1))))
col3.metric('Average return after '+str(n_t)+' days',str("{:.2%}".format((np.quantile(portf_returns[-1],q=0.5)-1))))






