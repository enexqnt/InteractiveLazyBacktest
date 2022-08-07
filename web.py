import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import seaborn as sns

st.title('Lazy portfolio backtester')


n_stocks=st.slider('Inserisci il numero di asset: ',min_value=2,max_value=12)


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
    tickers[k] = st.text_input('Asset number: ' +str(k))
    st.write(tickers[k])

##############################
# YAHOO FINANCE DATA DOWNLOAD
##############################
tickers=list(tickers.values())
tickers.sort()

data=yf.download(tickers,interval='1d')[['Adj Close']]
data.dropna(inplace=True)


##############################
# RETURNS COMPUTATION & PLOTTING
##############################
rets=data.pct_change().dropna()
cumrets=(rets+1).cumprod()

import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(cumrets)
st.pyplot(fig)


##############################
## WEIGHTS 
##############################
st.subheader('Specifica i pesi di portafoglio')

for k, v in weights.items():
    weights[k] = st.text_input('Weights of asset number: ' +str(k))
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

# Drawdown
dd=hist-hist.cummax()
fig = plt.figure()
plt.plot(dd)
plt.title('Drawdown storico del portafoglio')
st.pyplot(fig)

cagr=hist.mean()**(252/len(hist))-1
std=hist.pct_change().dropna().std()*(252**0.5)
sharpe=cagr/std
st.write('CAGR: '+str("{:.2%}".format(cagr)))
st.write('STD: '+str("{:.2%}".format(std)))
st.write('Sharpe ratio: '+str("{:.2%}".format(sharpe)))
st.write('Max Drawdown: '+str("{:.2%}".format(dd.min())))

##############################
## MONTE CARLO
##############################
st.header('Monte Carlo Simulation')

#Input
n_mc=st.slider('Inserisci il numero di simulazioni: ',min_value=100,max_value=1000)
n_t=st.slider('Inserisci il numero di giorni: ',min_value=5,max_value=250)

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
plt.plot(hist[-1]*trend(np.quantile(portf_returns[-1],q=0.99)),color='black', label='99% probabilità')
plt.plot(hist[-1]*trend(np.quantile(portf_returns[-1],q=0.01)),color='black',)
plt.plot(hist[-1]*trend(np.quantile(portf_returns[-1],q=0.5)),color='white')

# Display the chart
st.pyplot(f)

# Write some stats
st.write('In 99/100 casi non dovresti perdere più del ' +str("{:.2%}".format((np.quantile(portf_returns[-1],q=0.01)-1))))
st.write('In 99/100 casi non dovresti guadagnare più del ' +str("{:.2%}".format((np.quantile(portf_returns[-1],q=0.99)-1))))
st.write('Il rendimento medio simulato è pari al ' +str("{:.2%}".format((np.quantile(portf_returns[-1],q=0.5)-1))))






