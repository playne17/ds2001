#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 10:58:56 2021

@author: peterlayne
"""

# load modules
import numpy as np
import pandas as pd

# risk-free Treasury rate
R_f = 0.0175 / 252

# read in the market data
data = pd.read_csv('capm_market_data.csv')

Look at some records  
SPY is an ETF for the S&P 500 (the "stock market")  
AAPL is Apple  
The values are closing prices, adjusted for splits and dividends

data.head()

Drop the date column

data


Compute daily returns (percentage changes in price) for SPY, AAPL  
Be sure to drop the first row of NaN  
Hint: pandas has functions to easily do this

returns= data.pct_change(periods= 1)
returns=returns.drop(index= 0, axis= 0)
returns

#### 1. (1 PT) Print the first 5 rows of returns

print(returns.head())
aapl= returns.iloc[:,1:].values
spy= returns.iloc[:,:1].values


Save AAPL, SPY returns into separate numpy arrays  
#### 2. (1 PT) Print the first five values from the SPY numpy array, and the AAPL numpy array

print(aapl[:5])
print(spy[:5])

##### Compute the excess returns of AAPL, SPY by simply subtracting the constant *R_f* from the returns.
##### Specifically, for the numpy array containing AAPL returns, subtract *R_f* from each of the returns. Repeat for SPY returns.

NOTE:  
AAPL - *R_f* = excess return of Apple stock  
SPY - *R_f* = excess return of stock market


exret_aapl= aapl - R_f
exret_spy= spy - R_f

#### 3. (1 PT) Print the LAST five excess returns from both AAPL, SPY numpy arrays


print(exret_aapl[-5:])
print(exret_spy[-5:])

#### 4. (1 PT) Make a scatterplot with SPY excess returns on x-axis, AAPL excess returns on y-axis####
Matplotlib documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html

import matplotlib.pyplot as plt
plt.scatter(exret_spy, exret_aapl)

#### 5. (3 PTS) Use Linear Algebra (matrices) to Compute the Regression Coefficient Estimate, \\(\hat\beta_i\\)

Hint 1: Here is the matrix formula where *x′* denotes transpose of *x*.

\begin{aligned} \hat\beta_i=(x′x)^{−1}x′y \end{aligned} 

Hint 2: consider numpy functions for matrix multiplication, transpose, and inverse. Be sure to review what these operations do, and how they work, if you're a bit rusty.

spy_trans= np.transpose(exret_spy)

regcoeef= 1/(np.matmul(spy_trans, exret_spy)) * (np.matmul(spy_trans,exret_aapl))


You should have found that the beta estimate is greater than one.  
This means that the risk of AAPL stock, given the data, and according to this particular (flawed) model,  
is higher relative to the risk of the S&P 500.


def beta_find(x , y):
    regco= 1/(np.matmul(np.transpose(x), x)) * (np.matmul(np.transpose(x), y))
    return regco

beta_find(exret_spy, exret_aapl)

#### Measuring Beta Sensitivity to Dropping Observations (Jackknifing)

Let's understand how sensitive the beta is to each data point.   
We want to drop each data point (one at a time), compute \\(\hat\beta_i\\) using our formula from above, and save each measurement.

#### 6. (3 PTS) Write a function called `beta_sensitivity()` with these specs:

- take numpy arrays x and y as inputs
- output a list of tuples. each tuple contains (observation row dropped, beta estimate)

Hint: **np.delete(x, i).reshape(-1,1)** will delete observation i from array x, and make it a column vector

def beta_sensitivity(x , y, z):
    new_list=[]
    zx= list(range(z))
    for i in zx:
        newx= np.delete(x, i)
        newy= np.delete(y, i)
        regco= beta_find(newx, newy)
        out= (i, regco)
        new_list.append(out)
    return(new_list)
    
    
    
    



#### Call `beta_sensitivity()` and print the first five tuples of output.

beta_sensitivity(exret_spy, exret_aapl, 5)