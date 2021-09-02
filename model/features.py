import pandas as pd
import numpy as np

def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff() 

def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return**2))


def cal_WAP1(dfBook):
    dfBook['WAP1'] = (dfBook.bid_price1*dfBook.ask_size1+dfBook.ask_price1*dfBook.bid_size1)/(dfBook.bid_size1+dfBook.ask_size1)


def cal_WAP2(dfBook):
    dfBook['WAP2'] = (dfBook.bid_price2*dfBook.ask_size2+dfBook.ask_price2*dfBook.bid_size2)/(dfBook.bid_size2+dfBook.ask_size2)
    

def log_run(dfBook, col):
    dfBook[f'logReturn_{col}'] = dfBook.groupby(['time_id'])[col].apply(log_return)

def cal_vol(dfBook, col_names):
    dat = {'time_id': list(dfBook['time_id'].unique())}
    for col in col_names:
        dat[col] = list(dfBook.groupby(['time_id'])[col].agg(realized_volatility))
    return  pd.DataFrame(dat)
        
    