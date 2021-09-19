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

def cal_WAP3(dfBook):
    dfBook['WAP3'] = (dfBook.bid_price1*dfBook.bid_size1+dfBook.ask_price1*dfBook.ask_size1)/(dfBook.bid_size1+dfBook.ask_size1)
    
def cal_WAP4(dfBook):
    dfBook['WAP4'] = (dfBook.bid_price2*dfBook.bid_size2+dfBook.ask_price2*dfBook.ask_size2)/(dfBook.bid_size2+dfBook.ask_size2)    

def log_run(dfBook, col):
    #dfBook[f'time_w{w}'] = [f'{a}-{b}' for a, b in zip(dfBook.time_id, dfBook.seconds_in_bucket//w)]
    dfBook[f'logReturn_{col}'] = dfBook.groupby(['time_id'])[col].apply(log_return)

        
        
def cal_vol(dfBook, col_names, group_col='time_id'):
    dat = {'time_id': list(dfBook[group_col].unique())}
    for col in col_names:
        dat[col] = list(dfBook.groupby([group_col])[col].agg(realized_volatility))
    return  pd.DataFrame(dat)
        
def get_time_series(t, dfBook, cols, w, show = False): # w = window size, cols = list of feature name
    cols = [f'logReturn_{col}' for col in cols]

    # Grouping column by w
    dfBook['time_label'] =[f'{a}-{b}' for a, b in zip(dfBook.time_id, dfBook.seconds_in_bucket//w)]
    dfBook = dfBook.dropna()
    # Compute vol based on new group column
    a = cal_vol(dfBook, cols, 'time_label')
    
    # base time_id col
    time = [int(t.split('-')[0]) for t in list(a.time_id)]
    a.insert(0, "time_id0", time, True)
    
    if show: display(a.head())
    
    # time-series column name
    col_name = []
    for col in cols:
        col_tmp = [f"{col}_w{w}_prev{i*w}" for i in range(600//w)]
        col_name.extend(col_tmp)

    # new dataframe
    tmp = pd.DataFrame(columns = ['time_id'] + col_name)
    tmp['time_id'] = t #dfvol.time_id

    for t in list(tmp.time_id):
        b = a.loc[a.time_id0==t, cols].to_numpy()
        #print(list(b.flatten(order = 'F')))
        l_col = len(col_name)
        
        list_b = list(b.flatten(order = 'F'))
        l_b = len(list_b)
        if l_col!= l_b:
            new_b = [sum(list_b) / l_b]*(l_col-l_b)
            new_b = list_b + new_b
            #print(new_b, len(new_b), l_col)
            #print(sum(list_b) / l_b)
            #tmp_b = 
        else: new_b = list_b
        tmp.loc[tmp.time_id==t,col_name] = new_b #list(a[a.time_id0==t][cols])
        #assert len(col_name)== len(list(b.flatten(order = 'F'))), 'Too small window size on time-series, please edit the config file'
        
        #tmp.loc[tmp.time_id==t,col_name] = list(b.flatten(order = 'F')) #list(a[a.time_id0==t][cols])
        
    if show:
        display(tmp)
    return tmp