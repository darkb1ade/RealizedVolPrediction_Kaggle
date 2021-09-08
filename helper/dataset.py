from model.features import *
import pandas as pd
import yaml
import glob
import os
from IPython.display import display

#os.chdir('C:/Users/Darkblade/Documents/Kaggle/RealizedVolPrediction_Kaggle')
print('current path', os.getcwd())

class DataLoader():
    def __init__(self, mode):
        self.conf = yaml.load(open('config/main.yaml'), Loader=yaml.FullLoader)
        self.book_path = sorted(glob.glob(f"{self.conf['path']}/book_{mode}.parquet/*/*"))
        self.trade_path =  sorted(glob.glob(f"{self.conf['path']}/trade_{mode}.parquet/*/*"))
        self.book_path = [l.replace('\\','/') for l in self.book_path]
        self.book_path = [l.replace('//','/') for l in self.book_path]
        self.trade_path = [l.replace('\\','/') for l in self.trade_path]
        self.trade_path = [l.replace('//','/') for l in self.trade_path]

    def get_each_parquet(self, i, show = False): # mode = 'test', 'train'
        assert self.book_path[i].split('=')[-1].split('/')[0]==self.trade_path[i].split('=')[-1].split('/')[0], 'book and trade file not correspondence'
        
        stock_id = self.book_path[i].split('=')[-1].split('/')[0]
                                               
        dfBook = pd.read_parquet(self.book_path[i])
        dfTrade = pd.read_parquet(self.trade_path[i])     
        dfBook0 = dfBook.copy()
        dfTrade0 = dfTrade.copy()
        
        if show:
            display(dfBook.head())
            display(dfTrade.head())
            
        dfVol, dfBook_inter = self._cal_features(dfBook)
        dfVol['stock_id'] = stock_id
        cols = dfVol.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        dfVol = dfVol[cols]
        
        dfTrade, dfTrade_inter = self._cal_features(dfTrade, flag = 'o')
        for df in self._cal_features_time_series(dfVol.time_id, dfBook, show = False):
            dfVol = pd.merge(dfVol, df, on=["time_id"])
        dfVol = pd.merge(dfVol, dfTrade, on=["time_id"])
        print('result')
        display(dfVol)
            
        return dfVol, dfBook_inter, dfTrade_inter, dfBook0, dfTrade0 # return result after process and raw input
    
    def _cal_features_time_series(self, t, dfBook, show = False):
        tmp = {} # dict window_size: list of feature
        for k, v in self.conf['features'].items():
            #print(v)
            for w in v['Ex_win_size']: # if not blank
                if w in tmp:
                    tmp[w].append(v['name'])
                else: 
                    tmp[w] = [v['name']]

        for w, cols in tmp.items():
            yield get_time_series(t, dfBook, cols, w, show = show)
        
    def _cal_features(self, dfBook, flag = 'b', show = False):
        for k, v in self.conf['features'].items():
            if k[0]==flag:
                if v.get('func') is not None:
                    eval(v.get('func'))(dfBook)
                log_run(dfBook, v['name'])
                   
        dfBook = dfBook.dropna()
        col_name = dfBook.columns.tolist()
        col_name = [a for a in col_name if 'logReturn_' in a]
        #col_name = ['logReturn_'+v['name'] for k, v in self.conf['features'].items() if k[0]==flag ] #cal_vol
        dfVol = cal_vol(dfBook, col_name)
        
        # display result
        if show: 
            display(dfBook)
            display(dfVol)
            
        return dfVol, dfBook
    
    def get_all_parquet(self, show = False): # mode = 'test', 'train'
        df_volRes = pd.DataFrame()
        #df_tradeRes = pd.DataFrame()
        
        for lb, lt in zip(self.book_path, self.trade_path):
            assert lb.split('=')[-1][0]==lt.split('=')[-1][0], 'book and trade file not correspondence'
        
            stock_id = lb.split('=')[-1].split('/')[0]
            print('Reading stock id:', stock_id)
            
            dfBook = pd.read_parquet(lb)
            dfTrade = pd.read_parquet(lt)     
        
            if show:
                display(dfBook.head())
                display(dfTrade.head())
            print('feature calculation')
            dfVol, _ = self._cal_features(dfBook)
            dfVol['stock_id'] = stock_id
            cols = dfVol.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            dfVol = dfVol[cols]

            dfTrade, _ = self._cal_features(dfTrade, flag = 'o')
            
            print('time series feature calculation')
            for df in self._cal_features_time_series(dfVol.time_id, dfBook, show = False):
                dfVol = pd.merge(dfVol, df, on=["time_id"])
            dfVol = pd.merge(dfVol, dfTrade, on=["time_id"])
            df_volRes = pd.concat([df_volRes, dfVol])
            
            if show:
                display(df_volRes)
                
        return df_volRes

        
        
    def get_gt(self):
        return pd.read_csv(f"{self.conf['path']}/train.csv")