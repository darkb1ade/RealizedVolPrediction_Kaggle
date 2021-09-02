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
        
        
        if show:
            display(dfBook.head())
            display(dfTrade.head())
            
        dfVol = self._cal_features(dfBook)
        dfVol['stock_id'] = stock_id
        cols = dfVol.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        dfVol = dfVol[cols]
        display(dfVol)
            
        return dfVol, dfTrade
    
    def _cal_features(self, dfBook, show = False):
        for k, v in self.conf['features'].items():
            eval(v.get('func'))(dfBook)
            log_run(dfBook, v['name'])
        dfBook = dfBook.dropna()
        col_name = ['logReturn_'+v['name'] for k, v in self.conf['features'].items()]
        dfVol = cal_vol(dfBook, ['logReturn_WAP1','logReturn_WAP2'])
        
        # display result
        if show: 
            display(dfBook)
            display(dfVol)
            
        return dfVol
    
    def get_all_parquet(self, show = False): # mode = 'test', 'train'
        df_volRes = pd.DataFrame()
        df_tradeRes = pd.DataFrame()
        for lb, lt in zip(self.book_path, self.trade_path):
            assert lb.split('=')[-1][0]==lt.split('=')[-1][0], 'book and trade file not correspondence'
        
            stock_id = lb.split('=')[-1].split('/')[0]
            print('Reading stock id:', stock_id)
            
            dfBook = pd.read_parquet(lb)
            dfTrade = pd.read_parquet(lt)     
        
            if show:
                display(dfBook.head())
                display(dfTrade.head())
            
            dfVol = self._cal_features(dfBook)
            dfVol['stock_id'] = stock_id
            dfTrade['stock_id'] = stock_id
            #display(dfVol)
            df_volRes = pd.concat([df_volRes, dfVol])
            df_tradeRes = pd.concat([df_tradeRes, dfTrade])
        cols = df_volRes.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df_volRes = df_volRes[cols]
        return df_volRes, df_tradeRes
    
    def get_gt(self):
        return pd.read_csv(f"{self.conf['path']}/train.csv")