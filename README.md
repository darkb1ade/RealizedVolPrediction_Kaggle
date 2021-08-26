# RealizedVolPrediction_Kaggle

#### Load Proceesed Dataset (dataframe format)

'''python
from helper.dataset import DataLoader
dl = DataLoader('train')

# getting each stock (sample)
d1, _ = dl.get_each_parquet(0) 

# getting all stocks
df, _ = dl.get_all_parquet()
display(df)
'''
