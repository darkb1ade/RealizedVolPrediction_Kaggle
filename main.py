from helper.dataset import DataLoader
from IPython.display import display

dl = DataLoader('train')

# getting each stock (sample)
''' 
d1 = result dataframe
d_book_inter, d_trade_inter = intermediate dataframe
d_book, d_trade = initial dataframe
'''
#d1, d_book_inter, d_trade_inter,  d_book, d_trade = dl.get_each_parquet(0) # ex: stock_id = 0


# get groundtruth/target
#gt = dl.get_gt()
#print("Ground Truth")
#display(gt)

# getting all stocks
print('get all stock')
df= dl.get_all_parquet(show = True)
print("All stocks")
display(df)

df.to_csv('update10Sep_feature.csv',index = False)


