from helper.dataset import DataLoader
from IPython.display import display
import pandas as pd
import numpy as np
from model.LGBM import LGBMModel
# dl = DataLoader('train')
#
# # getting each stock (sample)
# d1, _ = dl.get_each_parquet(0)
#
# # get groundtruth/target
# gt = dl.get_gt()
# print("Ground Truth")
# display(gt)
#
# # getting all stocks
# df, _ = dl.get_all_parquet()
# print("All stocks")
# display(df)

def main():
    # get groundtruth/target
    dl = DataLoader('train')
    gt = dl.get_gt()

<<<<<<< HEAD
# getting each stock (sample)
''' 
d1 = result dataframe
d_book_inter, d_trade_inter = intermediate dataframe
d_book, d_trade = initial dataframe
'''
#d1, d_book_inter, d_trade_inter,  d_book, d_trade = dl.get_each_parquet(0) # ex: stock_id = 0
=======
>>>>>>> origin/feature/base_model

    df = pd.read_csv("data/update10Sep_feature.csv")
    dataset = pd.merge(df, gt, on=["stock_id", "time_id"])

<<<<<<< HEAD
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

=======
    feature_name = list(dataset.columns)
    feature_name.remove("time_id")
    feature_name.remove("target")
    lgbm_model = LGBMModel(feature_name, "target")
    train_df, test_df = np.split(dataset, [int(.8*len(dataset))])
    score, test_predictions, model = lgbm_model.train(train_df, test_df)
    print(f"Test prediction: {test_predictions}, Score: {score}")

    # test_predictions, rmspe_score  = lgbm_model.train_and_test(train_df,test_df)
    test_predictions, rmspe_score, params = lgbm_model.optimize_params(train_df,test_df)
    print(f"Train and Test prediction: {test_predictions}, Score: {rmspe_score}")
>>>>>>> origin/feature/base_model

if __name__ == "__main__":
    main()
