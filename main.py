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


    df = pd.read_csv("data/update10Sep_feature.csv")
    dataset = pd.merge(df, gt, on=["stock_id", "time_id"])

    feature_name = list(dataset.columns)
    feature_name.remove("time_id")
    feature_name.remove("target")
    lgbm_model = LGBMModel(feature_name, "target")
    train_df, test_df = np.split(dataset, [int(.8*len(dataset))])
    score, test_predictions, model = lgbm_model.train(train_df, test_df)
    print(f"Test prediction: {test_predictions}, Score: {score}")

    test_predictions, rmspe_score = lgbm_model.train_and_test(train_df,test_df)
    print(f"Train and Test prediction: {test_predictions}, Score: {rmspe_score}")

if __name__ == "__main__":
    main()
