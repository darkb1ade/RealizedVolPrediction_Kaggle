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
    df = pd.read_csv("data/update8Sep_feature.csv")
    feature_name = list(df.columns)
    feature_name.remove("time_id")
    feature_name.remove("logReturn_price")
    lgbm_model = LGBMModel(feature_name, "logReturn_price")
    train_df, test_df = np.split(df, [int(.8*len(df))])
    test_predictions, score = lgbm_model.train(train_df, test_df)
    print(f"Test prediction: {test_predictions}, Score: {score}")

if __name__ == "__main__":
    main()