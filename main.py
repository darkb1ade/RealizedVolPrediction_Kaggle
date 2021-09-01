from helper.dataset import DataLoader
from IPython.display import display

dl = DataLoader('train')

# getting each stock (sample)
d1, _ = dl.get_each_parquet(0)

# get groundtruth/target
gt = dl.get_gt()
display(gt)

# getting all stocks
df, _ = dl.get_all_parquet()
display(df)