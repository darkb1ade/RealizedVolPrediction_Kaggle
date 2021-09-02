# RealizedVolPrediction_Kaggle

## Structure:
### Dataset (`helper/dataset.py`)

#### Load Processed Dataset (dataframe format)

```python
from helper.dataset import DataLoader
dl = DataLoader('train')

# getting each stock (sample)
d1, _ = dl.get_each_parquet(0) 

# get groundtruth/target
gt = dl.get_gt()
display(gt)

# getting all stocks
df, _ = dl.get_all_parquet()
display(df)
```


### Preprocessing (`model/features.py`)
Contains function to calculate all features. Need to add function name in `config/main.yaml` to automatically load 
these features when using `DataLoader`

#### Main features:
- WAP1
- WAP2
- ???

### Model (`model/BaseModel.py`)
Note: Any new model should have BaseModel as a base class 

Current implementation:
- LightGBM (Not finished): Performance = ??
- XGBoost (Not implemented)
- Neural Network (Not implemented)