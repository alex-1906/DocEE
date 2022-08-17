#%%
import pandas as pd
from src.eval_util import get_df, get_eval
df = pd.read_json('data/WikiEvents/preprocessed/train_medium.json')
df_co = pd.read_json('data/WikiEvents/preprocessed/coref/train_medium.json')

# %%
df
# %%
for idx,row in df.iterrows():
    for e in row.vertexSet:
        if len(e) > 1:
            print(e)
            break
# %%
