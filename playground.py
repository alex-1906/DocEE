#%%
import pandas as pd
from src.eval_util import get_df, get_eval
df = pd.read_json('data/WikiEvents/preprocessed/full_eval.json')

dev = pd.read_json('data/WikiEvents/preprocessed/dev.json').set_index('doc_id')

train_medium = pd.read_json('data/WikiEvents/preprocessed/train_medium.json').set_index('doc_id')

#%%
df.loc[train_medium.index]

# %%
df
# %%
train_medium
# %%
dev['len'] = dev['tokens'].apply(len)
train_medium['len'] = train_medium['tokens'].apply(len)

dev['event_len'] = dev['labels'].apply(len)
train_medium['event_len'] = train_medium['labels'].apply(len)

# %%
print(train_medium.event_len.mean())
print(dev.event_len.mean())
print(train_medium.len.mean())
print(dev.len.mean())
# %%
dev
# %%

import pandas as pd

df = pd.read_json("data/WikiEvents/preprocessed/coref_test.json")
print(len(df))
