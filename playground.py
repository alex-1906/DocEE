#%%
import pandas as pd

df = pd.read_json("data/WikiEvents/preprocessed/train_large.json")

#%%
df = df.drop(df[len(df.labels)==0].index)

# %%
df['len'] = df['labels'].apply(lambda x: len(x))
df = df.drop(df[df['len']==0].index)
# %%
df
# %%
