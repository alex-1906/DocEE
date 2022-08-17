#%%
import pandas as pd
import json
from src.data import collate_fn, parse_file
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.docred_util import to_docred

#%%

df = pd.read_json("data/WikiEvents/preprocessed/full_eval.json")
df = to_docred(df)
df.to_json("data/WikiEvents/preprocessed/full_eval_docred.json",orient="records")


#%%

df = to_docred(df)
#%%
df.vertexSet[2]
#%%
# co = pd.read_json("data/WikiEvents/preprocessed/full_eval.json")
# with open(f"data/WikiEvents/raw/train.jsonl") as f:
#     lines = f.read().splitlines()
#     df_inter = pd.DataFrame(lines)
#     df_inter.columns = ['json_element']
#     df_inter['json_element'].apply(json.loads)
#     df = pd.json_normalize(df_inter['json_element'].apply(json.loads))
# #%%
# co.iloc[2].event_mentions
# #%%
# df.iloc[205].event_mentions
# # %%
# from src.docred_util import to_docred

# co = to_docred(co,coref=True)
# # %%
# co.iloc[2].vertexSet

# # %%
# co