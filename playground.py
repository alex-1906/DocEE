#%%
import pandas as pd

# %%
roles = pd.read_json("data/Ontology/roles_shared.json")
mentions = pd.read_json("data/Ontology/mention_types.json")
# %%
print(len(roles))
print(len(mentions))
# %%
roles
# %%
mentions
# %%
