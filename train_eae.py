# %%
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModel, BertConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data import parse_file, collate_fn
import tqdm
import json
from transformers.optimization import AdamW
import numpy as np
import pandas as pd
import torch.nn.functional as F
from src.losses import ATLoss
from src.util import process_long_input
from transformers import BertConfig, RobertaConfig, DistilBertConfig, XLMRobertaConfig
from src.eval_util import get_eae_eval
from src.util import safe_div,compute_f1
from itertools import groupby
import wandb
import argparse
import string 
import random

random_string = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(10))
 
print(random_string)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=4, help="eval batch size")
parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint path")
parser.add_argument("--model", type=str, default="base", help="model")
parser.add_argument("--k_mentions", type=int, default=50, help="number of mention spans to perform relation extraction on")
parser.add_argument("--pooling", type=str, default="mean", help="mention pooling method (mean, max)")
parser.add_argument("--epochs", type=int, default=1, help="number of epochs")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate")
parser.add_argument("--project", type=str, default="my-test-project", help="project name for wandb")


args = parser.parse_args()
wandb.init(project=args.project)
wandb.config.update(args)
wandb.config.identifier = random_string


'''wandb.config = {
  "learning_rate": args.learning_rate,
  "epochs": args.epochs,
  "batch_size": args.batch_size,
  "shuffle": False
}'''
# %%

#########################
## Model Configuration ##
#########################
language_model = 'bert-base-uncased'
lm_config = AutoConfig.from_pretrained(
    language_model,
    num_labels=10,
)
lm_model = AutoModel.from_pretrained(
    language_model,
    from_tf=False,
    config=lm_config,
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

df = pd.read_json('data/WikiEvents/train_eval.json')


with open("data/Ontology/roles.json") as f:
    relation_types = json.load(f)
with open("data/Ontology/trigger_entity_types.json") as f:
    mention_types = json.load(f)
with open("data/Ontology/feasible_roles.json") as f:
    feasible_roles = json.load(f)

max_n = 9
train_loader = DataLoader(
    parse_file("data/WikiEvents/train_docred_format_small.json",
    tokenizer=tokenizer,
    relation_types=relation_types,
    max_candidate_length=max_n),
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collate_fn)

# %%
class Encoder(nn.Module):
    def __init__(self, config, model, cls_token_id, sep_token_id, relation_types, mention_types, feasible_roles, soft_mention = True):
        super().__init__()
        
        self.soft_mention = soft_mention

        self.config = config
        self.model = model

        self.entity_anchor = nn.Parameter(torch.zeros((66, 768)))
        torch.nn.init.uniform_(self.entity_anchor, a=-1.0, b=1.0)
        
        self.relation_embeddings = nn.Parameter(torch.zeros((57,3*768)))
        torch.nn.init.uniform_(self.relation_embeddings, a=-1.0, b=1.0)            
        self.nota_embeddings = nn.Parameter(torch.zeros((20,3*768)))
        torch.nn.init.uniform_(self.nota_embeddings, a=-1.0, b=1.0)


        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        self.ce_loss = nn.CrossEntropyLoss()
        self.at_loss = ATLoss()
        

        self.k_mentions = args.k_mentions
                
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        self.relation_types = relation_types
        self.mention_types = mention_types
        self.feasible_roles = feasible_roles
        
        
    def encode(self, input_ids, attention_mask):
        config = self.config
        if type(config) == BertConfig or type(config) == DistilBertConfig:
            start_tokens = [self.cls_token_id]
            end_tokens = [self.sep_token_id]
        elif type(config) == RobertaConfig or type(config) == XLMRobertaConfig:
            start_tokens = [self.cls_token_id]
            end_tokens = [self.sep_token_id, self.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def forward(self, input_ids, attention_mask, candidate_spans, relation_labels, entity_spans, entity_types, entity_ids):
        sequence_output, attention = self.encode(input_ids, attention_mask)
        loss = torch.zeros((1)).to(sequence_output)
        counter = 0
        batch_triples = []
        batch_events = []
        batch_text = []

        
        for batch_i in range(sequence_output.size(0)):
            #entity_spans = entity_spans[batch_i]
            text_i = self.tokenizer.convert_ids_to_tokens(input_ids[batch_i])
            batch_text.append(text_i)

            # ---------- Pooling Entity Embeddings and Attentions ------------
            entity_embeddings = []
            entity_attentions = []
            for ent in entity_spans[batch_i]:
                ent_embedding = torch.mean(sequence_output[batch_i, ent[0][0]:ent[0][1],:],0)
                entity_embeddings.append(ent_embedding)
                ent_attention = torch.mean(attention[batch_i,:,ent[0][0]:ent[0][1],:],1)
                entity_attentions.append(ent_attention)
            if(len(entity_embeddings) == 0):
                continue
            entity_embeddings = torch.stack(entity_embeddings)
            entity_attentions = torch.stack(entity_attentions)

                
            # ---------- Localized Context Pooling ------------
            relation_candidates = []
            localized_context = []
            concat_embs = []
            triggers = []
            for s in range(entity_embeddings.shape[0]):
                if entity_types[batch_i][s].split(".")[-1] != "TRIGGER":
                    continue
                triggers.append(s)
                for o in range(entity_embeddings.shape[0]):
                    if s != o:

                        relation_candidates.append((s,o))

                        A_s = entity_attentions[s,:,:]
                        A_o = entity_attentions[o,:,:]
                        A = torch.mul(A_o,A_s)
                        q = torch.sum(A,0)
                        a = q / q.sum()
                        H_T = sequence_output[batch_i].T
                        c = torch.matmul(H_T,a)
                        localized_context.append(c)

                        concat_emb = torch.cat((entity_embeddings[s],entity_embeddings[o],c),0)
                        concat_embs.append(concat_emb)
            if(len(localized_context) == 0):
                continue
            localized_context = torch.stack(localized_context)
            embs = torch.stack(concat_embs)
            
            triggers = list(set(triggers))
            # ---------- Pairwise Comparisons and Predictions ------------

            scores = torch.matmul(embs,self.relation_embeddings.T)
            nota_scores = torch.matmul(embs,self.nota_embeddings.T)
            nota_scores = nota_scores.max(dim=-1,keepdim=True)[0]
            scores = torch.cat((nota_scores, scores), dim=-1)
            predictions = torch.argmax(scores, dim=-1, keepdim=False)
            #Achtung: NOTA wird an 0. Stelle gesetzt
            
            #relation_labels.to(sequence_output)
            if self.training:
            # ---------- ATLoss with one-hot encoding for true labels ------------
                targets = []
                for r in relation_candidates:
                    onehot = torch.zeros(len(relation_types))
                    if r in relation_labels[batch_i]:
                        onehot[relation_labels[batch_i][r]] = 1.0
                    targets.append(onehot)
                targets = torch.stack(targets).to(self.model.device)
                loss += self.at_loss(scores,targets)
                counter += 1
            
            # ---------- Inference ------------
            triples = []
            for idx,pair in enumerate(relation_candidates):
                triple = {
                    pair:relation_types[predictions[idx]]
                }
                triples.append(triple)
            batch_triples.append(triples)
                
            events = []
            for t,v in groupby(triples,key=lambda x:next(iter(x.keys()))[0]):
                t_word = entity_types[batch_i][t]
                t_start = entity_spans[batch_i][t][0][0]
                t_end = entity_spans[batch_i][t][0][1]
                event_type = t_word.split(".TRIGGER")[0]
                event_id = entity_ids[batch_i][t]

                arguments = []
                for d in v:
                    dic = next(iter(d.items()))
                    o = dic[0][1]
                    r = dic[1]

                    if r in mymodel.feasible_roles[event_type]:
                        a_start = entity_spans[batch_i][o][0][0]
                        a_end = entity_spans[batch_i][o][0][1]
                        argument = {
                            'entity_id':entity_ids[batch_i][o],
                            'role':r,
                            'text':batch_text[batch_i][a_start:a_end][0],
                            'start':a_start,
                            'end':a_end,
                        }
                        arguments.append(argument)

                event = {
                    'id': entity_ids[batch_i][t],
                    'event_type':event_type,
                    'trigger': {'start':t_start ,'end':t_end, 'text':batch_text[batch_i][t_start:t_end][0]},
                    'arguments':arguments
                }
                events.append(event)
            batch_events.append(events)
        if(counter == 0):
                return torch.autograd.Variable(loss,requires_grad=True)
        else:
            return loss/counter, batch_triples, batch_events
        #TODO: Irgendetwas sollte implizit berücksichtigt werden?

#%%
mymodel = Encoder(lm_config,
                lm_model,
                cls_token_id=tokenizer.convert_tokens_to_ids(tokenizer.cls_token), 
                sep_token_id=tokenizer.convert_tokens_to_ids(tokenizer.sep_token),
                relation_types=relation_types,
                mention_types=mention_types,
                feasible_roles=feasible_roles,
                soft_mention=False
                 )
optimizer = AdamW(mymodel.parameters(), lr=args.learning_rate, eps=1e-6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mymodel.to(device)
mymodel.train()
#%%
# ---------- Training ------------
step_global = 0

losses = []
event_list = []
doc_id_list = []
with tqdm.tqdm(train_loader) as progress_bar:
    for sample in progress_bar:
        step_global += 1
        token_ids, input_mask, entity_spans, entity_types, entity_ids, relation_labels, text, token_map, candidate_spans, doc_ids = sample
        


        mymodel.train()
        loss ,triples, events = mymodel(token_ids.to(device), input_mask.to(device), candidate_spans, relation_labels, entity_spans, entity_types, entity_ids)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

        progress_bar.set_postfix({"L":f"{loss.item():.2f}"})
        wandb.log({"eae_loss": loss.item()})

        #insert events into dataframe
        for batch_i in range(token_ids.shape[0]):
            doc_id_list.append(doc_ids[batch_i])
            event_list.append(events[batch_i])
#%%
df['pred_events'] = pd.Series(event_list,index=doc_id_list)
df['pred_events'] = df['pred_events'].fillna("").apply(list)


#%%
idf_pred, idf_gold, idf_h_matched, idf_c_matched = 0,0,0,0
clf_pred, clf_gold, clf_h_matched, clf_c_matched = 0,0,0,0

for idx, row in df.iterrows(): 
    events = row['pred_events']
    gold_events = row['event_mentions']

    for ge,e in zip(gold_events,events):
        for g_arg in ge['arguments']:
            for arg in e['arguments']:
                #----- Head Matches -----
                if g_arg['entity_id'] == arg['entity_id']:
                    idf_h_matched += 1
                    idf_c_matched += 1
                    if g_arg['role'] == arg['role']:
                        clf_h_matched += 1
                        clf_c_matched += 1
                    
                #----- Coref Matches -----
                for coref in g_arg['corefs']:
                    if coref['entity_id'] == arg['entity_id']:
                        idf_c_matched += 1
                        if g_arg['role'] == arg['role']:
                            clf_c_matched += 1
            idf_gold += 1
        for arg in e['arguments']:
            idf_pred += 1
        clf_pred, clf_gold = idf_pred, idf_gold

        #----- Identification P,R,F1 -----
        idf_h_p, idf_h_r, idf_h_f1 = compute_f1(idf_pred, idf_gold, idf_h_matched)
        idf_c_p, idf_c_r, idf_c_f1 = compute_f1(idf_pred, idf_gold, idf_c_matched)

        #----- Classification P,R,F1 -----
        clf_h_p, clf_h_r, clf_h_f1 = compute_f1(idf_pred, idf_gold, clf_h_matched)
        clf_c_p, clf_c_r, clf_c_f1 = compute_f1(idf_pred, idf_gold, clf_c_matched)

#%%
print("***** Identification Report *****")
print(f"*Head* Matches: {idf_h_matched} Precision: {idf_h_p:.2f} Recall: {idf_h_r:.2f} F1-Score: {idf_h_f1:.2f}")
print(f"*Coref* Matches: {idf_c_matched} Precision: {idf_c_p:.2f} Recall: {idf_c_r:.2f} F1-Score: {idf_c_f1:.2f}")
print()
print("***** Classification Report *****")
print(f"*Head* Matches: {clf_h_matched} Precision: {clf_h_p:.2f} Recall: {clf_h_r:.2f} F1-Score: {clf_h_f1:.2f}")
print(f"*Coref* Matches: {clf_c_matched} Precision: {clf_c_p:.2f} Recall: {clf_c_r:.2f} F1-Score: {clf_c_f1:.2f}")
#%%

# %%
torch.save(mymodel.state_dict(), f"checkpoints/{args.project}_{random_string}.pt")

#mymodel.load_state_dict(torch.load(f"checkpoints/{args.project}_{random_string}.pt"))
# %%
