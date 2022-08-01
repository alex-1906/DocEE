import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModel, BertConfig

from torch.utils.data import DataLoader
from src.data import parse_file, collate_fn
import tqdm
import json
from transformers.optimization import AdamW
import numpy as np
import torch.nn.functional as F
from src.losses import ATLoss
from src.util import process_long_input
from transformers import BertConfig, RobertaConfig, DistilBertConfig, XLMRobertaConfig
from itertools import groupby
#%%
class Encoder(nn.Module):
    def __init__(self, config, model, cls_token_id, sep_token_id, relation_types, feasible_roles):
        super().__init__()
        
        self.config = config
        self.model = model
        
        self.relation_embeddings = nn.Parameter(torch.zeros((59,2*768)))
        torch.nn.init.uniform_(self.relation_embeddings, a=-1.0, b=1.0)            
        self.nota_embeddings = nn.Parameter(torch.zeros((20,2*768)))
        torch.nn.init.uniform_(self.nota_embeddings, a=-1.0, b=1.0)

        self.at_loss = ATLoss()
        self.m = nn.LogSoftmax(dim=1)
        self.nll_loss = nn.NLLLoss()
                
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        self.relation_types = relation_types
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


   
        
    def forward(self, input_ids, attention_mask, candidate_spans, relation_labels, entity_spans, entity_types, entity_ids, batch_text):
        sequence_output, attention = self.encode(input_ids, attention_mask)

        argex_loss = torch.zeros((1),requires_grad=True).to(sequence_output)

        counter = 0
        batch_triples = []
        batch_events = []
                    
        for batch_i in range(sequence_output.size(0)):

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
            objects = []
            for e in range(entity_embeddings.shape[0]):
                if entity_types[batch_i][e].split(".")[-1] == "TRIGGER":
                    triggers.append(e)
                else:
                    objects.append(e)

            for t in triggers:
                for o in objects:
                    relation_candidates.append((t,o))

                    A_s = entity_attentions[t,:,:]
                    A_o = entity_attentions[o,:,:]
                    A = torch.mul(A_o,A_s)
                    q = torch.sum(A,0)
                    a = q / (q.sum() + 1e-30)
                    H_T = sequence_output[batch_i].T
                    c = torch.matmul(H_T,a)
                    localized_context.append(c)

                    concat_emb = torch.cat((entity_embeddings[e],entity_embeddings[o]),0)
                    concat_embs.append(concat_emb)
            if(len(localized_context) == 0):
                continue
            embs = torch.stack(concat_embs)
            
            triggers = list(set(triggers))
            # ---------- Pairwise Comparisons and Predictions ------------

            prescores = torch.matmul(embs,self.relation_embeddings.T)
            prenota_scores = torch.matmul(embs,self.nota_embeddings.T)
            nota_scores = prenota_scores.max(dim=-1,keepdim=True)[0]
            scores = torch.cat((nota_scores, prescores), dim=-1)
            predictions = torch.argmax(scores, dim=-1, keepdim=False)
            
            if self.training:
            # ---------- ATLoss with one-hot encoding for true labels ------------
                # targets = []
                # for r in relation_candidates:
                #     onehot = torch.zeros(len(self.relation_types))
                #     if r in relation_labels[batch_i]:
                #         onehot[relation_labels[batch_i][r]] = 1.0
                #     else:
                #         onehot[0] = 1.0
                #     targets.append(onehot)
                # targets = torch.stack(targets).to(self.model.device)

                #scores = scores.clamp(min=1e-30)
                
                #argex_loss += self.at_loss(scores,targets)
                targets = []
                for r in relation_candidates:
                    if r in relation_labels[batch_i]:
                        targets.append(relation_labels[batch_i][r])
                    else:
                        targets.append(0)
                targets = torch.tensor(targets).to(self.model.device)

                
                
                argex_loss = self.nll_loss(self.m(scores), targets)

                #counter += 1
            
            # ---------- Inference ------------
            triples = []
            for idx,pair in enumerate(relation_candidates):
                triple = {
                    pair:self.relation_types[predictions[idx]]
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

                    if r in self.feasible_roles[event_type]:
                        a_start = entity_spans[batch_i][o][0][0]
                        a_end = entity_spans[batch_i][o][0][1]
                        argument = {
                            'entity_id':entity_ids[batch_i][o],
                            'role':r,
                            'text':"".join(i.strip("##") if "##" in i else " "+i for i in batch_text[batch_i][a_start:a_end]).lstrip(),
                            'start':a_start,
                            'end':a_end,
                        }
                        arguments.append(argument)
                event = {
                    'id':entity_ids[batch_i][t],
                    'event_type':event_type,
                    'trigger':{'start':t_start ,'end':t_end, 'text':"".join(i.strip("##") if "##" in i else " "+i for i in batch_text[batch_i][t_start:t_end]).lstrip()},
                    'arguments':arguments
                }
                events.append(event)
            batch_events.append(events)
        return argex_loss, batch_events