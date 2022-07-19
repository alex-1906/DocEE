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
    def __init__(self, config, model, cls_token_id, sep_token_id, relation_types, mention_types, feasible_roles, soft_mention = True, at_inference=True):
        super().__init__()
        print("updated encoder")
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
        

        self.k_mentions = 50
                
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        self.relation_types = relation_types
        self.mention_types = mention_types
        self.feasible_roles = feasible_roles

        self.at_inference = at_inference
        
        
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

    def forward(self, input_ids, attention_mask, candidate_spans, relation_labels, entity_spans, entity_types, entity_ids,batch_text):
        sequence_output, attention = self.encode(input_ids, attention_mask)
        loss = torch.zeros((1)).to(sequence_output)
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
                    onehot = torch.zeros(len(self.relation_types))
                    if r in relation_labels[batch_i]:
                        onehot[relation_labels[batch_i][r]] = 1.0
                    targets.append(onehot)
                targets = torch.stack(targets).to(self.model.device)
                loss += self.at_loss(scores,targets)
                counter += 1
            
            # ---------- Inference ------------
            if self.at_inference:
                pred_classes = self.at_loss.get_label(scores)
                triples = []
                for idx,pair in enumerate(relation_candidates):
                    preds = pred_classes[idx]
                    preds = [i[0] for i in preds.nonzero().tolist()]
                    #seltener Fall, dass ein Pair mehrere Labels hat
                    for p in preds:
                        triple = {
                            pair:self.relation_types[p]
                        }
                        triples.append(triple)
                batch_triples.append(triples)
            else:
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
                    'id': entity_ids[batch_i][t],
                    'event_type':event_type,
                    'trigger': {'start':t_start ,'end':t_end, 'text':"".join(i.strip("##") if "##" in i else " "+i for i in batch_text[batch_i][t_start:t_end]).lstrip()},
                    'arguments':arguments
                }
                events.append(event)
            batch_events.append(events)
        if(counter == 0):
                return torch.autograd.Variable(loss,requires_grad=True), batch_triples, batch_events
        else:
            return loss/counter, batch_triples, batch_events
        #TODO: Irgendetwas sollte implizit berücksichtigt werden?
