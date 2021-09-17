import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss, MSELoss
from torch.autograd import Variable
from slp.modules.regularization import GaussianNoise
from slp.util import mktensor
import numpy as np

from transformers import *
from transformers.modeling_bert import BertPreTrainingHeads

class DoubleHeadBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
    

    def forward(
        self,
        input_ids=None,
        source=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        #import ipdb; ipdb.set_trace()
        #return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return_dict = False
        outputs = self.bert(
            input_ids#,
            #attention_mask=attention_mask,
            #token_type_ids=token_type_ids,
            #position_ids=position_ids,
            #head_mask=head_mask,
            #inputs_embeds=inputs_embeds,
            #output_attentions=output_attentions,
            #output_hidden_states=output_hidden_states,
            #return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if source == 0 :
            if labels is not None:
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        else:    
            total_loss = None
            if labels is not None: #and next_sentence_label is not None:
                loss_fct = CrossEntropyLoss()
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
                #next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
                total_loss = masked_lm_loss #+ next_sentence_loss
            
            output = (prediction_scores,) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

class BertDCollator(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def pad(self, tensors):
        lengths = torch.tensor([len(s) for s in tensors],
                               device=self.device)
        max_length = torch.max(lengths)
        tensors = (pad_sequence(tensors,
                                batch_first=True,
                                padding_value=self.pad_indx)
                   .to(self.device))
        return tensors

    @staticmethod
    def get_inputs_and_targets(batch):
        inputs, targets, domains = map(list, zip(*batch))
        return inputs, targets, domains

    def __call__(self, batch):
        inputs, targets, domains = self.get_inputs_and_targets(batch)
        inputs = self.pad(inputs)
        inputs = mktensor(inputs, device=self.device, dtype=torch.long)
        targets = mktensor(targets, device=self.device, dtype=torch.long)
        domains = mktensor(domains, device=self.device, dtype=torch.long)
        return inputs, targets.to(self.device), domains.to(self.device)

class BertLMCollator(object):

    #tokenizer: PreTrainedTokenizer
    #mlm = True
    #mlm_probability = 0.15
    #pad_indx = 0
    #device  = 'cpu'
    def __init__(self, tokenizer, pad_indx=0, mlm=True, mlm_probability=0.15, device='cpu'):
        self.device = device
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.tokenizer = tokenizer
        self.pad_indx = pad_indx

    @staticmethod
    def get_inputs_and_targets(batch):
        inputs, targets, domains = map(list, zip(*batch))
        return inputs, targets, domains
    
    def mask_tokens(self, inputs):
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        #import ipdb; ipdb.set_trace()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def pad(self, tensors):
        lengths = torch.tensor([len(s) for s in tensors],
                               device=self.device)
        max_length = torch.max(lengths)
        tensors = (pad_sequence(tensors,
                                batch_first=True,
                                padding_value=self.pad_indx)
                   .to(self.device))
        return tensors
    
    def __call__(self, batch):
        inputs, _, domains = self.get_inputs_and_targets(batch)
        inputs = self.pad(inputs)
        inputs, targets = self.mask_tokens(inputs)
        inputs = mktensor(inputs, device=self.device, dtype=torch.long)
        targets = mktensor(targets, device=self.device, dtype=torch.long)
        domains = mktensor(domains, device=self.device, dtype=torch.long)
        return inputs, targets.to(self.device), domains.to(self.device)

class DoubleBertCollator(object):
    def __init__(self, collatorA, collatorB):
        self.collatorA = collatorA
        self.collatorB = collatorB
    
    @staticmethod
    def get_inputs_and_targets(batch):
        inputs, targets, domains = map(list, zip(*batch))
        return inputs, targets, domains

    def __call__(self, batch):
        inputs, targets, domains = self.get_inputs_and_targets(batch)
        if domains[0]==0:
            return self.collatorA(batch)
        else: 
            return self.collatorB(batch)

class DoubleSubsetRandomSampler(Sampler):
    def __init__(self, indices_source, indices_target, s_dataset_size, num_source, num_target):  
        self.indices_source = indices_source
        self.indices_target = indices_target
        self.s_dataset_size = s_dataset_size
        self.num_source = num_source
        self.num_target = num_target

    def __iter__(self):
        perm = torch.randperm(len(self.indices_source))
        tarperm = torch.randperm(len(self.indices_target))
        T = 0
        t = 0
        for i,s in enumerate(perm,1):
            yield self.indices_source[s]
            if i % self.num_source == 0:
                for j in range(self.num_target):
                    t = T + j
                    yield self.s_dataset_size + self.indices_target[tarperm[t]]
                T = t + 1

    def __len__(self):
        full = int(np.floor((len(self.indices_source) +len(self.indices_target)) / self.num_source))
        last = len(self.indices_source) % self.num_source
        return int(full * self.num_source + last)

class DoubleLoss(nn.Module):
   def __init__(self, loss_fn):
       super(DoubleLoss, self).__init__()
       self.loss_fn = loss_fn
   
   def forward(self, pred, tar, domains):
       #import ipdb; ipdb.set_trace()
       if not domains[0]:
          loss = self.loss_fn(pred, tar)
       else:
          loss = torch.tensor(0)
       return loss
