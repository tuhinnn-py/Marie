# %% [code]
''' Download spacy German and French extensions '''
!python -m spacy download de
!python -m spacy download fr

# %% [code]
"""
Using torchtext.datasets.Multi30k Dataset for machine to machine translation
"""

import spacy
from torchtext.data import BucketIterator, Field
from torchtext.datasets import Multi30k
from typing import List
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
e = spacy.load('en')
g = spacy.load('de')

def _tokenize(text : str, src : bool = False) -> List:
    _ = e if src else g
    return [tok.text for tok in _.tokenizer(text)]

'''
Preprocessing data through torchtext.data.Field's __init__ pipeline
'''

e_ = Field(sequential = True, use_vocab = True, tokenize = lambda x : _tokenize(x, True), lower = True, init_token = '<sos>', eos_token = '<eos>', pad_first = True)
g_ = Field(sequential = True, use_vocab = True, tokenize = lambda x : _tokenize(x, False), lower = True, init_token = '<sos>', eos_token = '<eos>', pad_first = True)

train_data, validation_data, test_data = Multi30k.splits(exts = ('.en', '.de'), fields = (e_, g_))

e_.build_vocab(train_data, max_size = 100000, min_freq = 0)
g_.build_vocab(train_data, max_size = 100000, min_freq = 0)

train_iterator, validation_iterator, test_iterator = BucketIterator.splits((train_data, validation_data, test_data), batch_size = 16, device = device)

# %% [code]
""" 
Using Custom Dataset which has been preprocessed and stored in a csv file 
"""

''' Reading csv files to buffer '''
import pandas as pd
import spacy
from torchtext.data import BucketIterator, Field, TabularDataset
from typing import List
import torch

x = pd.read_csv("../input/-tutils/eng-fr.csv").drop(columns = "Unnamed: 0")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
e = spacy.load('en')
f = spacy.load('fr')

def _tokenize(text : str, src : bool = False) -> List:
    _ = e if src else f
    return [tok.text for tok in _.tokenizer(text)]

'''
Preprocessing data through torchtext.data.Field's __init__ pipeline
'''

e_ = Field(sequential = True, use_vocab = True, tokenize = lambda x : _tokenize(x, True), lower = True, init_token = '<sos>', eos_token = '<eos>', pad_first = True)
f_ = Field(sequential = True, use_vocab = True, tokenize = lambda x : _tokenize(x, False), lower = True, init_token = '<sos>', eos_token = '<eos>', pad_first = True)

fields = {'eng' : ('src', e_), 'fr' : ('trg', f_)}
train_data = TabularDataset.splits(path = '../input/-tutils', train = 'eng-fr.csv', format = 'csv', fields = fields)

e_.build_vocab(train_data[0], max_size = 100000, min_freq = 0)
f_.build_vocab(train_data[0], max_size = 100000, min_freq = 0)

train_iterator = BucketIterator.splits(train_data, shuffle = True, batch_size = 32, device = device)[0]

# %% [code]
"""
Instantiate a Transformer object and predefine the Adam optimizer and training hyperparameters 
"""
from torch_utils import Transformer, generate_square_subsequent_mask, generate_padding_mask
import torch.nn as nn
import torch.nn.functional as F

EPOCHS = 3
SRC_VOCAB_SIZE = len(e_.vocab)
TGT_VOCAB_SIZE = len(f_.vocab)

SRC_PAD_IDX = e_.vocab.stoi['<pad>']
TRG_PAD_IDX = f_.vocab.stoi['<pad>']

model = Transformer(5000, SRC_PAD_IDX, TRG_PAD_IDX, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, nhead = 16, num_encoder_layers = 3, num_decoder_layers = 3, dim_feedforward = 2048, dropout = 0.08).to(device = device)
optim = torch.optim.Adam(model.parameters(), lr = 1e-4, betas = (0.9, 0.98), eps = 1e-08)

"""
def generate_padding_mask(src : Tensor, src_field : Field, pad : str = '<pad>', num_heads : int = 8)->Tensor:
    '''
    Shape : 
        - src : N, S
    where N is the batch size and S is the source sequence length
    '''
    bsz = src.shape[0]
    pad_idx = src_field.vocab.stoi[pad]
    out = (src == pad_idx).float()
    
    out_ = torch.reshape(out, (bsz, -1, 1))
    out = torch.reshape(out, (bsz, 1, -1))
    out = torch.bmm(out_, out)

    out = out.masked_fill(out == 1, float('-inf')).masked_fill(out == 0, float(0.0))  
    return torch.cat([out for _ in range(num_heads)], dim = 0)
"""

def train_model(model : nn.Module = model, epochs : int = EPOCHS, bkpt_idx : int  = 100) -> None:
    model.train()
    
    
    def schedule_lr(optim, step_num : int, d_model : int = 512, warmup_steps = 4000) -> None:
        lr_ =  d_model ** (-0.5) * min(step_num ** (-0.5), step_num * warmup_steps ** (1.5))
        for g in optim.param_groups:
                g['lr'] = lr_
        return lr_
    
    for i in range(epochs):
        for idx, batch in enumerate(train_iterator):
            src, tgt = batch.src.long(), batch.trg.long()
            '''
            Shape : 
                src - (S, N)
                tgt - (T, N)
            where S is the source length, T is the target length, N is the number of batches
            assert src.shape[1] == tgt.shape[1], Invalid shape Exception
            '''
            #lr_ = schedule_lr(optim, idx * src.shape[1] + 1)
            trg = tgt[:-1, :]
            targets = tgt[1:, ].contiguous().view(-1)
            
            #src_mask = generate_padding_mask(src.transpose(0, 1), e_)
            tgt_mask = generate_square_subsequent_mask(trg.shape[0]).to(device)
            #tgt_mask = torch.stack([tgt_mask for _ in range(tgt.shape[1] * 8)], dim = 0) + generate_padding_mask(trg.transpose(0, 1), g_)
            src_key_padding_mask = generate_padding_mask(src.transpose(0, 1), e_)
            tgt_key_padding_mask = generate_padding_mask(trg.transpose(0, 1), f_)
            
            '''
            Shape : 
                tgt_mask - (T, T)
                src_key_padding_mask - (N, S)
                tgt_key_padding_mask - (N, T)
            where T is the target length which is broadcasted over N, N is the number of batches
            '''
            
            out = model(src, trg, src_key_padding_mask = src_key_padding_mask, tgt_mask = tgt_mask, tgt_key_padding_mask = tgt_key_padding_mask)    
            optim.zero_grad()

            loss = F.cross_entropy(out.view(-1, out.shape[-1]), targets, ignore_index = TRG_PAD_IDX)
            loss.backward()
            optim.step()
            
        if((i + 1) % bkpt_idx == 0):
            print(f"Epoch : {i + 1}   Loss : {loss.item()} LR : {1e-04}")
            #state_dict = {'mod_' : model.state_dict(), 'optim_' : optim.state_dict(), 'loss' : loss.item()}
            #torch.save(state_dict, "./ckpt.pt")

# %% [code]
"""
Train the instantiated Transformer class
"""
train_model(bkpt_idx = 1)

# %% [code]
''' 
Visualise some of the training examples 
'''

batch = next(iter(train_iterator))
" ".join(list(map(lambda x : e_.vocab.itos[x], list(batch.src[:, 0].cpu().numpy()))))

# %% [code]
''' 
Visualize inferred translations from the model
'''

from torch import Tensor
from torchtext.data import Field
import torch.nn.functional as F
g_ = f_   
def decode_str(text : Tensor) -> str:
    text = text[1:-1, :].transpose(0, 1)[0]
    print(" >> ", " ".join(list(map(lambda x : g_.vocab.itos[x], list(text.numpy())))), sep = " ")

def get_custom_str(text : str, field : Field, device : str = device) -> Tensor:
    text = (lambda x : _tokenize(x, True))(text.lower())
    res_ = torch.zeros((len(text) + 2, 1)).to(device)
    
    out = e_.numericalize([text]).to(device)
    res_[1:-1, :] = out
    
    res_[0] = torch.full((1, 1), fill_value = field.vocab.stoi['<sos>'], dtype = res_.dtype, device = device)
    res_[-1] = torch.full((1, 1), fill_value = field.vocab.stoi['<eos>'], dtype = res_.dtype, device = device)
    
    return res_.long()

org_str = "Lift your hand if you can hear me." 
print(" >> ", org_str)
src = get_custom_str(org_str, e_)
trg = torch.full((len(g_.vocab), 1), fill_value = g_.vocab.stoi['<pad>'], dtype = torch.long, device = device)
trg[2] = torch.full((1, 1), fill_value = g_.vocab.stoi['<sos>'], dtype = trg.dtype, device = device)

for idx in range(len(g_.vocab)):
    tgt_mask = generate_square_subsequent_mask(idx + 3).to(device)
    src_key_padding_mask = generate_padding_mask(src.transpose(0, 1), e_)
    tgt_key_padding_mask = generate_padding_mask(trg[:idx + 3, :].transpose(0, 1), g_)

    out = torch.argmax(F.softmax(model(src, trg[:idx + 3, :], src_key_padding_mask = src_key_padding_mask, tgt_mask = tgt_mask, tgt_key_padding_mask = tgt_key_padding_mask), dim = -1), dim = -1)
    trg[idx + 3, :] = out[-1, :]
    
    if out[-1, 0] == g_.vocab.stoi['<eos>']:
        decode_str(trg[2:idx + 4, :].cpu())
        break    

# %% [code]
""" 
Loading a pretrained transformer
"""

import torch
ckpt_ = torch.load("../input/transformer/ckpt.pt")

from torch_utils import Transformer, generate_square_subsequent_mask, generate_padding_mask
import torch.nn as nn
import torch.nn.functional as F

SRC_VOCAB_SIZE = len(e_.vocab)
TGT_VOCAB_SIZE = len(g_.vocab)

SRC_PAD_IDX = e_.vocab.stoi['<pad>']
TRG_PAD_IDX = g_.vocab.stoi['<pad>']
model = Transformer(SRC_PAD_IDX, TRG_PAD_IDX, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, nhead = 16, num_encoder_layers = 3, num_decoder_layers = 3, dim_feedforward = 1024, dropout = 0.0).to(device = device)
model.load_state_dict(ckpt_["mod_"], strict = True)

# %% [code]
''' 
Convert text files into Tabular Dataset
'''

import pandas as pd

with open("../input/englishfrenchtranslation/english.txt", "r") as e_open:
    with open("../input/englishfrenchtranslation/french.txt", "r") as f_open:
        ''' Delimiter lambda function '''
        func = lambda txt : txt[:-1]
        eng, fr = list(map(func, e_open.readlines())), list(map(func, f_open.readlines()))
        assert len(eng) == len(fr), "Lengths of source and target must be same"
        
        data = {'eng' : eng, 'fr' : fr}
        x = pd.DataFrame(data = data)
        
        ''' Write the data to a csv file '''
        x.to_csv("./eng-fr.csv")