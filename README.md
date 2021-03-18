# **Marie**

<p align="center">
  <img width="700" height="448" src="https://github.com/tuhinnn-py/Marie/blob/main/Transformer.jpg">
</p>

### **Overview** 
---
*Before "**Attention Is All You Need**", Sequence-to-Sequence translations were dependent on complex recurrent architectures like Recurrent Neural Networks(RNNs), Long Short Term Memory(LSTMs) or Gated Recurrent Units(GRUs) with Bidirectional Long Short Term Memory architectures(BiLSTM) being the state-of-the-art model architecture for Natural Language Processing(NLP) tasks.* 

*Infact, some papers even suggested the use of a convolutional architecture to character or word level embeddings to effectively capture grassroot dependencies and relationships depending on the kernel window size(N), thus mimicing a N-gram language model for language modelling tasks. 
However recurrent neural networks come with a lot of problems, some of them being*

- *Recuurent Neural Networks are very difficult to train. Instead of **Stochastic Gradient Descent**, something called a **Truncated Gradient Descent Algorithm** is followed to roughly estimate the gradients for the entire instance, incase of large sentences for example.*
- *Though hypothetically, RNNs are capable of capturing long term **dependencies**(infact in theory, they work fine over an infinite window size as well), RNNs fail to capture long term dependencies. Complex architectures like BiLSTMs and GRUs come as an improvement, but recurrence simply doesn't cover it for large sentences.*
- *Depending on the singular values of weight matrices, gradients seem to explode(**Exploding Gradient Problem**) or diminish to zero(**Vanishing Gradient Problem**).*
- *However the biggest con probably might be the fact that RNNs are not **parallelizable**. Due to their inherent reccurent nature, where the output for the N-1th token serves as an additional input along with the Nth token for the Nth step, RNNs cannot be parallelized.*

*As an improvement to the previously exisiting recurrent architectures, in 2017 Google AI research(Asish Vaswani et. al.) published their groundbreaking transformer architecture in the paper "**Attention Is All You Need**, which is inherently parallelizable and can also capture really long term dependencies due to a mechanism, that the authors call in the paper "**Multi-Head Attention**".*

<p align="center">
  <img width="465" height="565" src="https://github.com/tuhinnn-py/Marie/blob/main/Transformer.png">
</p>

## Implementation: 
*In this repository you'll find .ipynb files demonstrating Transformer implementations from scratch in PyTorch. I'll try my best to walk you through the code as we go along. Firstly, let's import the modules that we would require in our project.*
```
  ''' Based on the novel Transformer architecture as described in the paper "Attention Is All You Need", Asish Vaswani et. al. 2017 '''

  import copy
  import math
  import torch
  import torch.nn as nn
  import torch.nn.functional as F

  from torch import Tensor
  from typing import Optional, Any, Callable
  from torch.nn.modules.activation import MultiheadAttention
  from torch.nn.init import xavier_uniform_
  from torchtext.data import Field

  # %% [code]
  # __utils__

  def _get_clones(module : nn.Module, N : int) -> nn.ModuleList:
      return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

  def generate_square_subsequent_mask(sz: int) -> Tensor:
      mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
      mask = mask.float().masked_fill(mask == 0, -1e9).masked_fill(mask == 1, float(0.0))
      return mask

  def generate_padding_mask(src : Tensor, src_field : Field, pad : str = '<pad>')->Tensor:
      '''
      Shape : 
          - src : N, S
      where N is the batch size and S is the source sequence length
      '''
      pad_idx = src_field.vocab.stoi[pad]
      return src == pad_idx

  # %% [code]
  class Embedding(nn.Module):

      def __init__(self, padding_idx : int = None, vocab_size : int = 10000, d_model : int = 512) -> None:
          super(Embedding, self).__init__()
          self.embed = nn.Embedding(vocab_size, d_model, padding_idx = padding_idx)

      def forward(self, x : Tensor) -> Tensor:
          return self.embed(x)

  # %% [code]
  class PositionalEncoding(nn.Module):

      def __init__(self, d_model : int = 512, dropout : float = 0.1, max_len : int = 5000) -> None:
          super(PositionalEncoding, self).__init__()
          self.dropout = nn.Dropout(p=dropout)

          pe = torch.zeros(max_len, d_model)
          position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
          div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
          pe[:, 0::2] = torch.sin(position * div_term)
          pe[:, 1::2] = torch.cos(position * div_term)
          pe = pe.unsqueeze(0).transpose(0, 1)
          self.register_buffer('pe', pe)

      def forward(self, x : Tensor) -> Tensor:
          x = x + self.pe[:x.size(0), :]
          return self.dropout(x)

  # %% [code]
  class TransformerEncoderLayer(nn.Module):

      def __init__(self, d_model : int = 512, nhead : int = 8, dim_feedforward : int = 2048, dropout : float = 0.1, activation :str = "relu"):

          super(TransformerEncoderLayer, self).__init__()
          self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, add_bias_kv=True, add_zero_attn=True)
          self.linear1 = nn.Linear(d_model, dim_feedforward)
          self.dropout = nn.Dropout(dropout)
          self.linear2 = nn.Linear(dim_feedforward, d_model)

          self.norm1 = nn.LayerNorm(d_model)
          self.norm2 = nn.LayerNorm(d_model)
          self.dropout1 = nn.Dropout(dropout)
          self.dropout2 = nn.Dropout(dropout)

          self.activation = _get_activation_fn(activation)

      def __setstate__(self, state):
          if 'activation' not in state:
              state['activation'] = F.relu
          super(TransformerEncoderLayer, self).__setstate__(state)

      def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:

          src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
          src = src + self.dropout1(src2)
          src = self.norm1(src)
          src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
          src = src + self.dropout2(src2)
          src = self.norm2(src)
          return src

  # %% [code]
  class TransformerDecoderLayer(nn.Module):

      def __init__(self, d_model : int = 512, nhead : int = 8, dim_feedforward: int = 2048, dropout : float = 0.1, activation : str = "relu"):

          super(TransformerDecoderLayer, self).__init__()
          self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, add_bias_kv=True, add_zero_attn=True)
          self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, add_bias_kv=True, add_zero_attn=True)

          self.linear1 = nn.Linear(d_model, dim_feedforward)
          self.dropout = nn.Dropout(dropout)
          self.linear2 = nn.Linear(dim_feedforward, d_model)

          self.norm1 = nn.LayerNorm(d_model)
          self.norm2 = nn.LayerNorm(d_model)
          self.norm3 = nn.LayerNorm(d_model)
          self.dropout1 = nn.Dropout(dropout)
          self.dropout2 = nn.Dropout(dropout)
          self.dropout3 = nn.Dropout(dropout)

          self.activation = _get_activation_fn(activation)

      def __setstate__(self, state):
          if 'activation' not in state:
              state['activation'] = F.relu
          super(TransformerDecoderLayer, self).__setstate__(state)

      def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

          tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
          tgt = tgt + self.dropout1(tgt2)
          tgt = self.norm1(tgt)
          tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                     key_padding_mask=memory_key_padding_mask)[0]
          tgt = tgt + self.dropout2(tgt2)
          tgt = self.norm2(tgt)
          tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
          tgt = tgt + self.dropout3(tgt2)
          tgt = self.norm3(tgt)
          return tgt

  def _get_activation_fn(activation):
      if activation == "relu":
          return F.relu
      elif activation == "gelu":
          return F.gelu

      raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

  # %% [code]
  class TransformerEncoder(nn.Module):

      def __init__(self, encoder_layer : nn.Module, num_layers : int = 6, norm : nn.Module = None):
          super(TransformerEncoder, self).__init__()
          self.layers = _get_clones(encoder_layer, num_layers)
          self.num_layers = num_layers
          self.norm = norm

      __constants__ = ['norm']

      def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
          output = src

          for mod in self.layers:
              output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

          if self.norm is not None:
              output = self.norm(output)

          return output

  # %% [code]
  class TransformerDecoder(nn.Module):

      def __init__(self, decoder_layer : nn.Module, num_layers : int = 6, norm : nn.Module = None):
          super(TransformerDecoder, self).__init__()
          self.layers = _get_clones(decoder_layer, num_layers)
          self.num_layers = num_layers
          self.norm = norm

      __constants__ = ['norm']

      def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
          output = tgt

          for mod in self.layers:
              output = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

          if self.norm is not None:
              output = self.norm(output)

          return output

  # %% [code]
  class Transformer(nn.Module):

      def __init__(self, MAX_LEN : int = 5000, src_padding_idx : int = None, tgt_padding_idx : int = None, src_vocab_size : int = 5000, tgt_vocab_size : int = 5000, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = "relu") -> None:

          super(Transformer, self).__init__()

          encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
          encoder_norm = nn.LayerNorm(d_model)
          self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

          decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
          decoder_norm = nn.LayerNorm(d_model)
          self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

          self._reset_parameters()

          self.d_model = d_model
          self.nhead = nhead

          self.src_embed = Embedding(src_padding_idx, src_vocab_size, d_model)
          self.tgt_embed = Embedding(tgt_padding_idx, tgt_vocab_size, d_model)

          self.src_pe_ = PositionalEncoding(d_model, dropout, MAX_LEN)
          self.tgt_pe_ = PositionalEncoding(d_model, dropout, MAX_LEN)

          self.linear1 = nn.Linear(d_model, tgt_vocab_size)
          #self.norm_ = nn.LayerNorm(dim_feedforward)
          #self.relu = nn.ReLU()
          #self.linear2 = nn.Linear(dim_feedforward, tgt_vocab_size)
          #self.soft = nn.Softmax(dim = -1)
          #self.tgt_vocab_size = tgt_vocab_size

      def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
          """
          Shape:
              - src: :math:`(S, N, E)`.
              - tgt: :math:`(T, N, E)`.
              - src_mask: :math:`(S, S)`.
              - tgt_mask: :math:`(T, T)`.
              - memory_mask: :math:`(T, S)`.
              - src_key_padding_mask: :math:`(N, S)`.
              - tgt_key_padding_mask: :math:`(N, T)`.
              - memory_key_padding_mask: :math:`(N, S)`.
          """

          src = self.src_pe_(self.src_embed(src))
          tgt = self.tgt_pe_(self.tgt_embed(tgt))

          if memory_key_padding_mask is None:
              memory_key_padding_mask = src_key_padding_mask

          if src.size(1) != tgt.size(1):
              raise RuntimeError("the batch number of src and tgt must be equal")

          if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
              raise RuntimeError("the feature number of src and tgt must be equal to d_model")

          memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
          output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
          return self.linear1(output)

      def _reset_parameters(self):
          for p in self.parameters():
              if p.dim() > 1:
                  xavier_uniform_(p)
```
