import torch
import torch.nn as nn
import torch.nn.Functional as F
from torch.nn import LayerNorm

import math

""" 
BERT Model -> Encoder based Transformer Architectures

"""

class Attention(nn.Module):
    def __init__(self):
        pass
    # query, key, value: (B, h, seq_len, d_k) eg.(B, 12, seq_len, 64)
    def forward(self, q, k, v, mask=None, dropout=None):
        # torch.transpose(input, dim0, dim1): The given dimensions dim0 and dim1 are swapped.
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        # scores: (B, h, seq_len, seq_len)
        if mask is not None:
            # masked_fill(mask, value): Fills elements of self tensor with value where mask is True. 
            scores = scores.masked_fill(mask==0, -1e9)
        # p_attn: (B, h, seq_len, seq_len) 
        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)
        # torch.matmul(p_attn, value): (B, h, seq_len, d_k), p_attn: (B, h, seq_len, seq_len)
        return torch.matmul(p_attn, v), p_attn
    
class MultiHeadAttension(nn.Module):
    """
    
    """
    def __init__(self, h, model_d, dropout=0.2):
        super().__init__()
        assert model_d % h == 0

        # assume d_v always equals to d_k
        # eg. d_model = 768, h = 12, d_k = 64
        self.d_k = model_d // h
        self.h = h

        self.linear_layers = nn.ModuleList([
            # 768->768
            nn.Linear(model_d, model_d) for _ in range(3)
        ])
        self.output_layer = nn.Linear(model_d, model_d)

        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    # input query, key, value will be x with (B, seq_len, d_model).
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        # 1. Do all linear projections in batch from model_d -> h x d_k
        # eg.  768 -> (12, 64)
        # l(x): (B, seq_len, d_model) -> (B, seq_len, h, d_k) -> (B, h, seq_len, d_k) 
        # output: (B, h, seq_len, d_k)  eg.(B, 12, seq_len, 64)
        q, k, v = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linear_layers, (q, k, v))]

        # 2). Apply attention on all the projected vector in batch 
        # x:(B, h, N, d_k) eg.(B, 12, seq_len, 64), attn:(B, h, seq_len, seq_len) eg.(B, 12, seq_len, seq_len)
        x, attn = self.attention(q, k, v, mask=mask, dropout=self.dropout)

        # 3). Concatenate  using a view and apply a final linear layer
        # x:(B, h, seq_len, d_k) -> (B, seq_len, h, d_k)  -> (B, seq_len, h * d_k)  eg.(B, seq_len, 768)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        # (B ,seq_len, d_model) eg.(B, seq_len, 768) 
        return self.output_layer(x)
    
class SubLayerConnection(nn.Module):
    """ 
    
    """
    def __init__(self, size, dropout):
        super(SubLayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Droput(dropout)

    # the input 'sublayer' will be Multihead attention or FF
    def forward(self, x, sublayer):
        """
        Applying residual connection to any sublayer with same size
        """
        return x + self.dropout(sublayer(self.norm(x)))

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, model_d, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # d_ff: d_model x 4,  eg. 768 x 4 = 3072
        self.w_1 = nn.Linear(model_d, d_ff)                                     
        self.w_2 = nn.Linear(d_ff, model_d)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    # x: (B, seq_len, d_model)
    def forward(self, x):
        # (B, seq_len, d_model)
        return self.w_2(self.dropout(self.activation(self.w_1(x))))           # (B, seq_len, d_model)


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection 
    
    """
    def __init__(self, hidden, attn_heads, ff_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadAttension(h=attn_heads, model_d=hidden)
        self.feed_forward = PositionwiseFeedForward(model_d = hidden, d_ff = ff_hidden, dropout=dropout)
        self.input_sublayer = SubLayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SubLayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
    # x: (B, seq_len, hidden)   eg.(B, seq_len, 768)
    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x : self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        # (B, seq_len, hidden)
        return self.dropout(x)

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)

class PositionalEmbedding(nn.Module):
    def __init__(self, model_d, max_len=512):
        super().__init__()

        # compute positional encoding once in log space
        pe = torch.zeros(max_len, model_d).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqeeze(1)
        # (d_model//2,) eg. (384,)
        div_term = (torch.arange(0, model_d, 2).float() * -(math.log(10000.0) / model_d)).exp()   
        a = position * div_term
        
        pe[:, 0::2] = torch.sin(a)
        pe[:, 1::2] = torch.cos(a)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size = 512):
        super().__init__(3, embed_size, padding_idx=0)

class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """
    def __init__(self, vocab_size, embed_size, dropout=0.1):
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label):    
        # sequence: (B, seq_len), segment_label: (B, seq_len)
        # make BERT Embedding using token embedding, 
        # position embedding, and segment embedding                                          
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        # (B, seq_len, hidden), (1, seq_len, hidden), 
        # (B, seq_len, hidden), respectively. 
        # Broadcasting;  
        return self.dropout(x)
    
class BERT(nn.Module):
    """ 
    BERT model : Bidirectional Encoder Representations from Transformers

    """
    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
    # x: (B, seq_len),      segment_info: (B, seq_len)
    def forward(self, x, segment_info):                                         
        # attention masking for padded token
        # attention mask size: torch.ByteTensor([batch_size, 1, seq_len, seq_len])  
        # Why? Because of 'scores' shape in Attention class: (B, h, seq_len, seq_len)
        # (B, seq_len) -> (B, 1, seq_len) -> (B, seq_len, seq_len) -> (B, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)       
        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

##########################################################
"""
BERT Language Model : Next Sentence Prediction Model + Masked Language Model

"""
###########################################################
class NextSentencePrediction(nn.Module):
    """
    2 class classification model -> is_next, is_not_next

    """
    def __init__(self, hidden):
        """
        @param hidden : BERT model output size
        """
        super().__init__()
        # binary classification
        self.linear = nn.Linear(hidden, 2)
        # along with last dim
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        # 0 for the first token <CLS> for the NSP.
        return self.softmax(self.lienar(x[:, 0]))

class MaskedLanguageModel(nn.Module):
    """
    
    """
    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))  

class BERTLM(nn.Module):
    """ 
    
    """
    def __init__(self, bert: BERT, vocab_size):
        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.next_sentence(x), self.mask_lm(x)