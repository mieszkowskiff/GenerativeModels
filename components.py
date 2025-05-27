import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class UNet_Tranformer(nn.Module):
    def __init__(
            self, 
            marginal_prob_std, 
            channels = [32, 64, 128, 256], 
            embed_dim = 256,
            text_dim = 256, 
            nClass = 10
        ):
        super().__init__()
        # Embedding layers
        self.time_embed = nn.Sequential(
            TimeProjection(embed_dim = embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        self.cond_embed = nn.Embedding(nClass, text_dim)

        # Other model properties
        self.act = nn.SiLU()
        self.marginal_prob_std = marginal_prob_std
        
        # Encoding layers
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride = 1, bias = False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels = channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride = 2, bias = False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels = channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels = channels[2])
        self.attn3 = SpatialCrossAttention(channels[2], text_dim)

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels = channels[3])
        self.attn4 = SpatialCrossAttention(channels[3], text_dim)

        # Decoding layers
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride = 2, bias = False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels = channels[2])

        self.tconv3 = nn.ConvTranspose2d(channels[2], channels[1], 3, stride = 2, bias = False, output_padding = 1)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels = channels[1])

        self.tconv2 = nn.ConvTranspose2d(channels[1], channels[0], 3, stride = 2, bias = False, output_padding = 1)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels = channels[0])

        self.tconv1 = nn.ConvTranspose2d(channels[0], 1, 3, stride = 1)


    def forward(self, x, t, y = None):
        # Embed time and text
        embed = self.act(self.time_embed(t))
        y_embed = self.cond_embed(y).unsqueeze(1)
        
        # Encoding
        h1 = self.act(self.gnorm1(self.conv1(x) + self.dense1(embed)))
        h2 = self.act(self.gnorm2(self.conv2(h1) + self.dense2(embed)))
        h3 = self.act(self.gnorm3(self.conv3(h2) + self.dense3(embed)))
        h3 = self.attn3(h3, y_embed)
        h4 = self.act(self.gnorm4(self.conv4(h3) + self.dense4(embed)))
        h4 = self.attn4(h4, y_embed)

        # Decoding
        h = self.act(self.tgnorm4(self.tconv4(h4) + self.dense5(embed)))
        h = self.act(self.tgnorm3(self.tconv3(h + h3) + self.dense6(embed)))
        h = self.act(self.tgnorm2(self.tconv2(h + h2) + self.dense7(embed)))
        h = self.tconv1(h + h1)

        # Normalize predicted noise by std at time t
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h
    

class Attention(nn.Module):
    """
    Implements a single-head attention mechanism. This class supports both self-attention
    and cross-attention depending on the context provided.

    Args:
        embed_dim (int): The dimensionality of the embedding space.
        hidden_dim (int): The dimensionality of the hidden states.
        context_dim (int, optional): The dimensionality of the context for cross-attention. 
                                     If None, self-attention is performed.
        num_heads (int, optional): The number of attention heads. Default is 1.
    """
    def __init__(self, embed_dim, hidden_dim, context_dim = None):
        super(Attention, self).__init__()
        self.query = nn.Linear(hidden_dim, embed_dim, bias = False)
        if context_dim is None:
            self.self_attn = True
            self.key = nn.Linear(hidden_dim, embed_dim, bias = False)
            self.value = nn.Linear(hidden_dim, hidden_dim, bias = False)
        else:
            self.self_attn = False
            self.key = nn.Linear(context_dim, embed_dim, bias = False)
            self.value = nn.Linear(context_dim, hidden_dim, bias = False)

    def forward(self, tokens, context = None):
        if self.self_attn:
            Q, K, V = self.query(tokens), self.key(tokens), self.value(tokens)
        else:
            Q, K, V = self.query(tokens), self.key(context), self.value(context)
        
        scoremats = torch.einsum('bth,bsh->bts', Q, K)
        attnmats = F.softmax(scoremats, dim=1)
        ctx_vecs = torch.einsum("bts,bsh->bth", attnmats, V)
        return ctx_vecs

class TransformerBlock(nn.Module):
    """
    Implements a Transformer block that includes self-attention, cross-attention, 
    and a feed-forward network with normalization layers.

    Args:
        hidden_dim (int): The dimensionality of the hidden states.
        context_dim (int): The dimensionality of the context for cross-attention.
    """
    def __init__(self, hidden_dim, context_dim):
        super(TransformerBlock, self).__init__()
        self.attn_self = Attention(hidden_dim, hidden_dim)
        self.attn_cross = Attention(hidden_dim, hidden_dim, context_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU()
        )

    def forward(self, x, context = None):
        x = self.attn_self(self.norm1(x)) + x
        x = self.attn_cross(self.norm2(x), context = context) + x
        x = self.ffn(self.norm3(x)) + x
        return x

class SpatialCrossAttention(nn.Module):
    """
    Implements a Spatial Cross Attention that applies a Transformer block to spatial data, 
    typically images. This allows spatial interactions within the Transformer architecture.

    Args:
        hidden_dim (int): The dimensionality of the hidden states.
        context_dim (int): The dimensionality of the context for cross-attention.
    """
    def __init__(self, hidden_dim, context_dim):
        super(SpatialCrossAttention, self).__init__()
        self.transformer = TransformerBlock(hidden_dim, context_dim)

    def forward(self, x, context = None):
        b, c, h, w = x.shape
        x_in = x
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.transformer(x, context)
        x = rearrange(x, 'b (h w) c -> b c h w', h = h, w = w)
        return x + x_in

class TimeProjection(nn.Module):
    def __init__(self, embed_dim, time_steps = 1000):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Embedding(time_steps, embed_dim)
    
    def forward(self, t):
        return self.proj(t)

class Dense(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.dense(x)[..., None, None]


