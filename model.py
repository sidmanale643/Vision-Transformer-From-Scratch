import math
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self , in_channels , patch_size , d_model):
        super().__init__()
        
        self.patch = nn.Conv2d(in_channels, d_model , kernel_size = patch_size , stride = patch_size , padding = "valid")        
        self.flatten = nn.Flatten(2)
        self.project = nn.Linear(d_model , d_model , bias = False)
        
    def forward(self , x):
        patches = self.patch(x)
        flattened_patches = self.flatten(patches).transpose(1,2)
        patch_embeddings = self.project(flattened_patches)
        return patch_embeddings
    
class MHA(nn.Module):
    def __init__(self , d_model , n_heads):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model , d_model)
        self.w_k = nn.Linear(d_model , d_model)
        self.w_v = nn.Linear(d_model , d_model)
        self.w_o = nn.Linear(d_model , d_model)
              
    def forward(self , Q , K , V ):
        
        batch_size , seq_len , d_model = Q.size      
        
        Q = self.w_q(Q)
        Q = Q.view(batch_size , seq_len , self.n_heads , self.n_heads * self.d_k).transpose(1,2)
        K = self.w_k(K)
        K = K.view(batch_size , seq_len , self.n_heads , self.n_heads * self.d_k).transpose(1,2)
        V = self.w_v(V)
        V = V.view(batch_size , seq_len , self.n_heads , self.n_heads * self.d_k).transpose(1,2)
        
        attention_scores = torch.matmul(Q , K.transpose(1,2)) / math.sqrt(self.d_k)
        attention_weights = torch.softmax(attention_scores , dim = -1)
        attention_values =  torch.matmul(attention_weights , V)
        
        attention_values_concat = attention_values.transpose(1, 2).contiguous().view(batch_size, seq_len , d_model ) 
        
        attention_out = self.w_o(attention_values_concat)
        return attention_out
                                                
class FFN(nn.Module):
    def __init__(self , d_model , dropout):
        super().__init__()
        
        self.L1 =  nn.Linear(d_model , 4*d_model)
        self.L2 = nn.Linear(4*d_model , d_model)
        self.gelu = nn.GELU(inplace = True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self , x):
        x = self.L1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        out = self.L2(x)
        return out

class Block(nn.Module):
    def __init__(self , d_model , dropout , n_heads):
        super().__init__()
        
        self.mha = MHA(d_model , n_heads)
        self.ffn = FFN( d_model , dropout)
        self.norm_1 = nn.LayerNorm()
        self.norm_2= nn.LayerNorm()
    
    def forward(self , x):
        
        mha_out = self.mha(x)
        norm_1 = self.norm_1(x + mha_out)
        ffn_out = self.ffn(norm_1)
        norm_2 = self.norm_2(norm_1 + ffn_out)
        return norm_2
    
class VisionTransformer(nn.Module):
    def __init__(self , in_channels , n_patches , patch_size , d_model , dropout , n_blocks):
        super().__init__()

        self.patch_embeddings = PatchEmbed(in_channels , patch_size , d_model)
        self.pos_em = nn.Embedding(n_patches , d_model)
        self.blocks = nn.ModuleList([Block(d_model , dropout) for _ in range(n_blocks)])
    
    def forward(self , x):
        patch_embeddings = self.patch_embeddings(x)
        pos_emb = self.pos_em(x)
        input = pos_emb + patch_embeddings
        for block in self.blocks:
            input = block(input)
        
        return input
        
        
x = torch.rand(1 , 3 , 224 , 224)
pt = PatchEmbed(3 , (16,16) , 768)
i = pt(x)
print(i.shape) 