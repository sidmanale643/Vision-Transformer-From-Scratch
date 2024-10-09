import math
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self , in_channels , patch_size , d_model):
        super().__init__()
        
        self.patch = nn.Conv2d(in_channels, d_model , kernel_size = patch_size , stride = patch_size , padding = "valid")        
        self.flatten = nn.Flatten(2)
        
        
    def forward(self , x):
        patches = self.patch(x)
        flattened_patches = self.flatten(patches).transpose(1,2)
        return flattened_patches
    
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
        batch_size, seq_len, d_model = Q.size()    
        
        Q = self.w_q(Q)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(K)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(V)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = torch.softmax(attention_scores, dim = -1)
        attention_values = torch.matmul(attention_weights, V)
        
        attention_values_concat = attention_values.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        attention_out = self.w_o(attention_values_concat)
        return attention_out

                                                
class FFN(nn.Module):
    def __init__(self , d_model , dropout):
        super().__init__()
        
        self.L1 =  nn.Linear(d_model , 4*d_model)
        self.L2 = nn.Linear(4*d_model , d_model)
        self.gelu = nn.GELU()
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
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2= nn.LayerNorm(d_model)
    
    def forward(self , x):
        
        norm_1  = self.norm_1(x)
        mha_out = self.mha(x , x , x ) + norm_1
        norm_2 = self.norm_2(mha_out)
        ffn_out = self.ffn(norm_2) + mha_out
        return ffn_out
    
class VisionTransformer(nn.Module):
    def __init__(self , in_channels , img_size ,  patch_size , d_model , dropout , n_heads , n_classes , n_blocks):
        super().__init__()
        
        self.n_patches = (img_size // patch_size) ** 2
        self.n_classes = n_classes
        self.patch_embeddings = PatchEmbed(in_channels , patch_size , d_model)
        self.pos_em = nn.Embedding(self.n_patches + 1, d_model)
        self.blocks = nn.ModuleList([Block(d_model , dropout , n_heads) for _ in range(n_blocks)])
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))

        self.fc = nn.Linear(d_model , self.n_classes)
        
    def forward(self, x):
        
        patch_embeddings = self.patch_embeddings(x)
        
        batch_size = x.size(0)
        positions = torch.arange(0,  self.n_patches  , device=x.device).unsqueeze(0).expand(batch_size, -1) 
        
        pos_emb = self.pos_em(positions ) 
         
        cls_tokens = self.cls.expand(batch_size, -1, -1)  
        input_ = torch.cat((cls_tokens, patch_embeddings + pos_emb), dim=1)  

        for block in self.blocks:
            input_ = block(input_)
        
        logits = self.fc(input_[: , 0])
        return logits
        
img = torch.randn(1 , 3 , 224 , 224 , dtype = torch.float)

in_channels , img_size , patch_size , d_model , dropout , n_heads  , n_classes ,  n_blocks = 3, 224 , 16 , 768 , 0.1 , 8 , 1000 , 12
       
vit = VisionTransformer(in_channels ,  img_size  , patch_size , d_model , dropout , n_heads , n_classes , n_blocks)
out = vit(img)
print(out)