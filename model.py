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

class VisionTransformer(nn.Module):
    def __init__(self ):
        super().__init__()
        
        # self.d_model = d_model
        # self.patch_size = patch_size
        pass
    
x = torch.rand(1 , 3 , 224 , 224)
pt = PatchEmbed(3 , (16,16) , 768)
i = pt(x)
print(i.shape)