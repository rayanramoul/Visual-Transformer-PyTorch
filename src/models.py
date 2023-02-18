from src.utils import transform_image_to_patch, calculate_positional_embeddings

import torch 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiHeadSelfAttention(torch.nn.Module):
    def  __init__(self, d, number_heads=2):
        super(MultiHeadSelfAttention, self).__init__()
        self.d = d
        self.number_heads = number_heads
        
        assert d % number_heads == 0 
        
        d_head = int(d / number_heads)
        
        self.q_mappings = torch.nn.ModuleList([torch.nn.Linear(d_head, d_head) for _ in range(number_heads)])
        self.k_mappings = torch.nn.ModuleList([torch.nn.Linear(d_head, d_head) for _ in range(number_heads)])
        self.v_mappings = torch.nn.ModuleList([torch.nn.Linear(d_head, d_head) for _ in range(number_heads)])
        
        self.d_head = d_head
        self.softmax = torch.nn.Softmax(dim=-1)
        
    def forward(self, sequences):
        # sequences: (n, sequence_length, d)  d  is for token dimension
        # output: (n, sequence_length, d)
        result = [] # Faster than list()
        for sequence in sequences:
            sequence_result = []
            for head in range(self.number_heads):
                q_mappping = self.q_mappings[head]
                k_mappping = self.k_mappings[head]
                v_mappping = self.v_mappings[head]
                
                # q, k, v: (sequence_length, d_head)
                seq = sequence[:, head * self.d_head : (head + 1) * self.d_head]
                q, k, v = q_mappping(seq), k_mappping(seq), v_mappping(seq)
                
                # attention: (sequence_length, sequence_length)
                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                sequence_result.append(attention @ v)
            result.append(torch.hstack(sequence_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])
    
class VitBlock(torch.nn.Module):
    def __init__(self, d, number_heads=2, mlp_ratio=4):
        super(VitBlock, self).__init__()
        self.d = d  
        self.number_heads = number_heads
        
        self.norm1 = torch.nn.LayerNorm(d)
        self.mhsa = MultiHeadSelfAttention(d, number_heads)
        self.norm2 = torch.nn.LayerNorm(d)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d, mlp_ratio * d),
            torch.nn.GELU(),
            torch.nn.Linear(mlp_ratio * d, d)
        )
    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out)) # Residual Connection
        return out

class ViT(torch.nn.Module):
    def __init__(self,  dimensions=(1, 28, 28), number_patches=7, hidden_dimension=8, number_heads=2, number_blocks=2, output_dimension=10) -> None:
        super(ViT,  self).__init__()
        
        self.dimensions  = dimensions # (C, H, W)
        self.number_patches = number_patches
        
        assert dimensions[1] %  number_patches == 0
        assert dimensions[2] %  number_patches == 0
        
        self.patch_size = (dimensions[1] // number_patches, dimensions[2] // number_patches)
        
        # Linear projection
        self.input_dimension = int(dimensions[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = torch.nn.Linear(self.input_dimension, hidden_dimension)
        
        # Learnable classification token
        self.class_token  = torch.nn.Parameter(torch.randn(1, hidden_dimension))
        
        # Positional embeddings
        self.positional_embeddings = torch.nn.Parameter(calculate_positional_embeddings(number_patches ** 2 + 1, hidden_dimension))
        self.positional_embeddings.requires_grad = False
        
        # Add Transformer encoder blocks
        self.blocks = torch.nn.ModuleList([VitBlock(hidden_dimension, number_heads) for _ in range(number_blocks)])
        
        # Classification Head
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dimension, output_dimension),
            torch.nn.Softmax(dim=-1)
        )
        
    def forward(self, images):
        n, _, _, _ = images.shape
        patches = transform_image_to_patch(images, self.number_patches).to(device)
        tokens = self.linear_mapper(patches)
        
        # Add class token
        # Classification token is put as the first token of each sequence
        tokens = torch.stack([torch.vstack([self.class_token, token]) for token in tokens])
        
        # Add positional embeddings
        input_pos_emb = self.positional_embeddings.repeat(n, 1, 1)
        output = tokens + input_pos_emb
        
        # Transformer Encoder block 
        for block in self.blocks:
            output = block(output)
            
        # Pass Through MLP 
        # Get the classification  token only
        output = output[:, 0]    
        return self.mlp(output)
    
