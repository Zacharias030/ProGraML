"""This is an export from the Runtime_Analysis notebook"""

# In[0]
from pathlib import Path
import sys, os

# make this file executable from anywhere
#if __name__ == '__main__':
full_path = os.path.realpath(__file__)
#full_path = !pwd
#full_path = full_path[0]
print(full_path)
REPO_ROOT = full_path.rsplit('ProGraML', maxsplit=1)[0] + 'ProGraML'
print(REPO_ROOT)
#insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, REPO_ROOT)
REPO_ROOT = Path(REPO_ROOT)


# In[1]
from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric import utils
from torch_geometric.utils import softmax as scatter_softmax
from torch_geometric.data import DataLoader


# # Dense vs Sparse Self-Attention

def print_state_dict(mod):
    for n, t in mod.state_dict().items():
        print(n, t.size())

def num_parameters(mod) -> int:
    """Compute the number of trainable parameters in a nn.Module and its children."""
    num_params = sum(param.numel() for param in mod.parameters(recurse=True) if param.requires_grad)
    return num_params, f"{num_params * 4 / 1e6:.3f}MB"



from deeplearning.ml4pl.models.ggnn.modeling import SelfAttentionMessageLayer
from deeplearning.ml4pl.models.ggnn.configs import BaseConfig



config = BaseConfig.from_dict({
    'edge_type_count': 3,
    'backward_edges': True,
    'hidden_size': 200,
    'transformer_attn_bias': True,
    'transformer_num_heads': 8,
    'transformer_attn_dropout': 0.0,
})


# ## Check equality of sparse implementation with reference

sparse_attn = SelfAttentionMessageLayer(config).to(device='cuda')

#print_state_dict(sparse_attn)
#print("")
#print(sparse_attn)
#print("")
#num_parameters(sparse_attn)
# In[1]

# get dummy data

num_n = 5
heads = 8
hidd = 200

random_emb = nn.Parameter(torch.randn(9000, hidd, device='cuda'), requires_grad=True)

ones = torch.ones(num_n, num_n, device='cuda')

adj = torch.tril(ones, diagonal=0)
#adj = ones
print("adj matrix: edges go from (row -> column) if [row, column] == 1")
print(adj)

attn_mask = torch.tril(ones * float('-inf'), diagonal=-1)

print("")
print("attn_mask.t(): We print the transposed attention mask,")
print("bc. the reference implementation indexes the attn_mask <target, src>")
print(attn_mask.t())

edge_index = utils.dense_to_sparse(adj)[0]
print(edge_index.size())
#print(edge_index.t()[:10])

x = torch.randint(9000, (num_n,))
print(x.size())

node_states = random_emb[x]



# get sparse attn on dummy data
sa_node_states, sa_weights = sparse_attn(edges=edge_index, node_states=node_states)
sa_sum = torch.sum(sa_node_states)

print(sa_sum)


in_proj = sparse_attn.qkv_in_proj.weight
in_proj_bias = sparse_attn.qkv_in_proj.bias
out_proj = sparse_attn.out_proj.weight
out_proj_bias = sparse_attn.out_proj.bias


ns = node_states.unsqueeze(1)
da_node_states, da_weights = F.multi_head_attention_forward(
                ns, ns, ns, 200, 8,
                in_proj, in_proj_bias,
                None, None, False,
                0.0, out_proj, out_proj_bias,
                training=False,
                key_padding_mask=None, need_weights=True, 
                attn_mask=attn_mask)
da_sum = torch.sum(da_node_states)
da_weights = da_weights.squeeze()

print(da_sum)
# %%
print(da_weights.t())


sa_weights_matrix = torch.zeros(num_n, num_n, device='cuda')
for i, (s, t) in enumerate(edge_index.t()):
    sa_weights_matrix[s, t] = sa_weights[i]
print(sa_weights_matrix)
#print(sa_weights)

# In[1]

# assert similarity of attention weights for complete graphs
torch.all(torch.abs(da_weights.squeeze() - sa_weights_matrix ) < 1e-7)

# assert similarity of self-attention output (new node states)
torch.all(torch.abs(da_node_states.squeeze() - sa_node_states) < 1e-7)

print("")

# %%


# %%


# %%
