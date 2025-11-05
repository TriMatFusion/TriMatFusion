import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d
import torch_geometric
from torch_geometric.nn import Set2Set, CGConv
from torch import nn
from transformers import AutoModel



class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=5, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** 0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        B = x.size(0)
        Q = self.q(x).view(B, self.num_heads, self.head_dim)
        K = self.k(y).view(B, self.num_heads, self.head_dim)
        V = self.v(y).view(B, self.num_heads, self.head_dim)

        attn = (Q * K).sum(-1) / self.scale             
        attn = F.softmax(attn, dim=-1).unsqueeze(-1)     
        out  = (attn * V).reshape(B, -1)                 
        return self.dropout(self.proj(out))               


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super(GatedFusion, self).__init__()
        self.gate = nn.Linear(dim * 2, dim)
        
    def forward(self, A, B):
        g = torch.sigmoid(self.gate(torch.cat((A, B), dim=1))) 
        return g * A + (1 - g) * B 

class ModalFusion(torch.nn.Module):
    def __init__(
        self,
        data,
        dim1=64,
        dim2=64,
        pre_fc_count=1,    
        gc_count=3,   
        post_fc_count=1,
        pool="global_mean_pool",
        batch_norm="True",
        batch_track_stats="True",
        act="relu",
        dropout_rate=0.0,
        **kwargs
    ):
        super(ModalFusion, self).__init__()
        
        self.batch_track_stats = True 
        self.batch_norm = batch_norm
        self.pool = pool
        self.act = act
        self.dropout_rate = dropout_rate
        
        self.bert = AutoModel.from_pretrained('./matscibert')   
        for param in self.bert.parameters():
            param.requires_grad = False  
        
        self.text_proj = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, dim2)
        )
        

        self.fingerprint_proj = nn.Sequential(
                                nn.Linear(8, dim2),
                                nn.ReLU()
                                )
        

        # Multi-head cross-attention and gated fusion
        self.attn_graph_text       = MultiHeadCrossAttention(dim2, num_heads=5)
        self.attn_graph_fingerprint = MultiHeadCrossAttention(dim2, num_heads=5)
        self.gate_graph_text = GatedFusion(dim2)
        self.gate_graph_fingerprint = GatedFusion(dim2)
        
        assert gc_count > 0, "Need at least 1 GC layer"        
        gc_dim = dim1
        post_fc_dim = dim1
        output_dim = 1

        # Pre-GNN MLP
        self.pre_lin_list = torch.nn.ModuleList()
        for i in range(pre_fc_count):
            if i == 0:
                lin = torch.nn.Linear(data.num_features, dim1)
                self.pre_lin_list.append(lin)
            else:
                lin = torch.nn.Linear(dim1, dim1)
                self.pre_lin_list.append(lin)

        # GNN layers
        self.conv_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(gc_count):
            conv = CGConv(
                gc_dim, data.num_edge_features, aggr="mean", batch_norm=False
            )
            self.conv_list.append(conv)
            bn = BatchNorm1d(gc_dim, track_running_stats=self.batch_track_stats)
            self.bn_list.append(bn)

        # Post-GNN MLP
        self.post_lin_list = torch.nn.ModuleList()
        for i in range(post_fc_count): 
            if i == 0:
                ##Set2set pooling has doubled dimension
                if self.pool == "set2set":
                    lin = torch.nn.Linear(post_fc_dim * 2, dim2)
                else:
                    lin = torch.nn.Linear(post_fc_dim, dim2)   
                self.post_lin_list.append(lin)
            else:
                lin = torch.nn.Linear(dim2, dim2)
                self.post_lin_list.append(lin)
        self.lin_out = torch.nn.Linear(dim2, output_dim)
    
        ##Set up set2set pooling (if used)
        if self.pool == "set2set":
            self.set2set = Set2Set(post_fc_dim, processing_steps=3)


    def forward(self, data):   
        # Pre-GNN
        for i in range(0, len(self.pre_lin_list)):   
            if i == 0:
                out = self.pre_lin_list[i](data.x)   
                out = getattr(F, self.act)(out)  
            else:
                out = self.pre_lin_list[i](out)
                out = getattr(F, self.act)(out)
                
        # GNN layers
        for i in range(0, len(self.conv_list)):  
            if len(self.pre_lin_list) == 0 and i == 0:
                out = self.conv_list[i](data.x, data.edge_index, data.edge_attr)
                out = self.bn_list[i](out)
            else:    
                out = self.conv_list[i](out, data.edge_index, data.edge_attr)  
                out = self.bn_list[i](out)         
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
            
        # Graph pooling
        if self.pool == "set2set":
            out = self.set2set(out, data.batch)
        else:
            out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
            
        # Post-GNN MLP    
        for i in range(0, len(self.post_lin_list)):  
            out = self.post_lin_list[i](out)
            out = getattr(F, self.act)(out)

        text_outputs = self.bert(input_ids=data.text_input_ids, attention_mask=data.text_attention_mask)
        text_pooled_output = text_outputs.last_hidden_state[:, 0, :]  
        text_emb = self.text_proj(text_pooled_output)   


        fp = data.fingerprint.view(text_emb.shape[0], 6, 8).mean(1)  
        finger_emb = self.fingerprint_proj(fp)  


        # Multi-head cross-attention
        graph_text_attn = self.attn_graph_text(out, text_emb)
        graph_fingerprint_attn = self.attn_graph_fingerprint(out, finger_emb)            

        # Gated fusion 
        fused_graph_text = self.gate_graph_text(out, graph_text_attn)  
        fused_graph_fingerprint = self.gate_graph_fingerprint(out, graph_fingerprint_attn)  

        # Final representation
        final_fusion = fused_graph_text + fused_graph_fingerprint  # Element‑wise sum
        fussed = self.lin_out(final_fusion)   
                
        if fussed.shape[1] == 1:
            return fussed.view(-1)
        else:
            return fussed
