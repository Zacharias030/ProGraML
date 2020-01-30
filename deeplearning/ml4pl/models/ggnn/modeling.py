# Copyright 2019 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Modules that make up the pytorch GNN models."""
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim

# Dependency moved into SelfAttention Message Layer
# from torch_geometric.utils import softmax as scatter_softmax


SMALL_NUMBER = 1e-8

def print_state_dict(mod):
    for n, t in mod.state_dict().items():
        print(n, t.size())

def num_parameters(mod) -> int:
    """Compute the number of trainable parameters in a nn.Module and its children.
    OBS:
        This function misses some parameters, i.e. in pytorch's official MultiheadAttention layer,
        while the state dict doesn't miss any!
    """
    num_params = sum(param.numel() for param in mod.parameters(recurse=True) if param.requires_grad)
    return num_params, f"{num_params * 4 / 1e6:.3f}MB"

def assert_no_nan(tensor_list):
    for i, t in enumerate(tensor_list):
        assert not torch.isnan(t).any(), f"{i}: {tensor_list}"


################################################
# Main Model classes
################################################
class BaseGNNModel(nn.Module):
    def __init__(self):
        super().__init__()

    def setup(self, test_only):
        # move model to device before making optimizer!
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.to(self.dev)
        print(f"Moved model to {self.dev}")

        if test_only:
            self.opt = None
            self.eval()
        else:
            self.opt = self.get_optimizer(self.config)

    def get_optimizer(self, config):
        return optim.AdamW(self.parameters(), lr=config.lr)

    def forward(self, vocab_ids, labels, edge_lists, pos_lists=None, node_types=None, *igd, **kwigd):
        raw_in = self.node_embeddings(vocab_ids)

        raw_out, raw_in, *unroll_stats = self.gnn(
            edge_lists, raw_in, pos_lists, node_types
        )  # OBS! self.gnn might change raw_in inplace, so use the two outputs
        # instead!

        nodewise_readout, _ = self.readout(raw_in, raw_out)
        logits = nodewise_readout


        # accuracy, pred_targets, correct, targets
        metrics_tuple = self.metrics(logits, labels)


        outputs = (
            (logits,) + metrics_tuple + (None,) + tuple(unroll_stats)
        )

        return outputs

    def num_parameters(self) -> int:
        """Compute the number of trainable parameters in this nn.Module and its children."""
        return sum(param.numel() for param in self.parameters(recurse=True) if param.requires_grad)


class GraphTransformerModel(BaseGNNModel):
    """Transformer Encoder for Graphs."""
    def __init__(self, config, pretrained_embeddings=None, test_only=False):
        super().__init__()
        self.config = config
        self.node_embeddings = NodeEmbeddings(config)
        self.gnn = TransformerEncoder(config)
        self.readout = Readout(config)

        self.loss = Loss(config)
        self.metrics = Metrics()

        self.setup(test_only)


class GGNNForPretrainingModel(BaseGNNModel):
    """Gated Graph Neural Network model for pretraining"""
    def __init__(self, config, pretrained_embeddings=None, test_only=False):
        super().__init__()
        self.config = config
        self.node_embeddings = NodeEmbeddingsForPretraining(config)
        self.gnn = GGNNProper(config)
        self.readout = Readout(config)

        self.loss = MLMLoss(config)
        self.metrics = Metrics()

        self.setup(test_only)


class GGNNModel(BaseGNNModel):
    def __init__(self, config, pretrained_embeddings=None, test_only=False):
        super().__init__()
        self.config = config

        # input layer
        if getattr(config, 'use_selector_embeddings', False):
            self.node_embeddings = NodeEmbeddingsWithSelectors(config, pretrained_embeddings)
        else:
            self.node_embeddings = NodeEmbeddings(config, pretrained_embeddings)


        # Readout layer
        self.readout = Readout(config)

        # GNN
        # make readout available to label_convergence tests in GGNN Proper (at runtime)
        self.gnn = GGNNProper(config, readout=self.readout)


        # maybe tack on the aux readout
        self.has_aux_input = getattr(self.config, "has_aux_input", False)
        if self.has_aux_input:
            self.aux_readout = AuxiliaryReadout(config)

        # eval and training
        self.loss = Loss(config)
        self.metrics = Metrics()

        self.setup(test_only)

    def forward(
        self,
        vocab_ids,
        labels,
        edge_lists,
        selector_ids=None,
        pos_lists=None,
        num_graphs=None,
        graph_nodes_list=None,
        node_types=None,
        aux_in=None,
        test_time_steps=None,
    ):
        raw_in = self.node_embeddings(vocab_ids, selector_ids)
        raw_out, raw_in, *unroll_stats = self.gnn(
            edge_lists, raw_in, pos_lists, node_types, test_time_steps
        )  # OBS! self.gnn might change raw_in inplace, so use the two outputs
        # instead!

        if self.config.has_graph_labels:
            assert graph_nodes_list is not None and num_graphs is not None, 'has_graph_labels requires graph_nodes_list and num_graphs tensors.'
        nodewise_readout, graphwise_readout = self.readout(
            raw_in,
            raw_out,
            graph_nodes_list=graph_nodes_list,
            num_graphs=num_graphs
        )

        logits = graphwise_readout if self.config.has_graph_labels else nodewise_readout
        if self.has_aux_input:
            logits, graphwise_readout = self.aux_readout(logits, aux_in)

        # accuracy, pred_targets, correct, targets
        metrics_tuple = self.metrics(logits, labels)

        outputs = (
            (logits,) + metrics_tuple + (graphwise_readout,) + tuple(unroll_stats)
        )

        return outputs


################################################
# GNN Proper: Message+Aggregate, Update
################################################

# GNN proper, i.e. everything between input and readout.
# Will rely on the different msg+aggr and update modules to build up a GNN.

class GGNNProper(nn.Module):
    def __init__(self, config, readout=None):
        super().__init__()
        self.backward_edges = config.backward_edges
        self.layer_timesteps = config.layer_timesteps
        self.position_embeddings = config.position_embeddings

        # optional eval time unrolling parameter
        self.test_layer_timesteps = getattr(config, 'test_layer_timesteps', 0)
        self.unroll_strategy = getattr(config, 'unroll_strategy', 'none')
        self.max_timesteps = getattr(config, 'max_timesteps', 1000)
        self.label_conv_threshold = getattr(config, 'label_conv_threshold', 0.995)
        self.label_conv_stable_steps = getattr(config, 'label_conv_stable_steps', 1)

        # make readout avalable for label_convergence tests
        if self.unroll_strategy == "label_convergence":
            assert not self.config.has_aux_input, "aux_input is not supported with label_convergence"
            assert readout, "Gotta pass instantiated readout module for label_convergence tests!"
            self.readout = readout

        # Message and update layers
        self.message = nn.ModuleList()
        for i in range(len(self.layer_timesteps)):
            self.message.append(GGNNMessageLayer(config))

        self.update = nn.ModuleList()
        for i in range(len(self.layer_timesteps)):
            self.update.append(GGNNUpdateLayer(config))

    def forward(self, edge_lists, node_states, pos_lists=None, node_types=None, test_time_steps=None):
        old_node_states = node_states.clone()

        if self.backward_edges:
            back_edge_lists = [x.flip([1]) for x in edge_lists]
            edge_lists.extend(back_edge_lists)

            # For backward edges we keep the positions of the forward edge!
            if self.position_embeddings:
                pos_lists.extend(pos_lists)

        # we allow for some fancy unrolling strategies.
        # Currently only at eval time, but there is really no good reason for this.
        if self.training or self.unroll_strategy == "none":
            layer_timesteps = self.layer_timesteps
        elif self.unroll_strategy == "constant":
            layer_timesteps = self.test_layer_timesteps
        elif self.unroll_strategy == "edge_count":
            assert (
                test_time_steps is not None
            ), f"You need to pass test_time_steps or not use unroll_strategy '{self.unroll_strategy}''"
            layer_timesteps = [min(test_time_steps, self.max_timesteps)]
        elif self.unroll_strategy == "data_flow_max_steps":
            assert (
                test_time_steps is not None
            ), f"You need to pass test_time_steps or not use unroll_strategy '{self.unroll_strategy}''"
            layer_timesteps = [min(test_time_steps, self.max_timesteps)]
        elif self.unroll_strategy == "label_convergence":
            node_states, unroll_steps, converged = self.label_convergence_forward(
                edge_lists, node_states, pos_lists, node_types, initial_node_states=old_node_states
            )
            return node_states, old_node_states, unroll_steps, converged
        else:
            raise TypeError(
                "Unreachable! "
                f"Unroll strategy: {self.unroll_strategy}, training: {self.training}"
            )

        for (layer_idx, num_timesteps) in enumerate(layer_timesteps):
            for t in range(num_timesteps):
                messages = self.message[layer_idx](edge_lists, node_states, pos_lists)
                node_states = self.update[layer_idx](messages, node_states, node_types)
        return node_states, old_node_states

    def label_convergence_forward(
        self, edge_lists, node_states, pos_lists, node_types, initial_node_states
    ):
        assert (
            len(self.layer_timesteps) == 1
        ), f"Label convergence only supports one-layer GGNNs, but {len(self.layer_timesteps)} are configured in layer_timesteps: {self.layer_timesteps}"

        stable_steps, i = 0, 0
        old_tentative_labels = self.tentative_labels(
            initial_node_states, node_states
        )

        while True:
            messages = self.message[0](edge_lists, node_states, pos_lists)
            node_states = self.update[0](messages, node_states, node_types)
            new_tentative_labels = self.tentative_labels(
                initial_node_states, node_states
            )
            i += 1

            # return the new node states if their predictions match the old node states' predictions.
            # It doesn't matter during testing since the predictions are the same anyway.
            stability = (
                (new_tentative_labels == old_tentative_labels)
                .to(dtype=torch.get_default_dtype())
                .mean()
            )
            if stability >= self.label_conv_threshold:
                stable_steps += 1

            if stable_steps >= self.label_conv_stable_steps:
                return node_states, i, True

            if i >= self.max_timesteps:  # maybe escape
                return node_states, i, False

            old_tentative_labels = new_tentative_labels

        raise ValueError("Serious Design Error: Unreachable code!")

    def tentative_labels(self, initial_node_states, node_states):
        logits, _ = self.readout(initial_node_states, node_states)
        preds = F.softmax(logits, dim=1)
        predicted_labels = torch.argmax(preds, dim=1)
        return predicted_labels

class TransformerEncoder(nn.Module):
    def __init__(self, config, readout=None):
        super().__init__()
        self.backward_edges = config.backward_edges
        self.layer_timesteps = config.layer_timesteps
        self.position_embeddings = getattr(config, 'position_embeddings', False)
        self.use_node_types = getattr(config, 'use_node_types', False)
        assert not self.position_embeddings, "not implemented"
        assert not self.use_node_types, "not implemented"

        # Message and update layers
        self.message = nn.ModuleList()
        for i in range(len(self.layer_timesteps)):
            self.message.append(SelfAttentionMessageLayer(config))

        self.update = nn.ModuleList()
        for i in range(len(self.layer_timesteps)):
            self.update.append(TransformerUpdateLayer(config))

    def forward(self, edge_lists, node_states, pos_lists=None, node_types=None, test_time_steps=None):
        old_node_states = node_states.clone()

        if self.backward_edges:
            back_edge_lists = [x.flip([1]) for x in edge_lists]
            edge_lists.extend(back_edge_lists)

            # For backward edges we keep the positions of the forward edge!
            #if getattr(self, 'position_embeddings', False):
            #    pos_lists.extend(pos_lists)

        for (layer_idx, num_timesteps) in enumerate(self.layer_timesteps):
            for t in range(num_timesteps):
                messages = self.message[layer_idx](edge_lists, node_states, pos_lists)
                node_states = self.update[layer_idx](messages, node_states, node_types)
        return node_states, old_node_states


###### Message Layers

class SelfAttentionMessageLayer(nn.Module):
    """Implements ransformer scaled dot-product self-attention,
    in a sparse setting on a graph. This reduces the time and space complexity
    from O(N^2 * D) to O(M * D), which is much better if the graph has an average degree
    that is O(1), i.e. M \in O(n) instead of O(n^2)!

    The layer supports optionally embedding edge-position information.

    Args:
        edge_lists (for each edge type)
        node_states <N, D+S>
        pos_lists <M> (optionally)
    Returns:
        incoming messages per node of shape <N, D+S>
    """
    # TODO(Zach) adapt for position information
    # --> add positions to node_states after index_selecting them into q',k',v' shape
    # --> think about where the position should be added (cf. transformer xl derivation / xlnet)
    # TODO(Adapt for edge_type information)
    # --> edge types should get their own k, v projections, but not their own queries.

    def __init__(self, config):
        super().__init__()
        from torch_geometric.utils import softmax as scatter_softmax
        self.edge_type_count = config.edge_type_count * 2 if config.backward_edges else config.edge_type_count
        self.embed_dim = config.hidden_size

        self.bias = config.attn_bias
        self.num_heads = config.attn_num_heads
        self.dropout_p = config.attn_dropout
        
        head_dim = self.embed_dim // self.num_heads
        assert head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # projection from input to q, k, v
        # Myle Ott et al. apparently observed that initializing the qkv_projection (in one matrix)
        #   with
        #        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        #   to be much better than only xavier.
        self.qkv_in_proj = LinearNet(self.embed_dim, self.embed_dim * 3, bias=self.bias, gain=1 / math.sqrt(2))
        self.out_proj = LinearNet(self.embed_dim, self.embed_dim, bias=self.bias)

        self.dropout = nn.Dropout(p=self.dropout_p, inplace=True)


    def forward(self, edge_lists=None, node_states=None, pos_lists=None, edges=None, need_weights=False):
        """edge_lists: [<M_i, 2>, ...]"""

        # Glue Code:
        assert node_states is not None

        # since we don't support edge-types yet, we just concatenate them here.
        if edge_lists is not None:
            assert edges is None
            edges = torch.cat(edge_lists, dim=0)
        else:
            assert edges is not None

        edge_sources = edges[0, :]
        edge_targets = edges[1, :]


        # ~~~ Sparse Self-Attention ~~~
        # The implementation follows the official pytorch implementation, but sparse.
        # Legend:
        #   Model hidden size D,
        #   number of attention heads h,
        #   number of edges M,
        #   number of nodes N
        num_nodes, embed_dim = node_states.size()
        assert embed_dim == self.embed_dim

        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # 1) get Q, K, V from node_states
        #   (needs to be merged with step 2 if we want to use positions..., bc
        #   they need to be added before retrieving Q, K, V)

        q, k, v = self.qkv_in_proj(node_states).chunk(3, dim=1)


        # 2) get Q', K', V' \in <M, D> by doing an F.emb lookup on Q, K, V (maybe transposed)
        #   according to index
        #       edge_target for Q, and
        #       edge_sources for K, V
        #   since the receiver of messages is querying her neighbors.
        q_prime = torch.index_select(q, dim=0, index=edge_targets)
        k_prime = torch.index_select(k, dim=0, index=edge_sources)
        v_prime = torch.index_select(v, dim=0, index=edge_sources)

        # 3) Q' * K' (hadamard) and sum over D dimension,
        #   then scaled by sqrt(D)
        # 3*) multi-head: If we want h multiple heads, then we should only sum the h segments of size D//h here.
        #   We will end up with <M, h> unnormalized attention scores.
        scores_prime = q_prime * k_prime
        # sum segments of head_dim size into num_head chunks
        scores = scores_prime.transpose(0,1).view(self.num_heads, head_dim, -1).sum(dim=1).t().contiguous()
        scaling = float(head_dim) ** -0.5
        scores = scores * scaling
        assert scores.size() == (q_prime.size()[0], self.num_heads) # <M, num_heads>

        # 4) Scattered Softmax:
        #   Perform a softmax by normalizing scores with the sum of those scores
        #   where edge_targets coincide (meaning incoming edges to the same target are normalized)
        #   we end up with <M> normalized self-attention scores
        # 4*) multi-head: here we run the scattered_softmax in parallel over the h dimensions independently.

        # <M, num_heads>
        attn_output_weights = scatter_softmax(scores, index=edge_targets, num_nodes=node_states.size()[0])  # noqa: F821
        attn_output_weights = self.dropout(attn_output_weights)

        # 5) V' * %4: weight values V' <M> by attention.
        # The result up to here are the messages traveling across edges.
        # 5* a) multi-head: get a view of V' with dim D_v split into <D//h, h>
        #   then get back the old view
        v_prime = v_prime.transpose(0, 1)
        v_prime = v_prime.view(self.num_heads, head_dim, -1)
        v_prime = v_prime.permute(2,0,1) # v_prime now: <M, num_heads, head_dim>

        attn_out_per_edge = v_prime * attn_output_weights.unsqueeze(2)
        attn_out_per_edge = attn_out_per_edge.view(-1, embed_dim)

        # 6) Scatter Add: aggregate messages via index_add with index edge_target
        # to end up with
        #       messages <N, D>
        attn_out = torch.zeros_like(node_states)
        attn_out.index_add_(0, edge_targets, attn_out_per_edge)

        # 5* b) Additionally project from the concatenation back to D. cf. vaswani et al. 2017
        attn_out = self.out_proj(attn_out)

        # now we have messages_by_targets! finally...

        if need_weights:
            # average attention weights over heads (sorted like the edges)
            attn_output_weights = attn_output_weights.sum(dim=1) / self.num_heads
            return attn_out, attn_output_weights
        return attn_out


class GGNNMessageLayer(nn.Module):
    """Implements the MLP message function of the GGNN architecture,
    optionally with position information embedded on edges.
    Args:
        edge_lists (for each edge type)
        node_states <N, D+S>
        pos_lists <M> (optionally)
    Returns:
        incoming messages per node of shape <N, D+S>"""

    def __init__(self, config):
        super().__init__()
        self.forward_and_backward_edge_type_count = (
            config.edge_type_count * 2
            if config.backward_edges
            else config.edge_type_count
        )
        self.msg_mean_aggregation = config.msg_mean_aggregation
        self.dim = config.hidden_size

        self.transform = LinearNet(
            self.dim,
            self.dim * self.forward_and_backward_edge_type_count,
            bias=config.use_edge_bias,
            dropout=config.edge_weight_dropout,
        )

        self.pos_transform = None
        if config.position_embeddings:
            self.register_buffer(
                "position_embs",
                PositionEmbeddings()(
                    torch.arange(512, dtype=torch.get_default_dtype()),
                    config.emb_size,
                    dpad=getattr(config, 'selector_size', 0),
                ),
            )
            self.pos_transform = LinearNet(
                self.dim,
                self.dim,
                bias=config.use_edge_bias,
                dropout=config.edge_weight_dropout,
            )

    def forward(self, edge_lists, node_states, pos_lists=None):
        """edge_lists: [<M_i, 2>, ...]"""

        if self.pos_transform:
            pos_gating = 2 * torch.sigmoid(self.pos_transform(self.position_embs))

        # all edge types are handled in one matrix, but we
        # let propagated_states[i] be equal to the case with only edge_type i
        propagated_states = (
            self.transform(node_states)
            .transpose(0, 1)
            .view(self.forward_and_backward_edge_type_count, self.dim, -1)
        )

        messages_by_targets = torch.zeros_like(node_states)
        if self.msg_mean_aggregation:
            device = node_states.device
            bincount = torch.zeros(
                node_states.size()[0], dtype=torch.long, device=device
            )

        for i, edge_list in enumerate(edge_lists):
            edge_targets = edge_list[:, 1]
            edge_sources = edge_list[:, 0]

            messages_by_source = F.embedding(
                edge_sources, propagated_states[i].transpose(0, 1)
            )

            if self.pos_transform:
                pos_list = pos_lists[i]
                pos_by_source = F.embedding(pos_list, pos_gating)
                messages_by_source.mul_(pos_by_source)

            messages_by_targets.index_add_(0, edge_targets, messages_by_source)

            if self.msg_mean_aggregation:
                bins = edge_targets.bincount(minlength=node_states.size()[0])
                bincount += bins

        if self.msg_mean_aggregation:
            divisor = bincount.float()
            divisor[bincount == 0] = 1.0  # avoid div by zero for lonely nodes
            messages_by_targets /= divisor.unsqueeze_(1) + SMALL_NUMBER

        return messages_by_targets


class PositionEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, positions, demb, dpad: int = 0):
        """Transformer-like sinusoidal positional embeddings.
                Args:
                position: 1d long Tensor of positions,
                demb: int    size of embedding vector
            """
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))

        sinusoid_inp = torch.ger(positions, inv_freq)
        pos_emb = torch.cat(
            (torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1
        )

        if dpad > 0:
            in_length = positions.size()[0]
            pad = torch.zeros((in_length, dpad))
            pos_emb = torch.cat([pos_emb, pad], dim=1)
            assert torch.all(
                pos_emb[:, -1] == torch.zeros(in_length)
            ), f"test failed. pos_emb: \n{pos_emb}"

        return pos_emb

    # def forward(self, positions, dim, out):
    #     assert dim > 0, f'dim of position embs has to be > 0'
    #     power = 2 * (positions / 2) / dim
    #     position_enc = np.array(
    #         [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
    #          for pos in range(n_pos)])
    #     out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    #     out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    #     out.detach_()
    #     out.requires_grad = False


#######  Update Layers

class GGNNUpdateLayer(nn.Module):
    """GRU update function of GGNN architecture, optionally distinguishing two kinds of node types.
    Args:
        incoming messages <N, D+S> (from message layer),
        node_states <N, D+S>,
        node_types <N> (optional)
    Returns:
        updated node_states <N, D+S>
    """
    def __init__(self, config):
        super().__init__()
        self.dropout = config.graph_state_dropout
        # TODO(github.com/ChrisCummins/ProGraML/issues/27): Maybe decouple hidden
        # GRU size: make hidden GRU size larger and EdgeTrafo size non-square
        # instead? Or implement stacking gru layers between message passing steps.

        self.gru = nn.GRUCell(
            input_size=config.hidden_size, hidden_size=config.hidden_size
        )

        # currently only admits node types 0 and 1 for statements and identifiers.
        self.use_node_types = getattr(config, 'use_node_types', False)
        if self.use_node_types:
            self.id_gru = nn.GRUCell(
                input_size=config.hidden_size, hidden_size=config.hidden_size
            )

    def forward(self, messages, node_states, node_types=None):
        if self.use_node_types:
            assert node_types is not None, "Need to provide node_types <N> if config.use_node_types!"
            output = torch.zeros_like(node_states, device=node_states.device)
            stmt_mask = node_types == 0
            output[stmt_mask] = self.gru(messages[stmt_mask], node_states[stmt_mask])
            id_mask = node_types == 1
            output[id_mask] = self.id_gru(messages[id_mask], node_states[id_mask])
        else:
            output = self.gru(messages, node_states)

        if self.dropout > 0.0:
            F.dropout(output, p=self.dropout, training=self.training, inplace=True)
        return output

class TransformerUpdateLayer(nn.Module):
    """Represents the residual MLP around the self-attention in the transformer
    encoder layer. The implementation is sparse for usage in GNNs.

    Args:
        messages <N, D+S> (from self-attention layer)
        node_states <N, D+S>
        node_types <N> (optional and not yet implemented!)
    Returns:
        updated node_states
    """
    def __init__(self, config):
        super().__init__()
        activation = config.tfmr_act # relu or gelu, default relu
        dropout = config.tfmr_dropout # default 0.1
        dim_feedforward = config.tfmr_ff_sz # ~ 2.5 * model dim

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(config.hidden_size, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, config.hidden_size)

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self.get_activation_fn(activation)

    def get_activation_fn(self, activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        else:
            raise RuntimeError("activation should be relu/gelu, not %s." % activation)

    def forward(self, messages, node_states, node_types=None):
        assert node_types is None, 'not implemented.'

        # message layer is elsewhere!
        #messages = self.self_attn(src, src, src)[0]

        # 1st 'Add & Norm' block (cf. vaswani et al. 2017, fig. 1)
        node_states = node_states + self.dropout1(messages)
        node_states = self.norm1(node_states)

        # 'Feed Forward' block
        messages = self.linear2(self.dropout(self.activation(self.linear1(node_states))))

        # 2nd 'Add & Norm' block
        node_states = node_states + self.dropout2(messages)
        node_states = self.norm2(node_states)

        return node_states


########################################
# Output Layer
########################################


class Readout(nn.Module):
    """aka GatedRegression. See Eq. 4 in Gilmer et al. 2017 MPNN."""

    def __init__(self, config):
        super().__init__()
        self.has_graph_labels = config.has_graph_labels
        self.num_classes = config.num_classes

        self.regression_gate = LinearNet(
            2 * config.hidden_size, self.num_classes, dropout=config.output_dropout,
        )
        self.regression_transform = LinearNet(
            config.hidden_size, self.num_classes, dropout=config.output_dropout,
        )

    def forward(self, raw_node_in, raw_node_out, graph_nodes_list=None, num_graphs=None):
        gate_input = torch.cat((raw_node_in, raw_node_out), dim=-1)
        gating = torch.sigmoid(self.regression_gate(gate_input))
        nodewise_readout = gating * self.regression_transform(raw_node_out)

        graph_readout = None
        if self.has_graph_labels:
            assert graph_nodes_list is not None and num_graphs is not None, 'has_graph_labels requires graph_nodes_list and num_graphs tensors.'
            # aggregate via sums over graphs
            device = raw_node_out.device
            graph_readout = torch.zeros(num_graphs, self.num_classes, device=device)
            graph_readout.index_add_(
                dim=0, index=graph_nodes_list, source=nodewise_readout
            )
        return nodewise_readout, graph_readout


class LinearNet(nn.Module):
    """Single Linear layer with WeightDropout, ReLU and Xavier Uniform
    initialization. Applies a linear transformation to the incoming data:
    :math:`y = xA^T + b`

    Args:
    in_features: size of each input sample
    out_features: size of each output sample
    bias: If set to ``False``, the layer will not learn an additive bias.
    Default: ``True``

    Shape:
    - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
    additional dimensions and :math:`H_{in} = \text{in\_features}`
    - Output: :math:`(N, *, H_{out})` where all but the last dimension
    are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    """

    def __init__(self, in_features, out_features, bias=True, dropout=0.0, gain=1.0):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.gain = gain
        self.test = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.test, gain=self.gain)
        if self.bias is not None:
            #    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            #    bound = 1 / math.sqrt(fan_in)
            #    nn.init.uniform_(self.bias, -bound, bound)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        if self.dropout > 0.0:
            w = F.dropout(self.test, p=self.dropout, training=self.training)
        else:
            w = self.test
        return F.linear(input, w, self.bias)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, dropout={}".format(
            self.in_features, self.out_features, self.bias is not None, self.dropout,
        )


#######################################
# Adding Graph Level Features to Model

class AuxiliaryReadout(nn.Module):
  """Produces per-graph predictions by combining
    the per-graph predictions with auxiliary features"""

  # TODO(github.com/ChrisCummins/ProGraML/issues/27): I don't like that we only
  # introduce the global features AFTER the per node predictions have been made
  # and not while we do those! This is limiting the expressivity of the model.
  def __init__(self, config):
    super().__init__()
    self.num_classes = config.num_classes
    self.log1p_graph_x = getattr(config, "log1p_graph_x", False)
    assert (
      config.has_graph_labels
    ), "We expect aux readout in combination with graph labels, not node labels"
    self.feed_forward = None

    self.batch_norm = nn.BatchNorm1d(config.num_classes + config.aux_in_len)
    self.feed_forward = nn.Sequential(
      nn.Linear(
        config.num_classes + config.aux_in_len, config.aux_in_layer_size,
      ),
      nn.ReLU(),
      nn.Dropout(1 - config.output_dropout),
      nn.Linear(config.aux_in_layer_size, config.num_classes),
    )

  def forward(self, graph_features, auxiliary_features):
    assert (
      graph_features.size()[0] == auxiliary_features.size()[0]
    ), "every graph needs aux_features. Dimension mismatch."
    if self.log1p_graph_x:
      auxiliary_features.log1p_()

    aggregate_features = torch.cat((graph_features, auxiliary_features), dim=1)

    normed_features = self.batch_norm(aggregate_features)
    out = self.feed_forward(normed_features)
    return out, graph_features


###########################
# GGNNInput: Embeddings
###########################
class NodeEmbeddingsForPretraining(nn.Module):
    """NodeEmbeddings with added embedding for [MASK] token."""

    def __init__(self, config):
        super().__init__()

        print("Initializing with random embeddings for pretraining.")
        self.node_embs = nn.Embedding(config.vocab_size + 1, config.emb_size)

    def forward(self, vocab_ids):
        embs = self.node_embs(vocab_ids)
        return embs


class NodeEmbeddings(nn.Module):
    """Construct node embeddings from node ids
    Args:
    pretrained_embeddings (Tensor, optional) – FloatTensor containing weights for
    the Embedding. First dimension is being passed to Embedding as
    num_embeddings, second as embedding_dim.

    Forward
    Args:
    vocab_ids: <N, 1>
    Returns:
    node_states: <N, config.hidden_size>
    """

    # TODO(github.com/ChrisCummins/ProGraML/issues/27):: Maybe LayerNorm and
    # Dropout on node_embeddings?
    # TODO(github.com/ChrisCummins/ProGraML/issues/27):: Make selector embs
    # trainable?

    def __init__(self, config, pretrained_embeddings=None):
        super().__init__()
        self.inst2vec_embeddings = config.inst2vec_embeddings
        self.emb_size = config.emb_size

        if config.inst2vec_embeddings == "constant":
            print("Using pre-trained inst2vec embeddings frozen.")
            assert pretrained_embeddings is not None
            self.node_embs = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=True
            )
        elif config.inst2vec_embeddings == "zero":
            init = torch.zeros(config.vocab_size, config.emb_size)
            self.node_embs = nn.Embedding.from_pretrained(init, freeze=True)
        elif config.inst2vec_embeddings == "constant_random":
            init = torch.rand(config.vocab_size, config.emb_size)
            self.node_embs = nn.Embedding.from_pretrained(init, freeze=True)
        elif config.inst2vec_embeddings == "finetune":
            print("Fine-tuning inst2vec embeddings")
            assert pretrained_embeddings is not None
            self.node_embs = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=False
            )
        elif config.inst2vec_embeddings == "random":
            print("Initializing with random embeddings")
            self.node_embs = nn.Embedding(config.vocab_size, config.emb_size)
        elif config.inst2vec_embeddings == "none":
            print("Initializing with a embedding for statements and identifiers each.")
            self.node_embs = nn.Embedding(2, config.emb_size)
        else:
            raise NotImplementedError(config.inst2vec_embeddings)


    def forward(self, vocab_ids, *ignored_args, **ignored_kwargs):
        if self.inst2vec_embeddings == 'none':
            # map IDs to 1 and everything else to 0
            ids = (vocab_ids == 8565).to(torch.long)  # !IDENTIFIER token id
            embs = self.node_embs(ids)
        else:  # normal embeddings
            embs = self.node_embs(vocab_ids)

        return embs


class NodeEmbeddingsWithSelectors(NodeEmbeddings):
    """Construct node embeddings as content embeddings + selector embeddings.

    Args:
    pretrained_embeddings (Tensor, optional) – FloatTensor containing weights for
    the Embedding. First dimension is being passed to Embedding as
    num_embeddings, second as embedding_dim.

    Forward
    Args:
    vocab_ids: <N, 1>
    selector_ids: <N, 1>
    Returns:
    node_states: <N, config.hidden_size>
    """
    def __init__(self, config, pretrained_embeddings=None):
        super().__init__(config, pretrained_embeddings)

        self.node_embs = super().forward
        assert config.use_selector_embeddings, "This Module is for use with use_selector_embeddings!"

        selector_init = torch.tensor(
            # TODO(github.com/ChrisCummins/ProGraML/issues/27): x50 is maybe a
            # problem for unrolling (for selector_embs)?
            [[0, 50.0], [50.0, 0]],
            dtype=torch.get_default_dtype(),
        )
        self.selector_embs = nn.Embedding.from_pretrained(
            selector_init, freeze=True
        )

    def forward(self, vocab_ids, selector_ids):
        node_embs = self.node_embs(vocab_ids)
        selector_embs = self.selector_embs(selector_ids)
        embs = torch.cat((node_embs, selector_embs), dim=1)
        return embs


#############################
# Loss Accuracy Prediction
#############################


class Loss(nn.Module):
    """[Binary] Cross Entropy loss with weighted intermediate loss"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.num_classes == 1:
            self.loss = nn.BCELoss()  # in: (N, *), target: (N, *)
        else:
            # class labels '-1' don't contribute to the gradient!
            # however in most cases it will be more efficient to gather
            # the relevant data into a dense tensor
            self.loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

    def forward(self, inputs, targets):
        """inputs: (predictions) or (predictions, intermediate_predictions)"""
        loss = self.loss(inputs[0], targets)
        if getattr(self.config, 'has_aux_input', False):
            loss += self.config.intermediate_loss_weight * self.loss(
                inputs[1], targets
            )
        return loss

class MLMLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, logits, targets, mlm_target_mask):
        assert len(logits[mlm_target_mask, :].shape) == 2 , f"len is {len(logits[mlm_target_mask].shape)} not 2"
        assert logits[mlm_target_mask, :].size()[-1] == self.config.vocab_size + 1, f"should have vcab_size + 1 classes not {logits[mlm_target_mask].size()[-1]}"
        loss = F.nll_loss(
            F.log_softmax(
                #logits.view(-1, logits.size(-1)),
                logits[mlm_target_mask, :],
                dim=-1,
                dtype=torch.float32,
            ),
            #targets.view(-1),
            targets[mlm_target_mask],
            reduction='mean',
            ignore_index=-1,
        )
        return loss


class Metrics(nn.Module):
    """Common metrics and info for inspection of results.
    Args:
    logits, labels
    Returns:
    (accuracy, pred_targets, correct_preds, targets)"""

    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        # be flexible with 1hot labels vs indices
        if len(labels.size()) == 2:
            targets = labels.argmax(dim=1)
        elif len(labels.size()) == 1:
            targets = labels
        else:
            raise ValueError(f"labels={labels.size()} tensor is is neither 1 nor 2-dimensional. :/")


        pred_targets = logits.argmax(dim=1)
        correct_preds = targets.eq(pred_targets).float()
        accuracy = torch.mean(correct_preds)
        return accuracy, logits, correct_preds, targets


# Huggingface implementation
# perplexity = torch.exp(torch.tensor(eval_loss)), where loss is just the ave
