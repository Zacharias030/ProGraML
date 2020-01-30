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
"""C."""
from typing import List

class ProGraMLBaseConfig(object):
    def __init__(self):
        self.name = self.__class__.__name__

        # Training Hyperparameters
        self.num_epochs = 25
        self.batch_size = 128
        self.lr: float = 0.00025
        self.patience = 100
        self.clip_grad_norm: float = 0.0
        self.train_subset = [0, 100]
        self.random_seed: int = 42

        # Model Hyperparameters
        self.emb_size: int = 200
        self.edge_type_count: int = 3

        self.vocab_size: int = 8568

        # inst2vec_embeddings can now be 'none' as well!
        # this reduces the tokens that the network sees to only
        # !IDENTIFIERs and !UNK statements
        #  One of {zero, constant, random, random_const, finetune, none}
        self.inst2vec_embeddings = 'random'

    @classmethod
    def from_dict(cls, params):
        """instantiate Config from params dict that overrides default values where given."""
        config = cls()
        if params is None:
            return config

        for key in params:
            if hasattr(config, key):
                setattr(config, key, params[key])
            else:
                print(f"(*CONFIG FROM DICT*  Default {config.name} doesn't have a key {key}. Will add key to config anyway!")
                setattr(config, key, params[key])
        return config

    def to_dict(self):
        config_dict = {a: getattr(self, a) for a in dir(self) if not a.startswith('__') and not callable(getattr(self, a))}
        return config_dict

    def check_equal(self, other):
        # take either config object or config_dict
        other_dict = other if isinstance(other, dict) else other.to_dict()
        if not self.to_dict() == other_dict:
            print(f"WARNING: GGNNConfig.check_equal() FAILED:\nself and other are unequal: "
                  f"The difference is {set(self.to_dict()) ^ set(other.to_dict())}.\n self={self.to_dict()}\n other={other_dict}")

class GGNN_POJ104_Config(ProGraMLBaseConfig):
    def __init__(self):
        super().__init__()
        ###############
        # Model Hyperparameters
        self.layer_timesteps: List[int] = [2, 2, 2, 2]
        # currently only admits node types 0 and 1 for statements and identifiers.
        self.use_node_types: bool = False
        self.use_edge_bias: bool = True
        self.position_embeddings: bool = False

        # Aggregate by mean or by sum
        self.msg_mean_aggregation: bool = True
        self.backward_edges: bool = True

        ###############
        # Regularization
        self.output_dropout: float = 0.0  # dropout prob = 1-keep_prob
        self.edge_weight_dropout: float = 0.0
        self.graph_state_dropout: float = 0.2

        ###############
        # Dataset inherent, don't change!
        self.num_classes: int = 104
        self.has_graph_labels: bool = True
        self.has_aux_input: bool = False

        # self.use_selector_embeddings: bool = False
        # self.selector_size: int = 2 if getattr(self, 'use_selector_embeddings', False) else 0
        # TODO(Zach) Maybe refactor non-rectangular edge passing matrices for independent hidden size.
        # hidden size of the whole model
        self.hidden_size: int = self.emb_size + getattr(self, 'selector_size', 0)

class GraphTransformerConfig(ProGraMLBaseConfig):
    def __init__(self):
        super().__init__()
        self.backward_edges: bool = True
        self.layer_timesteps: List[int] = [2, 2, 2, 2]
        self.use_node_types: bool = False
        self.position_embeddings: bool = False

        # Dataset inherent, don't change!
        self.num_classes: int = 104
        self.has_graph_labels: bool = True
        self.hidden_size: int = self.emb_size + getattr(self, 'selector_size', 0)

        # Self-Attn Layer
        self.attn_bias = True
        self.attn_num_heads = 8 # choose among 4,5,8,10 for emb_sz 200
        self.attn_dropout = 0.1

        # Transformer Update Layer
        self.tfmr_act = 'relu' # relu or gelu, default relu
        self.tfmr_dropout = 0.1 # default 0.1
        self.tfmr_ff_sz = 512 # ~ 2.5 model_dim (Bert: 768 - 2048, Trfm: base 512 - 2048, big 1024 - 4096)


class GGNN_POJ104_ForPretraining_Config(GGNN_POJ104_Config):
    def __init__(self):
        super().__init__()
        # Pretraining Parameters
        self.mlm_probability = 0.15
        self.mlm_statements_only = True
        self.mlm_exclude_unk_tokens = True
        self.mlm_mask_token_id = 8568
        self.unk_token_id = 8564

        # set for pretraining to vocab_size + 1 [MASK]
        self.num_classes = self.vocab_size + 1
        self.has_graph_labels: bool = False
