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
"""Configuration for GGNN models."""
from typing import List

class GGNNConfig(object):
    def __init__(self):
        # Training Hyperparameters
        self.num_epochs = 150
        self.batch_size = 32
        self.lr: float = 0.0005
        self.patience = 100
        self.clip_grad_norm: float = 0.0
        self.train_subset = [0, 100]
        self.random_seed: int = 42

        ###############
        # Model Hyperparameters
        self.layer_timesteps: List[int] = [2, 2, 2]
        self.emb_size: int = 200
        # currently only admits node types 0 and 1 for statements and identifiers.
        self.use_node_types = False
        self.use_edge_bias: bool = True
        
        # Aggregate by mean or by sum
        self.msg_mean_aggregation: bool = True
        self.backward_edges: bool = True

        ###############
        # Regularization
        self.output_dropout: float = 0.0  # dropout prob = 1-keep_prob
        self.edge_weight_dropout: float = 0.0
        self.graph_state_dropout: float = 0.1

        # TODO(github.com/ChrisCummins/ProGraML/issues/27):: Maybe refactor non-rectangular edge passing matrices for independent hidden size.
        # hidden size of the whole model
        self.position_embeddings: bool = False


        ###############
        # Model and dataset inherent, don't change!
        self.num_classes: int = 104
        self.edge_type_count: int = 3
        self.has_graph_labels: bool = True
        self.has_aux_input: bool = False
        self.vocab_size: int = 8568
        self.inst2vec_embeddings = 'random' #  One of {zero, constant, random, random_const, finetune}
        # could be made optional 
        self.use_selector_embeddings: bool = False
        self.selector_size: int = 2 if self.use_selector_embeddings else 0
        self.hidden_size: int = self.emb_size + self.selector_size


    @classmethod
    def from_dict(cls, params):
        """instantiate Config from params dict that overrides default values where given."""
        config = cls()
        if params is None:
            return config

        for key in params:
            if hasattr(config, key):
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
