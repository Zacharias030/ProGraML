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
    self.num_epochs = 300
    self.num_classes: int = 3 # 104
    self.batch_size = 4
    self.lr: float = 0.001
    self.patience = 100
    self.clip_grad_norm: float = 6.0

    self.vocab_size: int = 8568
    self.inst2vec_embeddings = 'random' #  One of {zero, constant, random, random_const, finetune}
    self.emb_size: int = 200
    self.use_selector_embeddings: bool = False
    self.selector_size: int = 2 if self.use_selector_embeddings else 0
    # TODO(github.com/ChrisCummins/ProGraML/issues/27):: Maybe refactor non-rectangular edge passing matrices for independent hidden size.
    # hidden size of the whole model
    self.hidden_size: int = self.emb_size + self.selector_size
    self.position_embeddings: bool = False #FLAGS.position_embeddings #TODO(Zach)
    ###############

    self.edge_type_count: int = 3
    self.layer_timesteps: List[int] = [2, 2, 2]
    self.use_edge_bias: bool = True
    self.msg_mean_aggregation: bool = True
    self.backward_edges: bool = True
    ###############

    self.output_dropout: float = 0.0  # dropout prob = 1-keep_prob
    self.edge_weight_dropout: float = 0.0
    self.graph_state_dropout: float = 0.0
    ###############

    self.has_graph_labels: bool = True
    self.has_aux_input: bool = False
    self.log1p_graph_x = False

    self.intermediate_loss_weight: float = 0.2
    #########
    self.unroll_strategy = 'none'
    self.test_layer_timesteps = '0'
    self.max_timesteps = 1000
    self.label_conv_threshold: float = 0.995
    self.label_conv_stable_steps: int = 1
