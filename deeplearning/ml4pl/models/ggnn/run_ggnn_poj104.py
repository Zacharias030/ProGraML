"""
Usage:
   run_ggnn_poj104.py [options]

Options:
    -h --help                       Show this screen.
    --data_dir=DATA_DIR             Directory(*) to of dataset. (*)=relative to repository root ProGraML/.
                                        [default: deeplearning/ml4pl/poj104/classifyapp_data]
    --log_dir LOG_DIR               Directory(*) to store logfiles and trained models relative to repository dir.
                                        [default: deeplearning/ml4pl/poj104/classifyapp_logs/]
    --config CONFIG                 Path(*) to a config json dump with params.
    --config_json CONFIG_JSON       Config json with params.
    --restore CHECKPOINT            Path(*) to a model file to restore from.
    --skip_restore_config           Whether to skip restoring the config from CHECKPOINT.
    --test                          Test the model without training.
"""

import pickle, time, os, json, sys
from pathlib import Path

from docopt import docopt
import tqdm
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader, InMemoryDataset


# make this file executable from anywhere
#if __name__ == '__main__':
full_path = os.path.realpath(__file__)
print(full_path)
repo_root = full_path.rsplit('ProGraML', maxsplit=1)[0] + 'ProGraML'
print(repo_root)
#insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, repo_root)
repo_root = Path(repo_root)


from deeplearning.ml4pl.models.ggnn.ggnn_modules import GGNNModel
from deeplearning.ml4pl.models.ggnn.ggnn_config_poj104 import GGNNConfig
from deeplearning.ml4pl.poj104.dataset import POJ104Dataset

class Learner(object):
    def __init__(self, args=None):
        self.args = args

        # Make class work without file being run as main
        if self.args is None:
            self.args = docopt(__doc__, argv=[])

        # prepare logging
        self.run_id = "_".join([time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])
        log_dir = repo_root / self.args.get("--log_dir", '.')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_file = os.path.join(log_dir, "%s_log.json" % self.run_id)
        self.test_log_file = os.path.join(log_dir, f"{self.run_id}_test_log.json")
        self.best_model_file = os.path.join(
            log_dir, "%s_model_best.pickle" % self.run_id
        )

        # load config
        params = None
        if self.args.get('--config', None) is not None:
            with open(repo_root / self.args['--config'], 'r') as f:
                params = json.load(f)
        elif self.args.get('--config_json', None) is not None:
            config_string = args['--config_json']
            # accept single quoted 'json'. This only works bc our json strings are simple enough.
            config_string = config_string.replace("'", '"')
            params = json.loads(config_string)
        self.config = GGNNConfig.from_dict(params=params)

        # set seeds, NB: the NN on CUDA is partially non-deterministic!
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        # create / restore model
        if self.args.get('--restore', None) is not None:
            self.model = self.restore_model(path=repo_root / self.args['--restore'])
        else: # initialize fresh model
            self.global_training_step = 0
            self.current_epoch = 1
            test_only = self.args.get('--test', False)
            self.model = GGNNModel(self.config, test_only=test_only)

        # load data
        self.data_dir = repo_root / self.args.get('--data_dir', '.')
        self.valid_data = DataLoader(POJ104Dataset(root=self.data_dir, split='val'), batch_size=self.config.batch_size * 2, shuffle=False)
        self.test_data = DataLoader(POJ104Dataset(root=self.data_dir, split='test'), batch_size=self.config.batch_size * 2, shuffle=False)
        
        self.train_data = None
        if not self.args.get('--test', None):
            self.train_data = DataLoader(POJ104Dataset(root=self.data_dir, split='train', train_subset=self.config.train_subset), batch_size=self.config.batch_size, shuffle=True)

        # log config to file
        config_dict = self.config.to_dict()
        with open(os.path.join(log_dir, "%s_params.json" % self.run_id), "w") as f:
            json.dump(config_dict, f)
        print(
            "Run %s starting with following parameters:\n%s"
            % (self.run_id, json.dumps(config_dict))
        )

    def run_epoch(self, loader, epoch_type):
        """
        args:
            loader: a pytorch-geometric dataset loader,
            epoch_type: 'train' or 'eval'
        returns:
            loss, accuracy, instance_per_second
        """

        bar = tqdm.tqdm(total=len(loader) * loader.batch_size, smoothing=0.01, unit='inst')

        epoch_loss = 0
        accuracies = []
        start_time = time.time()
        processed_graphs = 0

        for step, batch in enumerate(loader):
            num_graphs = batch.batch[-1].item() + 1
            processed_graphs += num_graphs

            ######### prepare input
            # move batch to gpu and prepare input tensors:
            batch.to(self.model.dev)

            edge_lists = []
            edge_positions = [] if self.config.position_embeddings else None
            for i in range(3):
                # mask by edge type
                mask = batch.edge_attr[:, 0].squeeze() == i # <M_i>
                edge_list = batch.edge_index[:, mask].t()
                edge_lists.append(edge_list)
                
                if self.config.position_embeddings:
                    edge_pos = batch.edge_attr[mask, 1]
                    edge_positions.append(edge_pos)


            #############
            # RUN MODEL FORWARD PASS

            # enter correct mode of model
            if epoch_type == "train":
                self.global_training_step += 1

                if not self.model.training:
                    self.model.train()
                outputs = self.model(
                    vocab_ids=batch.x.squeeze(),
                    labels=batch.y - 1, # labels start at 0!!!
                    edge_lists=edge_lists,
                    pos_lists=edge_positions,
                    num_graphs=num_graphs,
                    graph_nodes_list=batch.batch,
                )
            else:  # not TRAIN
                if self.model.training:
                    self.model.eval()
                    self.model.opt.zero_grad()
                with torch.no_grad():  # don't trace computation graph!
                    outputs = self.model(
                        vocab_ids=batch.x.squeeze(),
                        labels=batch.y - 1,
                        edge_lists=edge_lists,
                        pos_lists=edge_positions,
                        num_graphs=num_graphs,
                        graph_nodes_list=batch.batch,
                    )

            (
                logits,
                accuracy,
                logits,
                correct,
                targets,
                graph_features,
                *unroll_stats,
            ) = outputs

            loss = self.model.loss((logits, graph_features), targets)
            epoch_loss += loss.item() * num_graphs
            accuracies.append(np.array(accuracy.item()) * num_graphs)

            if epoch_type == "train":
                loss.backward()
                # TODO(github.com/ChrisCummins/ProGraML/issues/27):: Clip gradients
                # (done). NB, pytorch clips by norm of the gradient of the model, while
                # tf clips by norm of the grad of each tensor separately. Therefore we
                # change default from 1.0 to 6.0.
                # TODO(github.com/ChrisCummins/ProGraML/issues/27):: Anyway: Gradients
                # shouldn't really be clipped if not necessary?
                if self.model.config.clip_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.model.config.clip_grad_norm
                    )
                self.model.opt.step()
                self.model.opt.zero_grad()

            bar.set_postfix(loss=epoch_loss / processed_graphs, acc=np.sum(accuracies, axis=0) / processed_graphs)
            bar.update(num_graphs)

        bar.close()
        mean_loss = epoch_loss / processed_graphs
        accuracy = np.sum(accuracies, axis=0) / processed_graphs
        instance_per_sec = processed_graphs / (time.time() - start_time)
        return mean_loss, accuracy, instance_per_sec

    def train(self):
        log_to_save = []
        total_time_start = time.time()

        # we enter training after restore
        if self.args.get("--restore") is not None:
            print(f"== Epoch pre-validate epoch {self.current_epoch}")
            _, valid_acc, _, = self.run_epoch(self.valid_data, "val")
            best_val_acc = np.sum(valid_acc)
            best_val_acc_epoch = self.current_epoch
            print(
                "\r\x1b[KResumed operation, initial cum. val. acc: %.5f"
                % best_val_acc
            )
        else:
            (best_val_acc, best_val_acc_epoch) = (0.0, 0)

        # Training loop over epochs
        for epoch in range(self.current_epoch, self.config.num_epochs):
            print(f"== Epoch {epoch}/{self.config.num_epochs}")

            train_loss, train_acc, train_speed = self.run_epoch(
                self.train_data, "train"
            )
            print(
                "\r\x1b[K Train: loss: %.5f | acc: %s | instances/sec: %.2f"
                % (train_loss, f"{train_acc:.5f}", train_speed)
            )

            valid_loss, valid_acc, valid_speed = self.run_epoch(
                self.valid_data, "eval"
            )
            print(
                "\r\x1b[K Valid: loss: %.5f | acc: %s | instances/sec: %.2f"
                % (valid_loss, f"{valid_acc:.5f}", valid_speed)
            )

            test_loss, test_acc, test_speed = self.run_epoch(
                self.test_data, "eval"
            )
            print(
                "\r\x1b[K Test: loss: %.5f | acc: %s | instances/sec: %.2f"
                % (test_loss, f"{test_acc:.5f}", test_speed)
            )

            epoch_time = time.time() - total_time_start
            self.current_epoch = epoch

            log_entry = {
                "epoch": epoch,
                "time": epoch_time,
                "train_results": (train_loss, train_acc.tolist(), train_speed),
                "valid_results": (valid_loss, valid_acc.tolist(), valid_speed),
                "test_results": (test_loss, test_acc.tolist(), test_speed),
            }
            log_to_save.append(log_entry)
            with open(self.log_file, "w") as f:
                json.dump(log_to_save, f, indent=4)

            # TODO: sum seems redundant if only one task is trained.
            val_acc = np.sum(valid_acc)  # type: float
            if val_acc > best_val_acc:
                self.save_model(epoch, self.best_model_file)
                print(
                    "  (Best epoch so far, cum. val. acc increased to %.5f from %.5f. Saving to '%s')"
                    % (val_acc, best_val_acc, self.best_model_file)
                )
                best_val_acc = val_acc
                best_val_acc_epoch = epoch
            elif epoch - best_val_acc_epoch >= self.config.patience:
                print(
                    "Stopping training after %i epochs without improvement on validation accuracy."
                    % self.config.patience
                )
                break

    def test(self):
        log_to_save = []
        total_time_start = time.time()

        print(f"== Epoch: Test only run.")

        valid_loss, valid_acc, valid_speed = self.run_epoch(
            self.valid_data, "eval"
        )
        print(
            "\r\x1b[K Valid: loss: %.5f | acc: %s | instances/sec: %.2f"
            % (valid_loss, f"{valid_acc:.5f}", valid_speed)
        )

        test_loss, test_acc, test_speed = self.run_epoch(
            self.test_data, "eval"
        )
        print(
            "\r\x1b[K Test: loss: %.5f | acc: %s | instances/sec: %.2f"
            % (test_loss, f"{test_acc:.5f}", test_speed)
        )

        epoch_time = time.time() - total_time_start
        log_entry = {
            "epoch": 'test_only',
            "time": epoch_time,
            "valid_results": (valid_loss, valid_acc.tolist(), valid_speed),
            "test_results": (test_loss, test_acc.tolist(), test_speed),
        }
        log_to_save.append(log_entry)
        with open(self.log_file, "w") as f:
            json.dump(log_to_save, f, indent=4)

    def save_model(self, epoch, path):
        checkpoint = {
            'run_id': self.run_id,
            'global_training_step': self.global_training_step,
            'epoch': epoch,
            'config': self.config,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.model.opt.state_dict(),
        }
        torch.save(checkpoint, path)

    def restore_model(self, path):
        """loads and restores a model from file."""
        checkpoint = torch.load(path)
        self.run_id = checkpoint['run_id']
        self.global_training_step = checkpoint['global_training_step']
        self.current_epoch = checkpoint['epoch']

        config_dict = checkpoint['config'] if isinstance(checkpoint['config'], dict) else checkpoint['config'].to_dict()
        if not self.args.get('--skip_restore_config'):
            config = GGNNConfig.from_dict(config_dict)
            self.config = config
            print(f'*RESTORED* self.config from checkpoint {str(path)}.')
        else:
            print(f'Skipped restoring self.config from checkpoint!')
            self.config.check_equal(config_dict)
        
        test_only = self.args.get('--test', False)
        model = GGNNModel(self.config, test_only=test_only)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'*RESTORED* model parameters from checkpoint {str(path)}.')
        if not self.args.get('--test', None):  # only restore opt if needed. opt should be None o/w.
            model.opt.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'*RESTORED* optimizer parameters from checkpoint as well.')
        return model


if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)
    assert not (args['--config'] and args['--config_json']), "Can't decide which config to use!"
    learner = Learner(args=args)
    if args.get('--test', None):
        learner.test()
    else:
        learner.train()
