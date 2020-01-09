import pickle, time, os, json, sys
from pathlib import Path

import tqdm
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader, InMemoryDataset


# make this file executable from anywhere
if __name__ == '__main__':
    full_path = os.path.realpath(__file__)
    print(full_path)
    root = full_path.rsplit('ProGraML', maxsplit=1,)[0] + 'ProGraML'
    print(root)
    #insert at 1, 0 is the script path (or '' in REPL)
    sys.path.insert(1, root)


from deeplearning.ml4pl.models.ggnn.ggnn_modules import GGNNModel
from deeplearning.ml4pl.models.ggnn.ggnn_config_poj104 import GGNNConfig


class Learner(object):
    def __init__(self):

        self.args = {
            '--log_dir': 'deeplearning/ml4pl/poj104/classifyapp_logs/',
            '--data_dir': 'deeplearning/ml4pl/poj104/classifyapp_data',
            #'--test_only',
            }
        # override args from file
        #args_file = Path('deeplearning/ml4pl/poj104/dataset_.json')
        #if args_file.exists():
        #    with open(args_file, 'r') as f:
        #        update_args = json.load(f)
        #    self.args.update(update_args)

        # prepare logging
        self.run_id = "_".join([time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])
        log_dir = self.args.get("--log_dir", '.')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_file = os.path.join(log_dir, "%s_log.json" % self.run_id)
        self.test_log_file = os.path.join(log_dir, f"{self.run_id}_test_log.json")
        self.best_model_file = os.path.join(
            log_dir, "%s_model_best.pickle" % self.run_id
        )

        # load config
        self.config = GGNNConfig()

        # load data
        self.data_dir = self.args.get('--data_dir', '.')
        self.train_data = self.get_dataloader(self.data_dir + '/ir_train', self.config)
        self.valid_data = self.get_dataloader(self.data_dir + '/ir_val', self.config)
        #self.test_data = self.get_dataloader(self.data_dir + '/ir_test', self.config)

        # create model
        self.model = GGNNModel(self.config)
        self.global_training_step = 0
        self.current_epoch = 1

        # set some global config values
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # move model to gpu
        self.model.to(self.dev)

        # log config to file
        config_dict = {a: getattr(self.config, a) for a in dir(self.config) if not a.startswith('__')}
        with open(os.path.join(log_dir, "%s_params.json" % self.run_id), "w") as f:
            json.dump(config_dict, f)
        print(
            "Run %s starting with following parameters:\n%s"
            % (self.run_id, json.dumps(config_dict))
        )

    def get_dataloader(self, ds_base, config):
        ds_base = Path(ds_base) if type(ds_base) is str else ds_base
        datalist = []
        out_base = ds_base.parent / (ds_base.name + "_programl")
        print(f"=== DATASET {ds_base}: getting dataloader")

        folders = [x for x in out_base.glob("*") if x.is_dir() and x.name not in ['_nx', '_tuples']]
        for folder in tqdm.tqdm(folders):
            # skip classes that are larger than what config says to enable debugging with less data
            if int(folder.name) > config.num_classes:
                continue
            # print(f"=== Opening Folder {str(folder)} ===")
            for k, file in enumerate(folder.glob("*.data.p")):
                # print(f"{k} - Processing {str(file)} ...")
                with open(file, "rb") as f:
                    data = pickle.load(f)
                datalist.append(data)
        print(f" * COMPLETED * === DATASET {ds_base}: returning dataloader")
        return DataLoader(datalist, batch_size=config.batch_size, shuffle=True)

    def run_epoch(self, loader, epoch_type):
        """
        args:
            loader: a pytorch-geometric dataset loader,
            epoch_type: 'train' or 'eval'
        returns:
            loss, accuracy, instance_per_second
        """

        bar = tqdm.tqdm(loader)

        epoch_loss = 0
        accuracies = []
        start_time = time.time()
        processed_graphs = 0

        for step, batch in enumerate(bar):
            self.global_training_step += 1
            num_graphs = batch.batch[-1].item() + 1
            processed_graphs += num_graphs

            ######### prepare input
            # move batch to gpu and prepare input tensors:
            batch.to(self.dev)

            edge_lists = []
            edge_positions = []
            for i in range(3):
                # mask by edge type
                mask = batch.edge_attr[:, 0].squeeze() == i # <M_i>
                edge_list = batch.edge_index[:, mask].t()
                edge_lists.append(edge_list)
            
                #[torch.zeros_like(edge_lists[i])[:, 1] for i in range(3)]
                edge_pos = batch.edge_attr[mask, 1]
                edge_positions.append(edge_pos)


            #############
            # enter correct mode of model
            if epoch_type == "train":
                if not self.model.training:
                    self.model.train()
                outputs = self.model(
                    vocab_ids=batch.x.squeeze(),
                    selector_ids=torch.zeros_like(batch.x).to(self.dev), # move to dev #TODO make selectors optional
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
                        selector_ids=torch.zeros_like(batch.x).to(self.dev), # move to dev #TODO
                        labels=batch.y - 1,
                        edge_lists=edge_lists,
                        pos_lists=edge_positions,
                        num_graphs=num_graphs,
                        graph_nodes_list=batch.batch,
                    )

            # RUN MODEL FORWARD PASS

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

        mean_loss = epoch_loss / processed_graphs
        accuracy = np.sum(accuracies, axis=0) / processed_graphs
        instance_per_sec = processed_graphs / (time.time() - start_time)
        return mean_loss, accuracy, instance_per_sec

    def train(self, start_epoch=1):
        log_to_save = []
        total_time_start = time.time()
        if False:
            pass
            # TODO implement restoring!
        #if self.args.get("--restore") is not None:
        #    _, valid_acc, _, _ = self.run_epoch(
        #        "Resumed (validation)", self.valid_data, "val"
        #    )
        #    best_val_acc = np.sum(valid_acc)
        #    best_val_acc_epoch = 0
        #    print(
        #        "\r\x1b[KResumed operation, initial cum. val. acc: %.5f"
        #        % best_val_acc
        #    )
        else:
            (best_val_acc, best_val_acc_epoch) = (0.0, 0)
        for epoch in range(start_epoch, self.config.num_epochs):
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

            epoch_time = time.time() - total_time_start
            log_entry = {
                "epoch": epoch,
                "time": epoch_time,
                "train_results": (train_loss, train_acc.tolist(), train_speed),
                "valid_results": (valid_loss, valid_acc.tolist(), valid_speed),
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
        #TODO: test this
        checkpoint = torch.load(path)
        self.run_id = checkpoint['run_id']
        self.global_training_step = checkpoint['global_training_step']
        # epoch = checkpoint['epoch']
        config = checkpoint['config']
        assert config == self.config
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if not self.args.get('--test_only', None):  # only restore opt if needed. opt should be None o/w.
            self.model.opt.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == '__main__':
    learner = Learner()
    learner.train()
