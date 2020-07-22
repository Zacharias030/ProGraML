import pandas as pd
from itertools import cycle

import pickle, json
from matplotlib import pyplot as plt
import numpy as np
import tqdm
import torch


def get_all_runs(log_dir, subfolders=False, exclude=['test_only'], upper_bound_epochs=None, fuse_by_name=False, silent=False):
    print(f"Getting all runs from {log_dir.name}")
    assert subfolders is False, 'not implemented'
    logs = {}
    hyps = {}
    for file in sorted(list(log_dir.glob('*_log.json'))):
        
        try:
            with open(file, 'r') as f:
                log = pd.read_json(f, orient='records')
        except ValueError as e:
            try:
                with open(file, 'r') as f:
                    jsondata = json.load(f)
                    log = pd.DataFrame(jsondata)
            except Exception as ee:
                print(f'failing on {file}!')
                raise ee
                continue
            #raise e
        # handle 'test_only' epochs later!
        if log['epoch'].values[0] == 'test_only':
            continue

        run_name = file.name.rsplit('_log.json')[0]
        # skip weird files
        if run_name[:2] == '._':
            continue
        if not silent:
            print(run_name)
        if fuse_by_name:
            # check if we are in kfold mode
            if len(run_name.rsplit('_', maxsplit=1)[-1]) <= 2:
                splits = run_name.rsplit('_', maxsplit=3)
                short_run_name = splits[1] + '_' + splits[3]
            else:
                short_run_name = run_name.rsplit('_', maxsplit=2)[1]

        else:
            short_run_name = run_name

        # ~~~~~~~ try loading hyps
        hyp_file = file.parent / (run_name + '_params.json')
        try:
            with open(hyp_file, 'r') as h:
                hyp = json.load(h)
        except FileNotFoundError:
            hyp = None
        except UnicodeDecodeError as e:
            print(hyp_file)
            raise e
        except:
            hyp = None

        if short_run_name not in hyps:
            hyps[short_run_name] = hyp
        else:
            assert hyp == hyps[short_run_name], f"expected {hyps[short_run_name]}\n but got {hyp}!"

        # flatten dataframe
        try:
            valid = pd.DataFrame(log.valid_results.tolist(), columns=['valid_loss', 'valid_acc', 'valid_speed'])
            train = pd.DataFrame(log.train_results.tolist(), columns=['train_loss', 'train_acc', 'train_speed'])
        except AssertionError:
            try:
                valid = pd.DataFrame(log.valid_results.tolist(), columns=['valid_loss', 'valid_acc', 'valid_speed', 'valid_ppl'])
                train = pd.DataFrame(log.train_results.tolist(), columns=['train_loss', 'train_acc', 'train_speed', 'train_ppl'])
            except AssertionError:
                valid = pd.DataFrame(log.valid_results.tolist(), columns=['valid_loss', 'valid_acc', 'valid_speed', 'valid_ppl', 'valid_art', 'valid_ort'])
                train = pd.DataFrame(log.train_results.tolist(), columns=['train_loss', 'train_acc', 'train_speed', 'train_ppl', 'train_art', 'train_ort'])

        if hasattr(log, 'test_results'):
            try:
                test = pd.DataFrame(log.test_results.tolist(), columns=['test_loss', 'test_acc', 'test_speed'])
            except AssertionError:
                test = pd.DataFrame(log.test_results.tolist(), columns=['test_loss', 'test_acc', 'test_speed', 'test_ppl'])

            df = pd.concat([log.epoch, log.time, train, valid, test], axis=1)
        else:
            df = pd.concat([log.epoch, log.time, train, valid], axis=1)

        if upper_bound_epochs is not None:
            df = df[df['epoch'] <= upper_bound_epochs]

        if short_run_name not in logs:
            logs[short_run_name] = df
        else: #append!
            if not silent:
                print(f"fusing {run_name} with {short_run_name}")
            logs[short_run_name] = pd.concat([logs[short_run_name], df], axis=0)
    return logs, hyps


def make_label(name, split, hyps, display_list=None):
    shorthands = {'lr': "lr{hyp[lr]:.5f}",
                  'batch_size': "bs{hyp[batch_size]:3d}",
                  'train_subset':"ts{hyp[train_subset][1]:3d}",
                  'graph_state_dropout': "gsd{hyp[graph_state_dropout]:.1f}",
                  'output_dropout': "od{hyp[output_dropout]:.1f}",
                  'edge_weight_dropout': "ewd{hyp[edge_weight_dropout]:.1f}",
                  'position_embeddings': "pos{hyp[position_embeddings]:1d}",
                  'use_node_types': "ntyp{hyp[use_node_types]:1d}",
                  'use_edge_bias': "ebias{hyp[use_edge_bias]:1d}",
                  'msg_mean_aggregation': "mean{hyp[msg_mean_aggregation]:1d}",
                  'inst2vec_embeddings': "i2v:{hyp[inst2vec_embeddings][0]}",
                  'attn_bias': "attn_b:{hyp[attn_bias]:1d}",
                  'attn_num_heads': "attn_h:{hyp[attn_num_heads]:1d}",
                  'attn_dropout': "attn_d:{hyp[attn_dropout]:.1f}",
                  'tfmr_act': "tfmr_act:{hyp[tfmr_act]}",
                  'tfmr_dropout': "tfmr_d:{hyp[tfmr_dropout]}",
                  'tfmr_ff_sz': "tfmr_ff:{hyp[tfmr_ff_sz]}",
                  'layer_timesteps': "lt{hyp[layer_timesteps]}",
                  'gnn_layers': "gnn{hyp[gnn_layers]}",
                  'message_weight_sharing': "mshar{hyp[message_weight_sharing]}",
                  'update_weight_sharing': "ushar{hyp[update_weight_sharing]}",
                  'aux_use_better': "aux_better{hyp[aux_use_better]:1d}",
                 }

    hyp = hyps[name]

    # catch the no params file situation early
    if hyp is None:
        return name

    # set to display all _available_ hyperparams if display_list is none
    if display_list is None:
        display_list = [x for x in list(hyp.keys()) if x in shorthands]

    for k in display_list:
        if k not in hyp:
            display_list.remove(k)

    attrib = []
    for x in display_list:
        try:
            attrib.append(shorthands[x].format(hyp=hyp))
        except:
            print(x)
            print(hyp)
            attrib.append(shorthands[x].format(hyp=hyp))

    hyp_str = ' '.join(attrib)

    return hyp_str + '\n' + name


def annot_max(x, y, test_y, color='k', label=None, ax=None, invert_acc=True):
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    
    yargmax = np.argmax(y)
    xargmax = x[yargmax]
    xmax = x.max()
    ymax = y.max()
    test_y = test_y[np.argmax(np.array(y))]

    if invert_acc:  # display error % instead
        text = "val err {:.2f}%, test err {:.2f}% @ ep {:d}/{:d}".format((1-ymax)*100, (1-test_y)*100, int(xargmax), int(xmax))
    else:
        text = "val acc {:.4f}, test acc {:.4f} @ ep {:d}/{:d}".format(ymax, test_y, int(xargmax), int(xmax))
    if label:
        text = text + '\n' + label
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3",
                      fc='w',
                      ec=color,
                      lw=0.72)
    arrowprops = dict(arrowstyle="->",
                    #connectionstyle="angle,angleA=0,angleB=60",
                    color=color)
    kw = dict(xycoords='data',
              textcoords='data',
              #textcoords="axes fraction",
              arrowprops=arrowprops,
              bbox=bbox_props,
              ha="right",
              va="top")
    print(text)
    ax.annotate(text, xy=(xargmax, ymax), xytext=(xmax + np.random.uniform(-xmax/1.5,xmax/2) , max((ymax - 0.951)*24 + 0.7, 0.91 + np.random.uniform(-0.01, 0.01))) , **kw)  # xytext=(xmax,0.90 + (np.random.rand() - 0.5) / 5)


def plot_logs(logs, hyps, lower_ylim=0.85, display_list=None, legend_loc='best', plot_list='all', exclude_by_val_acc=False, filter_props={}):
    plt.figure(figsize=(24, 12))
    #cycle_colors=iter(plt.cm.hsv(np.linspace(0,0.97,len(logs)))) #jet / hsv / rainbow
    cycle_colors = cycle(iter(plt.rcParams['axes.prop_cycle'].by_key()['color']))
    _logs = {}
    for name, log in logs.items():
        take = True
        for prop, values in filter_props.items():
            if hyps[name][prop] not in values:
                take = False
        if take:
            _logs[name] = log
    logs = _logs
    
    for name, log in logs.items():
        if exclude_by_val_acc and log['valid_acc'].max() < 0.96:
            continue

        c = next(cycle_colors) #next(color)

        if plot_list == 'all' or 'valid_acc' in plot_list:
            if hasattr(log, 'valid_acc'):
                label = make_label(name, 'valid',hyps, display_list)
                plt.plot(log['epoch'].values, log['valid_acc'].values, ls='dashed', c=c, label=label.replace('\n', ' '))  # '_nolegend_',) # , 

        if plot_list == 'all' or 'train_acc' in plot_list:
            if hasattr(log, 'train_acc'):
                #label = make_label(name, 'train')
                plt.plot(log['epoch'].values, log['train_acc'].values, ls='dotted', c=c, label='train_acc')  # '_nolegend_') # make_label(name, 'train')

        if plot_list == 'all' or 'test_acc' in plot_list:
            if hasattr(log, 'test_acc'):
                label = make_label(name, 'test', hyps, display_list=display_list)
                plt.plot(log['epoch'].values, log['test_acc'].values, label=label.replace('\n', '  '), c=c)
                annot_max(x=log['epoch'].values, y=log['valid_acc'].values, test_y=log['test_acc'].values, color=c, label=label)


    plt.yticks(np.arange(0, 1, step=0.05))
    plt.ylim((lower_ylim, 1.0))
    plt.grid(which='major', axis='x', c='grey')
    plt.grid(which='minor', axis='y', c='silver', ls='--')
    plt.grid(which='major', axis='y', c='black')
    plt.minorticks_on()
    if legend_loc is not None:
        plt.legend(loc=legend_loc, prop={'family': 'monospace'})

    plt.show()
    
    # ~~~~~~~~~~~~ PPL PLOT ~~~~~~~~~~~~~~~~~~~~
    if plot_list == 'all' or any(['ppl' in x for x in plot_list]):
        plt.figure(figsize=(24, 12))
        cycle_colors = cycle(iter(plt.rcParams['axes.prop_cycle'].by_key()['color']))
        for name, log in logs.items():
            c = next(cycle_colors) #next(color)

            if plot_list == 'all' or 'valid_ppl' in plot_list:
                if hasattr(log, 'valid_ppl'):
                    label = make_label(name, 'valid_ppl', hyps, display_list=display_list)
                    plt.plot(log['epoch'].values, log['valid_ppl'].values, label=label.replace('\n','  ') ,c=c)
                if hasattr(log, 'train_ppl'):
                    label = make_label(name, 'train_ppl', hyps, display_list=display_list)
                    plt.plot(log['epoch'].values, log['train_ppl'].values, label=label.replace('\n','  ') , ls='dotted', c=c)

        plt.yticks(np.arange(1.0, 6.0, step=0.5))
        plt.ylim((1.0, 2.0))
        plt.grid(which='major', axis='x', c='grey')
        plt.grid(which='minor', axis='y', c='silver', ls='--')
        plt.grid(which='major', axis='y', c='black')
        plt.minorticks_on()
        if legend_loc is not None:
            plt.legend(loc=legend_loc, prop={'family': 'monospace'})

        plt.show()
