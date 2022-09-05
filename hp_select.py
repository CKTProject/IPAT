#!/usr/bin/env python3
#
# This is the script I use to tune the hyper-parameters automatically.
#
import subprocess

import hyperopt

import argparse
import os
import multiprocessing
import sys
import math

min_y = 0
min_c = None


def trial(hyperpm):
    global min_y, min_c
    # Plz set nbsz manually. Maybe a larger value if you have a large memory.
    cmd = 'python src/execute.py --embedding --node_classify --dimension 512 --hp --enhance'
    #cmd = 'CUDA_VISIBLE_DEVICES=5 ' + cmd
    for k in hyperpm:
        v = hyperpm[k]
        cmd += ' --' + k
        try:
            if int(v) == v:
                cmd += ' %d' % int(v)
            else:
                cmd += ' %g' % float('%.1e' % float(v))
        except ValueError:
            cmd += ' %s'%v
    try:
        print(cmd)
        val, tst = eval(subprocess.check_output(cmd, shell=True))
        #val, tst = 0.0, 0.0
    except subprocess.CalledProcessError:
        print('...')
        return {'loss': 0, 'status': hyperopt.STATUS_FAIL}
    print('one set of hyper val=%5.2f%% tst=%5.2f%% @ %s' % (val * 100, tst * 100, cmd))
    score = -val
    if score < min_y:
        min_y, min_c = score, cmd
    return {'loss': score, 'status': hyperopt.STATUS_OK}


#space = {'lr': hyperopt.hp.loguniform('lr', -8, 0),
#         'reg': hyperopt.hp.loguniform('reg', -10, 0),
#         'nlayer': hyperopt.hp.quniform('nlayer', 1, 6, 1),
#         'ncaps': 7,
#         'nhidden': hyperopt.hp.quniform('nhidden', 2, 32, 2),
#         'dropout': hyperopt.hp.uniform('dropout', 0, 1),
#         'routit': 6}
#hyperopt.fmin(trial, space, algo=hyperopt.tpe.suggest, max_evals=1000)
#print('>>>>>>>>>> val=%5.2f%% @ %s' % (-min_y * 100, min_c))



def run(model, dataset, reverse_type=1):

    share_space = {
            #'learning_rate':hyperopt.hp.loguniform('learning_rate', math.log(1e-5), math.log(1e-1)),
            'learning_rate':hyperopt.hp.choice('learning_rate', [0.01, 0.1]),
            #'ins_loss_w':hyperopt.hp.loguniform('ins_loss_w', math.log(1e-6), math.log(1)),
            'hinge_loss_w':hyperopt.hp.loguniform('hinge_loss_w', math.log(1e-6), math.log(1)),
            #'l2_r':hyperopt.hp.loguniform('l2_r', -10, -1),
            'negative_num':hyperopt.hp.quniform('negative_num', 5, 30, 5),
            'augment_num':hyperopt.hp.quniform('augment_num', 1, 30, 5),
            #'hdim':hyperopt.hp.quniform('hdim', 2, 32, 2),
            #'temperature':hyperopt.hp.loguniform('temperature', math.log(0.1), math.log(10)),
            'temperature':hyperopt.hp.choice('temperature', [0.1, 1.0, 10.]),
            'dropout':hyperopt.hp.uniform('dropout', 0, 1),
            'early_stop':hyperopt.hp.quniform('early_stop', 1, 10, 2),
            'neighbor_hop':hyperopt.hp.quniform('neighbor_hop', 1, 3, 1),
            'dataset':dataset,
            'model':model
    }
    id_space = {}

    if model in ['GraphCLR', 'GraphGaussianCLR', 'GraphSiameseCLR']:
        share_space.update({'aug_dgi_loss_w':hyperopt.hp.loguniform('aug_dgi_loss_w', math.log(1e-6), math.log(1)),})
        if model == 'GraphCLR':
            id_space['norm_loss_w'] = 0.
            #id_space = {
            #  'norm_loss_w':hyperopt.hp.loguniform('norm_loss_w', -12, -5),
        #}
        elif model == 'GraphGaussianCLR':
            #share_space['ins_loss_w'] = 0.
            pass

        elif model == 'GraphSiameseCLR':
            share_space['ins_loss_w'] = 0.
            share_space['temperature'] = 1.
            share_space['hinge_loss_w'] = 0.
            id_space = {
        'siamese_loss_w':hyperopt.hp.loguniform('siamese_loss_w', math.log(1e-6), math.log(1)),
        'siamese_pos_w':hyperopt.hp.quniform('siamese_pos_w', 1, 50, 5),
        }
    elif model == 'GAEAdvID':
        share_space.update({'aug_gae_loss_w':hyperopt.hp.loguniform('aug_gae_loss_w', math.log(1e-6), math.log(1))})
        share_space.update({'ins_loss_w':hyperopt.hp.loguniform('ins_loss_w', math.log(1e-6), math.log(1))})
        share_space.update({'norm_loss_w':hyperopt.hp.loguniform('norm_loss_w', math.log(1e-6), math.log(1))})
        id_space['augment_num'] = 1
        share_space['hinge_loss_w'] = 0.

    elif model in ['GAE', 'GAECLR', 'GAEGaussianCLR', 'GAESiameseCLR']:
        share_space.update({'aug_gae_loss_w':hyperopt.hp.loguniform('aug_gae_loss_w', math.log(1e-6), math.log(1))})
        if model == 'GAE':
            share_space['model'] = 'GAECLR'
            share_space['ins_loss_w'] = 0.
            share_space['temperature'] = 1.0
            share_space['aug_gae_loss_w'] = 0.
            share_space['hinge_loss_w'] = 0.
        elif model == 'GAECLR':
            pass
            #id_space['norm_loss_w'] = 0.
            #id_space = {
            #  'norm_loss_w':hyperopt.hp.loguniform('norm_loss_w', -12, -5),
        #}
        elif model == 'GAEGaussianCLR':
            #share_space['ins_loss_w'] = 0.
            share_space['hinge_loss_w'] = 0.
            pass

        elif model == 'GAESiameseCLR':
            share_space['ins_loss_w'] = 0.
            share_space['temperature'] = 1.
            share_space['hinge_loss_w'] = 0.
            id_space = {
        'siamese_loss_w':hyperopt.hp.loguniform('siamese_loss_w', math.log(1e-6), math.log(1)),
        'siamese_pos_w':hyperopt.hp.quniform('siamese_pos_w', 1, 50, 5),
        }
    elif model in ['Deepwalk', 'DeepwalkCLR', 'DeepwalkSiamese', 'DeepwalkGaussian']:
        share_space.update({
            #'number_of_walks':hyperopt.hp.quniform('number_of_walks', 1, 20, 2),
            #'walk_length':hyperopt.hp.quniform('walk_length', 10, 30, 5),
            #'window_size':hyperopt.hp.quniform('window_size', 1, 10, 1),
            'dw_neg_num':hyperopt.hp.quniform('dw_neg_num', 5, 20, 5),
            })
        if model == 'DeepwalkCLR':
            id_space['norm_loss_w'] = 0.
        elif model == 'DeepwalkSiamese':
            share_space['ins_loss_w'] = 0.
            share_space['temperature'] = 1.
            id_space = {
        'siamese_loss_w':hyperopt.hp.loguniform('siamese_loss_w', math.log(1e-6), math.log(1)),
        'siamese_pos_w':hyperopt.hp.quniform('siamese_pos_w', 1, 50, 5),
        }

    space = {**share_space, **id_space}
    algo = hyperopt.partial(hyperopt.tpe.suggest,n_startup_jobs=5)
    hyperopt.fmin(trial, space, algo=algo, max_evals=100)
    #hyperopt.fmin(trial, space, algo=hyperopt.tpe.suggest, max_evals=1000)
    #hyperopt.fmin(trial, space, algo=hyperopt.tpe.suggest, max_evals=300)
    print('>>>>>>>>>> best val=%5.2f%% @ %s' % (-min_y * 100, min_c))




if __name__ == '__main__':
    print('current process {0}'.format(os.getpid()))
    p = multiprocessing.Pool(processes=1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GraphCLR')
    parser.add_argument('--dataset', type=str, default='cora')


    args = parser.parse_args()
    run(args.model, args.dataset)
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All processes done!')

