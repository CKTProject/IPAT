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
    cmd = 'python src/douban_execute.py --embedding --node_classify --dimension 512 --hp --enhance --batch_size 500'
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



def run(model, dataset, gpuid):

    if model == 'AdvR':
        share_space = {'learning_rate':hyperopt.hp.choice('learning_rate', [0.01, 0.1, 1.0]),
            'aug_gae_loss_w':hyperopt.hp.loguniform('aug_gae_loss_w', math.log(1e-6), math.log(1)),
            'dropout':hyperopt.hp.uniform('dropout', 0, 1),
            'dataset':dataset,
            'model':model,
            'gpuid':gpuid
            }
    else:
        share_space = {'learning_rate':hyperopt.hp.choice('learning_rate', [0.01, 0.1, 1.0]),
            'aug_gae_loss_w':hyperopt.hp.loguniform('aug_gae_loss_w', math.log(1e-6), math.log(1)),
            'ins_loss_w':hyperopt.hp.loguniform('ins_loss_w', math.log(1e-6), math.log(1)),
            'norm_loss_w':hyperopt.hp.loguniform('norm_loss_w', math.log(1e-6), math.log(1)),
            #'l2_r':hyperopt.hp.loguniform('l2_r', -10, -1),
            'negative_num':hyperopt.hp.choice('negative_num', [5, 10, 20]),
            'augment_num':1,
            'temperature':hyperopt.hp.choice('temperature', [0.01, 0.1, 1.0, 10.]),
            'dropout':hyperopt.hp.uniform('dropout', 0, 1),
            'early_stop':hyperopt.hp.quniform('early_stop', 1, 10, 2),
            'neighbor_hop':hyperopt.hp.quniform('neighbor_hop', 1, 3, 1),
            'dataset':dataset,
            'model':model,
            'gpuid':gpuid
    }

    id_space = {}
    space = {**share_space, **id_space}
    algo = hyperopt.partial(hyperopt.tpe.suggest,n_startup_jobs=5)
    hyperopt.fmin(trial, space, algo=algo, max_evals=50)
    #hyperopt.fmin(trial, space, algo=hyperopt.tpe.suggest, max_evals=1000)
    #hyperopt.fmin(trial, space, algo=hyperopt.tpe.suggest, max_evals=300)
    print('>>>>>>>>>> best val=%5.2f%% @ %s' % (-min_y * 100, min_c))




if __name__ == '__main__':
    print('current process {0}'.format(os.getpid()))
    p = multiprocessing.Pool(processes=5)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LearnR')
    parser.add_argument('--dataset', type=str, default='douban')
    parser.add_argument("--gpuid", type=int, default=2, help='index of gpu')


    args = parser.parse_args()
    run(args.model, args.dataset, args.gpuid)
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All processes done!')

