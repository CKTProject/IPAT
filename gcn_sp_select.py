#!/usr/bin/env python3
#
# This is the script I use to tune the hyper-parameters automatically.
#
import subprocess
import argparse


if __name__ == '__main__':
    parser.add_argument("--gpuid", type=int, default=2, help='index of gpu')
    args = parser.parse_args()

    dropout_list = [0., 0.3, 0.5, 0.8]
    l2_r_list = [0., 1e-5, 1e-3]
    early_stop_list = [5, 10, 20]
    hidden_dim_list = [16, 32, 64, 128, 256, 512]
    for dropout in dropout_list:
        for l2_r in l2_r_list:
            for early_stop in early_stop_list:
                for hidden_dim in hidden_dim_list:
                    cmd = 'python src/douban_sp_execute.py --embedding --node_classify --gpuid %d --hidden_dim %d --hp --dropout %f --l2_r %f --early_stop %d'%(args.gpuid, hidden_dim, dropout, l2_r, early_stop)
                    print("***********START A NEW CMD************")
                    print(cmd)
                    try:
                        print(cmd)
                        val, tst = eval(subprocess.check_output(cmd, shell=True))
                    except subprocess.CalledProcessError:
                        print('CMD ERROR!!!!!!!')
                    print("***********END CMD************")


