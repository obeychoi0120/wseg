import os 
import glob
import argparse
import random

# USAGE: python make_split.py --train_list coco/train.txt --lb_ratio 0.03125 --out_dir coco/split/1_8/ --seed 0
# Randomly Sample from train list
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", default="voc12/train_aug_id.txt", type=str)
    parser.add_argument("--lb_ratio", default=0.25, type=float) # 0.25, 0.125, 0.0625, 0.03125
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--out_dir", default="voc12/split/1_4/", type=str)
    args = parser.parse_args()

    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    with open(args.train_list, 'r') as f:
        train_list = f.read().strip().split('\n')
    
    random.shuffle(train_list)

    # split train list into lb and ulb
    n_labels = int(len(train_list) * args.lb_ratio)
    lb_list = sorted(train_list[:n_labels])
    ulb_list = sorted(train_list[n_labels:])

    lb_list = '\n'.join(list(lb_list))
    ulb_list = '\n'.join(list(ulb_list))

    # Write lb and ulb files
    lb_file = os.path.join(args.out_dir, f'lb_train_{args.seed}.txt')
    ulb_file = os.path.join(args.out_dir, f'ulb_train_{args.seed}.txt')

    with open(lb_file, 'w') as f:
        f.write(lb_list)
    with open(ulb_file, 'w') as f:
        f.write(ulb_list)