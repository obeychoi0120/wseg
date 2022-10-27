import os
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import argparse

from data.dataset import get_categories

def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21, input_type='png', threshold=1.0, printlog=False):
    TP = []
    P = []
    T = []
    for i in range(num_cls):
        TP.append(multiprocessing.Value('i', 0, lock=True))
        P.append(multiprocessing.Value('i', 0, lock=True))
        T.append(multiprocessing.Value('i', 0, lock=True))
    
    categories = get_categories(num_cls)

    def compare(start, step, TP, P, T, input_type, threshold):
        for idx in range(start, len(name_list), step):
            name = name_list[idx]
            try:
                if input_type == 'png':
                    predict_file = os.path.join(predict_folder, '%s.png' % name)
                    predict = np.array(Image.open(predict_file))  # cv2.imread(predict_file)
                elif input_type == 'npy':
                    predict_file = os.path.join(predict_folder, '%s.npy' % name)
                    predict_dict = np.load(predict_file, allow_pickle=True).item()
                    h, w = list(predict_dict.values())[0].shape
                    tensor = np.zeros((num_cls, h, w), np.float32)

                    for key in predict_dict.keys():
                        tensor[key+1] = predict_dict[key] ### replace key+1 to key when crf input ###
                    tensor[0, :, :] = threshold ### discard(comment) when crf input ###
                    predict = np.argmax(tensor, axis=0).astype(np.uint8)

                gt_file = os.path.join(gt_folder, '%s.png' % name)
                gt = np.array(Image.open(gt_file))
                cal = gt < 255
                mask = (predict == gt) * cal

                for i in range(num_cls):
                    P[i].acquire()
                    P[i].value += np.sum((predict == i) * cal)
                    P[i].release()
                    T[i].acquire()
                    T[i].value += np.sum((gt == i) * cal)
                    T[i].release()
                    TP[i].acquire()
                    TP[i].value += np.sum((gt == i) * mask)
                    TP[i].release()
            except Exception as e:
                print(e)

    p_list = []
    for i in range(8):
        p = multiprocessing.Process(target=compare, args=(i, 8, TP, P, T, input_type, threshold))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = []
    Precision = []
    Recall = []
    for i in range(num_cls):
        IoU.append(TP[i].value / (T[i].value + P[i].value - TP[i].value + 1e-10))
        T_TP.append(T[i].value / (TP[i].value + 1e-10))
        P_TP.append(P[i].value / (TP[i].value + 1e-10))
        FP_ALL.append((P[i].value - TP[i].value) / (T[i].value + P[i].value - TP[i].value + 1e-10))
        FN_ALL.append((T[i].value - TP[i].value) / (T[i].value + P[i].value - TP[i].value + 1e-10))
        Precision.append(TP[i].value / (P[i].value + 1e-10))
        Recall.append(TP[i].value / (T[i].value + 1e-10))
        
    loglist = {}
    for i in range(num_cls):
        loglist[categories[i]] = IoU[i] * 100
        loglist[categories[i]+'_P'] = Precision[i] * 100
        loglist[categories[i]+'_R'] = Recall[i] * 100

    miou = np.mean(np.array(IoU))
    precision = np.mean(np.array(Precision))
    recall = np.mean(np.array(Recall))
    
    loglist['mIoU'] = miou * 100
    loglist['Precision'] = precision * 100
    loglist['Recall'] = recall * 100
    if printlog:
        for i in range(num_cls):
            if i % 2 != 1:
                print('%11s:%7.3f%%' % (categories[i], IoU[i] * 100), end='\t')
            else:
                print('%11s:%7.3f%%' % (categories[i], IoU[i] * 100))
        print('\n======================================================')
        print('%11s:%7.3f%%' % ('mIoU', miou * 100))
    return loglist


def writedict(file, dictionary):
    s = ''
    for key in dictionary.keys():
        sub = '%s:%s  ' % (key, dictionary[key])
        s += sub
        s += '\n'
    file.write(s)


def writelog(filepath, metric, comment):
    filepath = filepath
    logfile = open(filepath, 'a')
    import time
    logfile.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logfile.write('\t%s\n' % comment)
    writedict(logfile, metric)
    logfile.write('=====================================\n')
    logfile.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='voc12', choices=['voc12', 'coco'], type=str)
    parser.add_argument("--list", default='./VOC2012/ImageSets/Segmentation/train.txt', type=str)
    parser.add_argument("--predict_dir", default='./out_rw', type=str)
    parser.add_argument("--gt_dir", default='./VOC2012/SegmentationClass', type=str)
    parser.add_argument('--logfile', default='./evallog.txt', type=str)
    parser.add_argument('--comment', required=True, type=str)
    parser.add_argument('--type', default='png', choices=['npy', 'png'], type=str)
    parser.add_argument('--t', default=None, type=float)
    parser.add_argument('--curve', action='store_true')
    parser.add_argument('--max_th', default=60, type=int)
    args = parser.parse_args()

    if args.dataset == 'voc12':
        args.num_samples = 21
    elif args.dataset == 'coco':
        args.num_samples = 81

    if args.type == 'npy':
        assert args.t is not None or args.curve
    df = pd.read_csv(args.list, names=['filename'])
    name_list = df['filename'].values
    if not args.curve:
        loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, args.num_samples, args.type, args.t, printlog=True)
        writelog(args.logfile, loglist, args.comment)
    else:
        l = []
        p = []
        r = []
        for i in range(args.max_th):
            t = i / 100.0
            loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, args.num_samples, args.type, t)
            l.append(loglist['mIoU'])
            p.append(loglist['Precision'])
            r.append(loglist['Recall'])
            print('%d/%d background score: %.3f\tmIoU: %.3f%%\tPrecision: %.3f%%\tRecall: %.3f%%' % 
                (i, args.max_th, t, loglist['mIoU'], loglist['Precision'], loglist['Recall']))
        writelog(args.logfile, {'mIoU': l, 'Precision': p, 'Recall': r}, args.comment)
