import os 
import glob

if __name__=='__main__':
    file_path = 'split/lb*.txt'
    file_list = glob.glob(file_path)
    
    train_aug_file = 'train_aug_id.txt'
    with open(train_aug_file, 'r') as f:
        train_aug_list = set(f.read().strip().split('\n'))

    for file in file_list:
        with open(file, 'r') as f:
            lb_list = set(f.read().strip().split('\n'))
        ulb_list = train_aug_list - lb_list

        ulb_file = os.path.join('split', 'u'+os.path.basename(file))
        with open(ulb_file, 'w') as f:
            data = '\n'.join(list(ulb_list))
            f.write(data)
        
        print(ulb_file, 'Done.')