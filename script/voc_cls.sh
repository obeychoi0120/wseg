# NEED TO SET
DATASET_ROOT=../data/VOCdevkit/VOC2012/
WEIGHT_ROOT=pretrained
SALIENCY_ROOT=./SALImages
GPU=0,1,2,3


# Default setting
IMG_ROOT=${DATASET_ROOT}/JPEGImages
BACKBONE=resnet38_cls
SESSION="P_cls+irn+crf_cut0.95_157"
BASE_WEIGHT=${WEIGHT_ROOT}/ilsvrc-cls_rna-a1_cls1000_ep-0001.params


# 1. train classification network
CUDA_VISIBLE_DEVICES=${GPU} python3 contrast_train.py \
    --ssl                      \
    --session       ${SESSION} \
    --network       network.${BACKBONE} \
    --data_root     ${IMG_ROOT} \
    --weights       ${BASE_WEIGHT} \
    --crop_size     448 \
    --tau           0.4 \
    --max_iters     10000 \
    --iter_size     1    \
    --batch_size    8    \
    --p_cutoff      0.95 \
    --val_times     50   \
    --use_wandb          \



# 2. inference CAM
DATA=train # train / train_aug
TRAINED_WEIGHT=train_log/${SESSION}/checkpoint.pth
# 2. inference CAM
CUDA_VISIBLE_DEVICES=${GPU} python3 contrast_infer.py \
    --infer_list            data/voc12/${DATA}_id.txt \
    --img_root              ${IMG_ROOT} \
    --network               network.${BACKBONE} \
    --weights               ${TRAINED_WEIGHT} \
    --thr                   0.20 \
    --n_gpus                4 \
    --n_processes_per_gpu   1 1 1 1 \
    --cam_png               train_log/${SESSION}/result/cam_png \
    --cam_npy               train_log/${SESSION}/result/cam_npy \
    --crf                   train_log/${SESSION}/result/crf_png \
    --crf_t                 5 \
    --crf_alpha             8


# 3. evaluate CAM
GT_ROOT=${DATASET_ROOT}/SegmentationClassAug/
CUDA_VISIBLE_DEVICES=${GPU} python3 eval.py \
    --list          data/voc12/${DATA}_id.txt \
    --predict_dir   train_log/${SESSION}/result/cam_npy/ \
    --gt_dir        ${GT_ROOT} \
    --comment       $SESSION \
    --logfile       train_log/${SESSION}/result/train.txt \
    --type          npy \
    --max_th        100 \
    --curve
    # Use curve when type=npy
    # Change predict_dir cam_png|cam_npy|crf_png/crf_5_8|crf_seg

#####TODO######


# 3.5 Train IRN
# CUDA_VISIBLE_DEVICES=${GPU} python aff_prepare.py \
#   --voc12_root VOC2012 \
#   --cam_dir train_log/${SESSION}/result/cam_npy \
#   --out_crf $your_crf_alpha_dir 





# # 4. Generate Segmentation pseudo label
# python pseudo_label_gen.py \
#     --datalist data/voc12/${DATA}_id.txt \
#     --crf_pred train_log/${SESSION}/result/crf_png/crf_5_8 \
#     --label_save_dir train_log/${SESSION}/result/crf_seg
