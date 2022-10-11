# NEED TO SET
GPU=0,1,2,3
DATASET_ROOT=../../dataset/COCO
WEIGHT_ROOT=./pretrained
SAVE_ROOT=train_log
SESSION="coco_eps"

# Default setting
DATASET=coco
IMG_ROOT=${DATASET_ROOT}/train2014
SALIENCY_ROOT=${DATASET_ROOT}/SALImages
BACKBONE=resnet38_eps
BASE_WEIGHT=${WEIGHT_ROOT}/ilsvrc-cls_rna-a1_cls1000_ep-0001.params
TRAIN_LIST=data/${DATASET}/train.txt 
VAL_LIST=data/${DATASET}/val.txt

# 1. train classification network with EPS
CUDA_VISIBLE_DEVICES=${GPU} python3 contrast_train.py \
    --dataset           ${DATASET} \
    --train_list        ${TRAIN_LIST} \
    --val_list          ${VAL_LIST} \
    --session           ${SESSION} \
    --network           network.${BACKBONE} \
    --data_root         ${IMG_ROOT} \
    --saliency_root     ${SALIENCY_ROOT} \
    --weights           ${BASE_WEIGHT} \
    --resize_size       256 448 \
    --crop_size         321 \
    --tau               0.4 \
    --alpha             0.9 \
    --max_iters         256500 \
    --iter_size         1 \
    --batch_size        16


# 2. inference CAM
TRAINED_WEIGHT=${SAVE_ROOT}/${SESSION}/checkpoint_cls.pth
CUDA_VISIBLE_DEVICES=${GPU} python3 contrast_infer.py \
    --dataset               ${DATASET} \
    --infer_list            data/${DATASET}/train.txt \
    --img_root              ${IMG_ROOT} \
    --network               network.${BACKBONE} \
    --weights               ${TRAINED_WEIGHT} \
    --thr                   0.22 \
    --n_gpus                4 \
    --n_processes_per_gpu   1 1 1 1 \
    --cam_png               ${SAVE_ROOT}/${SESSION}/result/cam_png \
    --cam_npy               ${SAVE_ROOT}/${SESSION}/result/cam_npy \
    --crf                   ${SAVE_ROOT}/${SESSION}/result/crf_png \
    --crf_t                 5 \
    --crf_alpha             8

# 3. evaluate CAM
GT_ROOT=${DATASET_ROOT}/SegmentationClass/train2014/

CUDA_VISIBLE_DEVICES=${GPU} python3 eval.py \
    --dataset coco \
    --list data/${DATASET}/train.txt \
    --predict_dir ${SAVE_ROOT}/${SESSION}/result/cam_npy/ \
    --gt_dir ${GT_ROOT} \
    --comment $SESSION \
    --logfile ${SAVE_ROOT}/${SESSION}/result/train.txt \
    --max_th 30 \
    --type npy \
    --curve

# # 4. Generate Segmentation pseudo label
# python pseudo_label_gen.py \
#     --datalist data/voc12/${DATA}_id.txt \
#     --crf_pred train_log/${SESSION}/result/crf_png/crf_5_8 \
#     --label_save_dir train_log/${SESSION}/result/crf_seg
