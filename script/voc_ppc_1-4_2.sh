# NEED TO SET
DATASET_ROOT=../data/VOCdevkit/VOC2012
WEIGHT_ROOT=../pretrained
SALIENCY_ROOT=SALImages
DATASET='voc12'
GPU=0,1,2,3

# Default setting
IMG_ROOT=${DATASET_ROOT}/JPEGImages
SAL_ROOT=${DATASET_ROOT}/${SALIENCY_ROOT}
BACKBONE=resnet38_contrast
SESSION="P_SSL1-4_cutoff0.99-0.8"
BASE_WEIGHT=${WEIGHT_ROOT}/ilsvrc-cls_rna-a1_cls1000_ep-0001.params

# for SSL
SPLIT="1_4"
SPLIT_NUM="0"
LB_DATA_LIST=data/${DATASET}/split/${SPLIT}/lb_train_${SPLIT_NUM}.txt ############
ULB_DATA_LIST=data/${DATASET}/split/${SPLIT}/ulb_train_${SPLIT_NUM}.txt #########



# echo 'Image root : ', $IMG_ROOT
# echo 'Saliency root : ', $SAL_ROOT

# train classification network with Contrastive Learning
CUDA_VISIBLE_DEVICES=${GPU} python3 contrast_train.py \
    --ssl               \
    --train_list        ${LB_DATA_LIST} \
    --train_ulb_list    ${ULB_DATA_LIST} \
    --p_cutoff 0.99             \
    --min_p_cutoff 0.8          \
    --session           ${SESSION} \
    --use_wandb         \
    --network           network.${BACKBONE} \
    --data_root         ${IMG_ROOT} \
    --saliency_root     ${SAL_ROOT} \
    --weights           ${BASE_WEIGHT} \
    --crop_size         448 \
    --tau               0.4 \
    --max_iters         10000 \
    --iter_size         2 \
    --batch_size        8



# 2. inference CAM
DATA=train_aug # train / train_aug
TRAINED_WEIGHT=train_log/${SESSION}/checkpoint.pth

# Labeled
CUDA_VISIBLE_DEVICES=${GPU} python3 contrast_infer.py \
    # --infer_list            data/voc12/${DATA}_id.txt \
    --infer_list            ${LB_DATA_LIST} \
    --img_root              ${IMG_ROOT} \
    --network               network.${BACKBONE} \
    --weights               ${TRAINED_WEIGHT} \
    --thr                   0.22 \
    --n_gpus                4 \
    --n_processes_per_gpu   1 1 1 1 \
    --cam_png               train_log/${SESSION}/result/cam_png \
    --cam_npy               train_log/${SESSION}/result/cam_npy \
    --crf                   train_log/${SESSION}/result/crf_png \
    --crf_t                 5 \
    --crf_alpha             8

# Unlabeled
CUDA_VISIBLE_DEVICES=${GPU} python3 contrast_infer.py \
    --infer_list            ${ULB_DATA_LIST} \
    --img_root              ${IMG_ROOT} \
    --network               network.${BACKBONE} \
    --weights               ${TRAINED_WEIGHT} \
    --thr                   0.22 \
    --n_gpus                4 \
    --n_processes_per_gpu   1 1 1 1 \
    --cam_png               train_log/${SESSION}/result/cam_png \
    --cam_npy               train_log/${SESSION}/result/cam_npy \
    --is_unlabeled          \
    --pl_method             all \
    --crf                   train_log/${SESSION}/result/crf_png\
    --crf_t                 5 \
    --crf_alpha             8 \

# 3. evaluate CAM
GT_ROOT=${DATASET_ROOT}/SegmentationClassAug/

CUDA_VISIBLE_DEVICES=${GPU} python3 eval.py \
    --list          data/voc12/train_id.txt \
    --predict_dir   train_log/${SESSION}/result/cam_npy/ \
    --gt_dir        ${GT_ROOT} \
    --comment       $SESSION \
    --logfile       train_log/${SESSION}/result/train.log \
    --max_th        30 \
    --type          npy \
    --curve 
    # Use curve when type=npy
    # Change predict_dir cam_png|cam_npy|crf_png/crf_5_8|crf_seg

# # 4. Generate Segmentation pseudo label
python pseudo_label_gen.py \
    --datalist          data/voc12/${DATA}_id.txt \
    --crf_pred          train_log/${SESSION}/result/crf_png/crf_5_8 \
    --label_save_dir    train_log/${SESSION}/result/crf_seg
