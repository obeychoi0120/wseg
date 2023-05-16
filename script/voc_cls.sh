# NEED TO SET
DATASET_ROOT=../data/VOCdevkit/VOC2012/
WEIGHT_ROOT=pretrained
SALIENCY_ROOT=./SALImages

GPU=2


# Default setting
IMG_ROOT=${DATASET_ROOT}/JPEGImages
SAL_ROOT=${DATASET_ROOT}/${SALIENCY_ROOT}
BACKBONE=resnet38_cls
SESSION="0423/P_cls_new_t0.95_s7_163"
BASE_WEIGHT=${WEIGHT_ROOT}/ilsvrc-cls_rna-a1_cls1000_ep-0001.params
GT_ROOT=${DATASET_ROOT}/SegmentationClassAug/

# # 1. train classification network
# CUDA_VISIBLE_DEVICES=${GPU} python3 contrast_train.py \
#     --mode              ssl    \
#     --session           ${SESSION}  \
#     --network           network.${BACKBONE} \
#     --data_root         ${IMG_ROOT} \
#     --weights           ${BASE_WEIGHT} \
#     --crop_size         448     \
#     --tau               0.4     \
#     --max_iters         10000   \
#     --iter_size         2       \
#     --batch_size        8       \
#     --strong_crop_size  336     \
#     --val_times 	    40      \
#     --p_cutoff          0.95    \
#     --use_ema                   \
#     --use_cutmix                \
#     --use_wandb      

DATA=train # train / train_aug
TRAINED_WEIGHT=train_log/${SESSION}/checkpoint.pth

# 2. inference CAM
CUDA_VISIBLE_DEVICES=${GPU} python3 contrast_infer.py \
    --infer_list            data/voc12/${DATA}_id.txt \
    --img_root              ${IMG_ROOT} \
    --network               network.${BACKBONE} \
    --weights               ${TRAINED_WEIGHT}   \
    --thr                   0.20 \
    --n_gpus                1 \
    --n_processes_per_gpu   1 \
    --cam_png               train_log/${SESSION}/result/cam_png \
    --cam_npy               train_log/${SESSION}/result/cam_npy \
    --crf                   train_log/${SESSION}/result/crf_npy \
    --crf_t                 5 \
    --crf_alpha             8 \

GT_ROOT=${DATASET_ROOT}/SegmentationClassAug/
EVAL_DATA=train

# 3. evaluate CAM
CUDA_VISIBLE_DEVICES=${GPU} python3 eval.py \
    --list            data/voc12/${EVAL_DATA}_id.txt \
    --predict_dir     train_log/${SESSION}/result/cam_npy \
    --gt_dir          ${GT_ROOT} \
    --comment         ${SESSION} \
    --logfile         train_log/${SESSION}/result/train.log \
    --type            npy \
    --max_th          100 \
    --curve               
    # Use curve when type=npy
    # Change predict_dir cam_png|cam_npy|crf_png/crf_5_8|crf_seg
    
# 4. Generate Segmentation pseudo label
python pseudo_label_gen.py \
    --datalist          data/voc12/${EVAL_DATA}_id.txt \
    --crf_pred          train_log/${SESSION}/result/crf_npy/crf_5_24 \
    --label_save_dir    train_log/${SESSION}/result/crf_seg
