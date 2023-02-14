# NEED TO SET
DATASET_ROOT=../data/VOCdevkit/VOC2012/
WEIGHT_ROOT=pretrained
SALIENCY_ROOT=./SALImages

GPU=0,1


# Default setting
IMG_ROOT=${DATASET_ROOT}/JPEGImages
SAL_ROOT=${DATASET_ROOT}/${SALIENCY_ROOT}
BACKBONE=resnet38_cls
SESSION="cls+irn+crf_cut0.95_153"
BASE_WEIGHT=${WEIGHT_ROOT}/ilsvrc-cls_rna-a1_cls1000_ep-0001.params
GT_ROOT=${DATASET_ROOT}/SegmentationClassAug/

# 1. train classification network
CUDA_VISIBLE_DEVICES=${GPU} python3 contrast_train.py \
    --v2                      \
    --session       ${SESSION} \
    --network       network.${BACKBONE} \
    --data_root     ${IMG_ROOT} \
    --weights       ${BASE_WEIGHT} \
    --crop_size     448 \
    --max_iters     10000 \
    --iter_size     1 \
    --batch_size    8 \
    --p_cutoff      0.95 \
    --val_times     50  \
    --use_wandb         \
    --val_only        \


DATA=train # train / train_aug
TRAINED_WEIGHT=train_log/${SESSION}/checkpoint.pth

# 2. inference CAM
CUDA_VISIBLE_DEVICES=${GPU} python3 contrast_infer.py \
    --infer_list            data/voc12/${DATA}_id.txt \
    --img_root              ${IMG_ROOT} \
    --network               network.${BACKBONE} \
    --weights               ${TRAINED_WEIGHT}   \
    --thr                   0.20 \
    --n_gpus                2 \
    --n_processes_per_gpu   1 1 \
    --cam_png               train_log/${SESSION}/result/cam_png \
    --cam_npy               train_log/${SESSION}/result/cam_npy \
    --crf                   train_log/${SESSION}/result/crf_npy \
    --crf_t                 5 \
    --crf_alpha             4 24 \

# 3. evaluate CAM
CUDA_VISIBLE_DEVICES=${GPU} python3 eval.py \
    --list            data/voc12/${DATA}_id.txt \
    --predict_dir     train_log/${SESSION}/result/cam_npy \
    --gt_dir          ${GT_ROOT} \
    --comment         ${SESSION} \
    --logfile         train_log/${SESSION}/result/eval_cam.log \
    --type            npy \
    --max_th          100 \
    --curve               
    # Use curve when type=npy
    # Change predict_dir cam_png|cam_npy|crf_png/crf_5_8|crf_seg


# 3.5 Train IRN

Prepare
CUDA_VISIBLE_DEVICES=${GPU} python aff_prepare.py \
    --infer_list            data/voc12/${DATA}_id.txt \
    --voc12_root            ${DATASET_ROOT} \
    --cam_dir               train_log/${SESSION}/result/cam_npy \
    --out_crf               train_log/${SESSION}/result/crf_png \

# AffinityNet train
CUDA_VISIBLE_DEVICES=${GPU} python aff_train.py \
    --train_list        data/voc12/${DATA}.txt \
    --weights           ${BASE_WEIGHT}  \
    --voc12_root        ${DATASET_ROOT} \
    --la_crf_dir        train_log/${SESSION}/result/crf_npy/crf_5_4  \
    --ha_crf_dir        train_log/${SESSION}/result/crf_npy/crf_5_24 \
    --session_name      ${SESSION}

# Randow walk propagation & Evaluation 
# train.txt / train_aug.txt
# type: png
CUDA_VISIBLE_DEVICES=${GPU} python aff_infer.py \
    --infer_list        data/voc12/${DATA}.txt \
    --weights           train_log/${SESSION}/result/aff_checkpoint.pth \
    --voc12_root        ${DATASET_ROOT} \
    --cam_dir           train_log/${SESSION}/result/cam_npy \
    --out_rw            train_log/${SESSION}/result/cam_rw_png

# evaluate IRN CAM
CUDA_VISIBLE_DEVICES=${GPU} python eval.py \
    --list            data/voc12/${DATA}_id.txt \
    --predict_dir     train_log/${SESSION}/result/cam_rw_png \
    --gt_dir          ${GT_ROOT} \
    --comment         ${SESSION} \
    --logfile         train_log/${SESSION}/result/eval_irn_cam.log \
    --type            png \
    --max_th          100 \
    # --curve

# 4. Generate Segmentation pseudo label
python pseudo_label_gen.py \
    --datalist          data/voc12/${DATA}_id.txt \
    --crf_pred          train_log/${SESSION}/result/crf_npy/crf_5_24 \
    --label_save_dir    train_log/${SESSION}/result/crf_seg
