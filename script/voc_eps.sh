# NEED TO SET
DATASET_ROOT=../../dataset/VOC/VOCdevkit/VOC2012
WEIGHT_ROOT=./pretrained
SALIENCY_ROOT=./SALImages
GPU=0,1,2,3

# Default setting
SESSION="eps"
IMG_ROOT=${DATASET_ROOT}/JPEGImages
SAL_ROOT=${DATASET_ROOT}/${SALIENCY_ROOT}
BACKBONE=resnet38_eps
BASE_WEIGHT=${WEIGHT_ROOT}/ilsvrc-cls_rna-a1_cls1000_ep-0001.params


# train classification network with EPS
CUDA_VISIBLE_DEVICES=${GPU} python3 contrast_train.py \
  --use_wandb       \
  --session         ${SESSION} \
  --network         network.${BACKBONE} \
  --data_root       ${IMG_ROOT} \
  --saliency_root   ${SAL_ROOT} \
  --weights         ${BASE_WEIGHT} \
  --resize_size     256 512 \
  --crop_size       448 \
  --tau             0.4 \
  --max_iters       10000 \
  --iter_size       2 \
  --batch_size      8


DATA=train_aug # train / train_aug
TRAINED_WEIGHT=train_log/${SESSION}/checkpoint.pth
# 2. inference CAM (train/train_aug/val)
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
EVAL_DATA=train # train / val
GT_ROOT=${DATASET_ROOT}/SegmentationClassAug/

CUDA_VISIBLE_DEVICES=${GPU} python3 eval.py \
    --list          data/voc12/${EVAL_DATA}_id.txt \
    --predict_dir   train_log/${SESSION}/result/cam_npy/ \
    --gt_dir        ${GT_ROOT} \
    --comment       $SESSION \
    --logfile       train_log/${SESSION}/result/train.log \
    --max_th        40 \
    --type          npy \
    --curve


# 4. Generate Segmentation pseudo label
python pseudo_label_gen.py \
    --datalist          data/voc12/${DATA}_id.txt \
    --crf_pred          train_log/${SESSION}/result/crf_png/crf_5_8 \
    --label_save_dir    train_log/${SESSION}/result/crf_seg
