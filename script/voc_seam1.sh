# NEED TO SET
DATASET_ROOT=../data/VOCdevkit/VOC2012/
WEIGHT_ROOT=pretrained
SALIENCY_ROOT=./SALImages

GPU=1

# Default setting
SESSION="0515/P+_seam_n3k2_s7_163"
# SESSION="test"
IMG_ROOT=${DATASET_ROOT}/JPEGImages
BACKBONE=resnet38_seam
BASE_WEIGHT=${WEIGHT_ROOT}/ilsvrc-cls_rna-a1_cls1000_ep-0001.params

# 1. train classification network with Contrastive Learning
CUDA_VISIBLE_DEVICES=${GPU} python3 contrast_train.py \
    --mode              ssl    \
    --session           ${SESSION}  \
    --network           network.${BACKBONE} \
    --data_root         ${IMG_ROOT} \
    --weights           ${BASE_WEIGHT} \
    --crop_size         448     \
    --tau               0.4     \
    --max_iters         10000   \
    --iter_size         2       \
    --batch_size        8       \
    --val_times 	    40      \
    --p_cutoff          0.95    \
    --use_ema                   \
    --use_cutmix                \
    --patch_k           2       \
    --n_strong_augs     3       \
    --seed              7       \
    --use_wandb                 \
    # --use_geom_augs             \
    
      
DATA=train # train / train_aug
TRAINED_WEIGHT=train_log/${SESSION}/checkpoint.pth

# 2. inference CAM
CUDA_VISIBLE_DEVICES=${GPU} python3 contrast_infer.py \
    --infer_list            data/voc12/${DATA}_id.txt \
    --img_root              ${IMG_ROOT} \
    --network               network.${BACKBONE} \
    --weights               ${TRAINED_WEIGHT} \
    --thr                   0.22 \
    --n_gpus                1 \
    --n_processes_per_gpu   1 \
    --cam_png               train_log/${SESSION}/result/cam_png \
    --cam_npy               train_log/${SESSION}/result/cam_npy \
    --crf                   train_log/${SESSION}/result/crf_png \
    --crf_t                 5 \
    --crf_alpha             8 \

# 3. evaluate CAM (train set에 대해)
GT_ROOT=${DATASET_ROOT}/SegmentationClassAug/
CUDA_VISIBLE_DEVICES=${GPU} python3 eval.py \
    --list          data/voc12/train_id.txt \
    --predict_dir   train_log/${SESSION}/result/cam_npy/ \
    --gt_dir        ${GT_ROOT} \
    --comment       $SESSION \
    --logfile       train_log/${SESSION}/result/train.log \
    --max_th        50 \
    --type          npy \
    --curve 
    # Use curve when type=npy
    # Change predict_dir cam_png|cam_npy|crf_png/crf_5_8|crf_seg

# 4. Generate Segmentation pseudo label
python pseudo_label_gen.py \
    --datalist          data/voc12/${DATA}_id.txt \
    --crf_pred          train_log/${SESSION}/result/crf_png/crf_5_8 \
    --label_save_dir    train_log/${SESSION}/result/crf_seg