# NEED TO SET
DATASET_ROOT=../data/VOCdevkit/VOC2012/
WEIGHT_ROOT=pretrained
SALIENCY_ROOT=./SALImages
GT_ROOT=${DATASET_ROOT}/SegmentationClassAug/

GPU=0

# Default setting
IMG_ROOT=${DATASET_ROOT}/JPEGImages
SAL_ROOT=${DATASET_ROOT}/${SALIENCY_ROOT}
BACKBONE=resnet38_contrast
# SESSION="0724/ppc+eps_20k_s7_154"
SESSION="test"
BASE_WEIGHT=${WEIGHT_ROOT}/ilsvrc-cls_rna-a1_cls1000_ep-0001.params

# train classification network with Contrastive Learning
CUDA_VISIBLE_DEVICES=${GPU} python3 contrast_train.py \
    --mode              base    \
    --session           ${SESSION} \
    --network           network.${BACKBONE} \
    --data_root         ${IMG_ROOT} \
    --saliency_root     ${SAL_ROOT} \
    --weights           ${BASE_WEIGHT} \
    --num_workers       8       \
    --crop_size         448     \
    --tau               0.4     \
    --max_iters         20000   \
    --val_freq          500     \
    --iter_size         2       \
    --batch_size        8       \
    --seed              7       \
    # --use_wandb                 \
    # --n_strong_augs     5       \
    # --p_cutoff          0.95    \
    # --use_ema                   \
    # --bdry                      \
    # --bdry_size         3       \
    # --bdry_lambda       1.0     \
    # --use_cutmix                \
    # --patch_k           2       \
    # --recon_lambda      0.1     \
    # --use_geom_augs             \

TRAINED_WEIGHT=train_log/${SESSION}/checkpoint.pth

# # inference CAM (train_aug)
# echo "trainaug set Inference"
# INFER_DATA=train_aug # train / train_aug
# CUDA_VISIBLE_DEVICES=${GPU} python3 contrast_infer.py \
#     --infer_list            data/voc12/${INFER_DATA}_id.txt \
#     --img_root              ${IMG_ROOT} \
#     --network               network.${BACKBONE} \
#     --weights               ${TRAINED_WEIGHT} \
#     --thr                   0.22 \
#     --n_gpus                1    \
#     --n_processes_per_gpu   1    \
#     --cam_png               train_log/${SESSION}/result/${INFER_DATA}/cam_png \
#     --cam_npy               train_log/${SESSION}/result/${INFER_DATA}/cam_npy \
#     --crf                   train_log/${SESSION}/result/${INFER_DATA}/crf_npy \
#     --crf_t                 5   \
#     --crf_alpha             4 8 24 \

# # inference CAM (val)
# echo "val set Inference"
# INFER_DATA=val
# CUDA_VISIBLE_DEVICES=${GPU} python3 contrast_infer.py \
#     --infer_list            data/voc12/${INFER_DATA}_id.txt \
#     --img_root              ${IMG_ROOT} \
#     --network               network.${BACKBONE} \
#     --weights               ${TRAINED_WEIGHT} \
#     --thr                   0.22    \
#     --n_gpus                1       \
#     --n_processes_per_gpu   1       \
#     --cam_png               train_log/${SESSION}/result/${INFER_DATA}/cam_png \
#     --cam_npy               train_log/${SESSION}/result/${INFER_DATA}/cam_npy \
#     --crf                   train_log/${SESSION}/result/${INFER_DATA}/crf_npy \
#     --crf_t                 5   \
#     --crf_alpha             4 8 24 \


# # evaluate CAM (train)
# echo "train CAM evaluation"
# EVAL_DATA=train
# CUDA_VISIBLE_DEVICES=${GPU} python3 eval.py \
#     --list          data/voc12/${EVAL_DATA}_id.txt \
#     --predict_dir   train_log/${SESSION}/result/train_aug/cam_npy \
#     --gt_dir        ${GT_ROOT} \
#     --comment       ${SESSION} \
#     --logfile       train_log/${SESSION}/result/${EVAL_DATA}/${EVAL_DATA}.log \
#     --max_th        50 \
#     --type          npy \
#     --curve 
#     # Use curve when type=npy
#     # Change predict_dir cam_png|cam_npy|crf_png/crf_5_8|crf_seg

# # Generate Segmentation pseudo label(train)
# EVAL_DATA=train
# python pseudo_label_gen.py \
#     --datalist          data/voc12/${EVAL_DATA}_id.txt \
#     --crf_pred          train_log/${SESSION}/result/${EVAL_DATA}_aug/crf_npy/crf_5_8 \
#     --label_save_dir    train_log/${SESSION}/result/${EVAL_DATA}/crf_seg

# # evaluate CRF (train)
# echo "train CRF evaluation"
# EVAL_DATA=train
# CUDA_VISIBLE_DEVICES=${GPU} python3 eval.py \
#     --list          data/voc12/${EVAL_DATA}_id.txt \
#     --predict_dir   train_log/${SESSION}/result/${EVAL_DATA}/crf_seg \
#     --gt_dir        ${GT_ROOT} \
#     --comment       ${SESSION} \
#     --logfile       train_log/${SESSION}/result/${EVAL_DATA}/${EVAL_DATA}.log \
#     --max_th        50 \
#     --type          png \

# # Generate Segmentation pseudo label(train_aug)
# EVAL_DATA=train_aug
# python pseudo_label_gen.py \
#     --datalist          data/voc12/${EVAL_DATA}_id.txt \
#     --crf_pred          train_log/${SESSION}/result/${EVAL_DATA}/crf_npy/crf_5_8 \
#     --label_save_dir    train_log/${SESSION}/result/${EVAL_DATA}/crf_seg

# # evaluate CRF (train_aug)
# echo "trainaug CRF evaluation"
# CUDA_VISIBLE_DEVICES=${GPU} python3 eval.py \
#     --list          data/voc12/${EVAL_DATA}_id.txt \
#     --predict_dir   train_log/${SESSION}/result/${EVAL_DATA}/crf_seg \
#     --gt_dir        ${GT_ROOT} \
#     --comment       $SESSION \
#     --logfile       train_log/${SESSION}/result/${EVAL_DATA}/${EVAL_DATA}.log \
#     --max_th        50 \
#     --type          png \

# # Generate Segmentation pseudo label(val)
# EVAL_DATA=val
# python pseudo_label_gen.py \
#     --datalist          data/voc12/${EVAL_DATA}_id.txt \
#     --crf_pred          train_log/${SESSION}/result/${EVAL_DATA}/crf_npy/crf_5_8 \
#     --label_save_dir    train_log/${SESSION}/result/${EVAL_DATA}/crf_seg

# # evaluate CRF (val)
# echo "val CRF evaluation"
# CUDA_VISIBLE_DEVICES=${GPU} python3 eval.py \
#     --list          data/voc12/${EVAL_DATA}_id.txt \
#     --predict_dir   train_log/${SESSION}/result/${EVAL_DATA}/crf_seg \
#     --gt_dir        ${GT_ROOT} \
#     --comment       $SESSION \
#     --logfile       train_log/${SESSION}/result/${EVAL_DATA}/${EVAL_DATA}.log \
#     --max_th        50 \
#     --type          png \