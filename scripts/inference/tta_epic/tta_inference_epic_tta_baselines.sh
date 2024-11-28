tta_learning_rate=0.00025
PROP=1.0
METHOD=shot-im #tent, eta, midl, shot-im
UPDATE_BN_ONLY=True
seed=42
for i in "$@"
do
case $i in
    --tta-learning-rate=*)
    tta_learning_rate="${i#*=}"
    shift # past argument=value
    ;;
    --proportion=*)
    PROP="${i#*=}"
    shift # past argument=value
    ;;
    --method=*)
    METHOD="${i#*=}"
    shift # past argument=value
    ;;
    --update-bn-only=*)
    UPDATE_BN_ONLY="${i#*=}"
    shift # past argument=value
    ;;
    --seed=*)
    seed="${i#*=}"
    shift # past argument=value
    ;;
    *)
          # unknown option
    ;;
esac
done


echo "Running inference for EPIC sounds"
echo "Parameters:"
echo "TTA Learning rate: $tta_learning_rate"
echo "Proportion: $PROP"
echo "Method: $METHOD"
echo "Update BN only: $UPDATE_BN_ONLY"

if [[ "$PROP" == "0" ]]; then
    PROP=0.0
fi
if [[ "$PROP" == "1" ]]; then
    PROP=1.0
fi

# export CUDA_VISIBLE_DEVICES=3
CONFIG_FILE="configs/finetune/EPIC/vit-base/finetune_EPIC_vit_B_AV_bottleneck_not-shared.yaml"
# CONFIG_FILE="configs/finetune/EPIC/default.yaml" 

BASE_PATH=checkpoints/EPIC-KITCHENS

PATH_TO_CHECKPOINT=$BASE_PATH/EPIC-KITCHENS_baseline.ckpt

PATH_TO_ORIGINAL_PREDS="$BASE_PATH/EPIC_test_clips_singleview_0.0_video_original.txt"


tta_exp_folder_name=${METHOD}/audio_0.0_video_${PROP}/lr_${tta_learning_rate}
PREDICTIONS_DIR_1=$BASE_PATH/${tta_exp_folder_name}/EPIC_validation_audio_0.0_video_${PROP}_tta.txt
SAVE_CHECKPOINT_NAME=${tta_exp_folder_name}/tta_${PROP}.ckpt
SAVE_CHECKPOINT_NAME=$BASE_PATH/${tta_exp_folder_name}/tta_${PROP}.ckpt

if  [ ! -f "$PREDICTIONS_DIR_1" ]; then # if the checkpoint and the predictions exist, we skip the training
    echo "Checkpoint $SAVE_CHECKPOINT_NAME or the predictions from this stage $PREDICTIONS_DIR_1 DO NOT exist. Executing Python script..."
    python inference_lightning.py --cfg $CONFIG_FILE \
                precision 16 \
                seed $seed \
                DATA.metadata_file "datasets/EPIC_shards/EPIC_test_clips_singleview_${PROP}_video.csv" \
                INFERENCE.batch_size_per_gpu 8 \
                INFERENCE.num_workers 6 \
                INFERENCE.enabled True \
                INFERENCE.num_views 1 \
                INFERENCE.num_spatial_crops 1 \
                DATA.biased_sample_clip False \
                INFERENCE.checkpoint $PATH_TO_CHECKPOINT \
                wandb.use_wandb False \
                INFERENCE.type tta \
                INFERENCE.save_checkpoint True \
                INFERENCE.save_ckpt_name $SAVE_CHECKPOINT_NAME \
                INFERENCE.original_model_predictions $PATH_TO_ORIGINAL_PREDS \
                INFERENCE.predictions_dir $PREDICTIONS_DIR_1 \
                INFERENCE.tta_lr $tta_learning_rate \
                INFERENCE.TTA_METHOD $METHOD \
                INFERENCE.update_bn_only $UPDATE_BN_ONLY \
                INFERENCE.load_train_config False \
                DATA.filter_criteria nothing \

fi  

PREDICTIONS_DIR_2=checkpoints/EPIC-KITCHENS/empty_preds.txt


python process_tta_res.py --original_pred_file $PATH_TO_ORIGINAL_PREDS --tta_files $PREDICTIONS_DIR_1 $PREDICTIONS_DIR_2 --config-tta-lr $tta_learning_rate