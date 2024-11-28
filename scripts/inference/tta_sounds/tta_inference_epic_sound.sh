
lambda_mi=3.0
lambda_kl=3.0
tta_learning_rate=0.00025
PROP=1.0
METHOD=midl #tent, eta, midl, shot-im
seed=42

for i in "$@"
do
case $i in
    --lambda-mutual=*)
    lambda_mi="${i#*=}"
    shift # past argument=value
    ;;
    --lambda-kl=*)
    lambda_kl="${i#*=}"
    shift # past argument=value
    ;;
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
echo "Mutual info: $lambda_mi"
echo "KL: $lambda_kl"
echo "TTA Learning rate: $tta_learning_rate"
echo "Proportion: $PROP"

if [[ "$PROP" == "0" ]]; then
    PROP=0.0
fi
if [[ "$PROP" == "1" ]]; then
    PROP=1.0
fi

# export CUDA_VISIBLE_DEVICES=3
CONFIG_FILE="configs/finetune/EPIC_sounds/vit-base/finetune_EPIC_vit_B_AV_bottleneck_not-shared.yaml"

BASE_PATH=checkpoints/EPIC-SOUNDS

PATH_TO_CHECKPOINT=$BASE_PATH/EPIC-SOUNDS_baseline.ckpt

PATH_TO_ORIGINAL_PREDS="$BASE_PATH/EPIC_sounds_test_clips_singleview_0.0_audio_original.txt"

# filename = osp.join(self.config.exp_dir, self.config.exp_name, osp.splitext(osp.basename(self.config.DATA.metadata_file))[0] + flags_optional + '.txt')

FILTER_CRITERIA="has_audio"
tta_exp_folder_name=${METHOD}/${seed}/audio_${PROP}_video_0.0/mi_${lambda_mi}_kl_${lambda_kl}_lr_${tta_learning_rate}
PREDICTIONS_DIR_1=$BASE_PATH/${tta_exp_folder_name}/EPIC_sounds_validation_audio_${PROP}_video_0.0_tta_${FILTER_CRITERIA}.txt
SAVE_CHECKPOINT_NAME=$BASE_PATH/${tta_exp_folder_name}/tta_${PROP}.ckpt

if [ ! -f "$SAVE_CHECKPOINT_NAME" ] || [ ! -f "$PREDICTIONS_DIR_1" ]; then # if the checkpoint and the predictions exist, we skip the training
    echo "Checkpoint $SAVE_CHECKPOINT_NAME or the predictions from this stage $PREDICTIONS_DIR_1 DO NOT exist. Executing Python script..."
    python inference_lightning.py --cfg $CONFIG_FILE \
                precision 16 \
                seed $seed \
                DATA.metadata_file "datasets/EPIC_sounds_shards/EPIC_sounds_validation_audio_${PROP}_video_0.0_tta.csv" \
                INFERENCE.batch_size_per_gpu 3 \
                INFERENCE.num_workers 6 \
                INFERENCE.enabled True \
                INFERENCE.num_views 3 \
                INFERENCE.num_spatial_crops 1 \
                DATA.biased_sample_clip True \
                INFERENCE.checkpoint $PATH_TO_CHECKPOINT \
                wandb.use_wandb False \
                INFERENCE.type tta \
                INFERENCE.save_checkpoint True \
                INFERENCE.save_ckpt_name $SAVE_CHECKPOINT_NAME \
                INFERENCE.original_model_predictions $PATH_TO_ORIGINAL_PREDS \
                DATA.filter_criteria $FILTER_CRITERIA \
                INFERENCE.predictions_dir $PREDICTIONS_DIR_1 \
                INFERENCE.lambda_mi $lambda_mi \
                INFERENCE.lambda_kl $lambda_kl \
                INFERENCE.tta_lr $tta_learning_rate \
                INFERENCE.TTA_METHOD $METHOD \
                INFERENCE.load_train_config False \

fi  
PATH_TO_CHECKPOINT="$SAVE_CHECKPOINT_NAME"

FILTER_CRITERIA="no_audio"
PREDICTIONS_DIR_2=$BASE_PATH/${tta_exp_folder_name}/EPIC_sounds_test_clips_singleview_${PROP}_audio_tta_${FILTER_CRITERIA}.txt
if [ ! -f "$PREDICTIONS_DIR_2" ]; then # if the predictions exist, we skip the training
    echo "Predictions from this stage $PREDICTIONS_DIR_2 DO NOT exist. Executing Python script..."
    python inference_lightning.py --cfg $CONFIG_FILE \
                precision 16 \
                DATA.metadata_file "datasets/EPIC_sounds_shards/EPIC_sounds_test_clips_singleview_${PROP}_audio.csv" \
                INFERENCE.batch_size_per_gpu 4 \
                INFERENCE.num_workers 6 \
                INFERENCE.enabled True \
                INFERENCE.num_views 1 \
                INFERENCE.num_spatial_crops 1 \
                DATA.biased_sample_clip True \
                INFERENCE.checkpoint $PATH_TO_CHECKPOINT \
                wandb.use_wandb False \
                INFERENCE.type original \
                DATA.filter_criteria $FILTER_CRITERIA \
                INFERENCE.predictions_dir $PREDICTIONS_DIR_2 \
                INFERENCE.TTA_METHOD $METHOD \
                INFERENCE.load_train_config False \

fi

python process_tta_res.py --original_pred_file $PATH_TO_ORIGINAL_PREDS --tta_files $PREDICTIONS_DIR_1 $PREDICTIONS_DIR_2 \
            --config-lambda-mi $lambda_mi --config-lambda-kl $lambda_kl --config-tta-lr $tta_learning_rate --seed $seed
