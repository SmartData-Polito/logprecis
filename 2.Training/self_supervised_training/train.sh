#Experiment info
EXPERIMENT_IDENTIFIER=("secureBERT_LR_00005" "secureBERT_LR_00001" "secureBERT_LR_000005" "secureBERT_LR_000001") 
TASK="self_supervision"
DEVICES=0 #if running on cpu, add --no_cuda below
LOG_LEVEL="info"
OUTPUT_PATH="./results/"

#Model info
MODEL_NAME="ehsanaghaei/SecureBERT" #Chosen model
FINETUNED_PATH="" #Path, on your filesystem, to the finetuned model (e.g., if any domain-adapted)
TOKENIZER_NAME="ehsanaghaei/SecureBERT" #if you use a finetuned tokenizer, specify the path 
MAX_CHUNK_LENGTH=256

#Training details
BATCH_SIZE=16 #or 8
EPOCHS=10 #Training epochs
LR=(0.00005 0.00001 0.000005 0.000001)
MLM_PROBABILITY=0.15

#Input info
INPUT_FOLDER="../../1.Dataset/Training/Self_supervised/training_set.csv"
VALIDATION_PATH="../../1.Dataset/Training/Self_supervised/validation_set.csv"

for i in 0 1 2 3
do
    CUDA_VISIBLE_DEVICES="$DEVICES" python train.py --identifier "${EXPERIMENT_IDENTIFIER[i]}" --task "$TASK" \
        --model_name "$MODEL_NAME" --finetuned_path "$FINETUNED_PATH" --tokenizer_name "$TOKENIZER_NAME"  \
        --clean_start  --log_level "$LOG_LEVEL" --output_path "$OUTPUT_PATH" --mlm_probability "$MLM_PROBABILITY" \
        --input_data "$INPUT_FOLDER" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE"  --lr "${LR[i]}" \
        --validation_path "$VALIDATION_PATH" --max_chunk_length "$MAX_CHUNK_LENGTH" --adaptation "single_session"
done