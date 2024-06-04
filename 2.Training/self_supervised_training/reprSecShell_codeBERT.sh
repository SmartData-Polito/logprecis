########## RUN THIS FILE TO TRAIN THE EQUIVALENT OF `SmartDataPolito/SecureShellBert`

#Experiment info

TASK="self_supervision"
DEVICES=0 #if running on cpu, add --no_cuda below
LOG_LEVEL="info"
OUTPUT_PATH="./results/"

#Model info
MODEL_NAME="microsoft/codebert-base" #Chosen model
FINETUNED_PATH="" #Path, on your filesystem, to the finetuned model (e.g., if any domain-adapted)
TOKENIZER_NAME="microsoft/codebert-base" #if you use a finetuned tokenizer, specify the path 
MAX_CHUNK_LENGTH=512
ENTITY="statement"

#Training details
BATCH_SIZE=12 #or 8
EPOCHS=10 #Training epochs
#LR=(0.00001 0.00005 0.000075)
LR=0.000075
MLM_PROBABILITY=0.15
SEED=1

#Input info
INPUT_FOLDER="../../1.Dataset/Training/Self_supervised/training_set.csv"
VALIDATION_PATH="../../1.Dataset/Training/Self_supervised/validation_set.csv"

EXPERIMENT_IDENTIFIER=("baseline_codeBERT_000075") 
CUDA_VISIBLE_DEVICES="$DEVICES" python train.py --identifier "$EXPERIMENT_IDENTIFIER" --task "$TASK" \
    --model_name "$MODEL_NAME" --finetuned_path "$FINETUNED_PATH" --tokenizer_name "$TOKENIZER_NAME"  \
    --clean_start  --log_level "$LOG_LEVEL" --output_path "$OUTPUT_PATH" --mlm_probability "$MLM_PROBABILITY" \
    --input_data "$INPUT_FOLDER" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE"  --lr "$LR" --entity "$ENTITY"\
    --validation_path "$VALIDATION_PATH" --max_chunk_length "$MAX_CHUNK_LENGTH" --seed "$SEED"
#for i in 0 1 2 3
#do
#EXPERIMENT_IDENTIFIER=("baseline_codeBERT_${LR[i]}") 
#CUDA_VISIBLE_DEVICES="$DEVICES" python train.py --identifier "$EXPERIMENT_IDENTIFIER" --task "$TASK" \
#   --model_name "$MODEL_NAME" --finetuned_path "$FINETUNED_PATH" --tokenizer_name "$TOKENIZER_NAME"  \
#   --clean_start  --log_level "$LOG_LEVEL" --output_path "$OUTPUT_PATH" --mlm_probability "$MLM_PROBABILITY" \
#   --input_data "$INPUT_FOLDER" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE"  --lr "${LR[i]}" \
#   --validation_path "$VALIDATION_PATH" --max_chunk_length "$MAX_CHUNK_LENGTH" 
#done