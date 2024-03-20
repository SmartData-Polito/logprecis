#Experiment info
EXPERIMENT_IDENTIFIER="Your_self_supervised_eperiment_ID" 
TASK="self_supervision"
DEVICES=0 #if running on cpu, add --no_cuda below
LOG_LEVEL="info"
OUTPUT_PATH="./results/"

#Model info
MODEL_NAME="microsoft/codebert-base" #Chosen model
FINETUNED_PATH="" #Path, on your filesystem, to the finetuned model (e.g., if any domain-adapted)
TOKENIZER_NAME="microsoft/codebert-base" #if you use a finetuned tokenizer, specify the path 
MAX_CHUNK_LENGTH=256

#Training details
BATCH_SIZE=16 #or 8
EPOCHS=1 #Training epochs
LR=0.000005 #Or smaller, 0.000001
TRUNCATION="default" #"default", "simple_chunking", "context_chunking"
AVAILABLE_PERCENTAGE=1
ENTITY="statement" #["token", "statement"]

#Input info
INPUT_FOLDER="../../1.Dataset/Training/Self_supervised/training_set.csv"
VALIDATION_PATH="../../1.Dataset/Training/Self_supervised/validation_set.csv"


CUDA_VISIBLE_DEVICES="$DEVICES" python train.py --identifier "$EXPERIMENT_IDENTIFIER" --task "$TASK" \
    --model_name "$MODEL_NAME" --finetuned_path "$FINETUNED_PATH" --tokenizer_name "$TOKENIZER_NAME"  \
    --clean_start --truncation "$TRUNCATION" --log_level "$LOG_LEVEL" --output_path "$OUTPUT_PATH" \
    --input_data "$INPUT_FOLDER" --available_percentage "$AVAILABLE_PERCENTAGE" \
    --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr "$LR" --validation_path "$VALIDATION_PATH" \
    --max_chunk_length "$MAX_CHUNK_LENGTH" --entity "$ENTITY"