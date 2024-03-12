#Experiment info
EXPERIMENT_IDENTIFIER="Your_experiment_ID" 
DEVICES=0 #if running on cpu, add --no_cuda below
LOG_LEVEL="info"
OUTPUT_PATH="./experiments/"
BATCH_SIZE=16 #or 8
EPOCHS=1 #Training epochs
PATIENCE=4 #Patience before reduce on plateau
OBSERVED_VAL_METRIC="loss" #Which metric to consider to check if we are improving 
LR=0.000005 #Or smaller, 0.000001
#Model info
MODEL_NAME="microsoft/codebert-base" #Chosen model
FINETUNED_PATH="" #Path, on your filesystem, to the finetuned model (e.g., if any domain-adapted)
TOKENIZER_NAME="microsoft/codebert-base" #if you use a finetuned tokenizer, specify the path 
#Input info
INPUT_FOLDER="../../1.Dataset/Training/Supervised/train_corpus.parquet"
TRUNCATION="context_chunking" #"default", "simple_chunking", "context_chunking"
ENTITY="token"
AVAILABLE_PERCENTAGE=1
EVAL_SIZE=0.0


CUDA_VISIBLE_DEVICES="$DEVICES" python train.py --identifier "$EXPERIMENT_IDENTIFIER" \
    --model_name "$MODEL_NAME" --finetuned_path "$FINETUNED_PATH" --tokenizer_name "$TOKENIZER_NAME"  \
    --clean_start --truncation "$TRUNCATION" --log_level "$LOG_LEVEL" --output_path "$OUTPUT_PATH" \
    --entity "$ENTITY" --input_data "$INPUT_FOLDER" --available_percentage "$AVAILABLE_PERCENTAGE" \
    --epochs "$EPOCHS" --patience "$PATIENCE" --observed_val_metric "$OBSERVED_VAL_METRIC" \
    --batch_size "$BATCH_SIZE" --lr "$LR" --eval_size "$EVAL_SIZE"