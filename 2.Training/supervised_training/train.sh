#Experiment details
EXPERIMENT_IDENTIFIER="reproduce_logprecis" 
TASK="entity_classification"
DEVICES=0 #if running on cpu, add --no_cuda below
LOG_LEVEL="info"
OUTPUT_PATH="./results/"
SEED=1

#Input info
INPUT_FOLDER="../../1.Dataset/Training/Supervised/full_supervised_corpus.json"
EVAL_SIZE=0.0 

#Model info
MODEL_NAME="microsoft/codebert-base" #Chosen model
FINETUNED_PATH="SmartDataPolito/SecureShellBert" #Path, on your filesystem, to the finetuned model (e.g., if any domain-adapted) or online models
TOKENIZER_NAME="microsoft/codebert-base" #if you use a finetuned tokenizer, specify the path 
MAX_CHUNK_LENGTH=512

#Training details
BATCH_SIZE=8
EPOCHS=47 #Training epochs
LR=0.000001 #Or smaller, 0.000001
TRUNCATION="context_chunking" #"default", "simple_chunking", "context_chunking"
ENTITY="token"
AVAILABLE_PERCENTAGE=1

CUDA_VISIBLE_DEVICES="$DEVICES" python ./train.py --identifier "$EXPERIMENT_IDENTIFIER" --task "$TASK" \
    --model_name "$MODEL_NAME" --finetuned_path "$FINETUNED_PATH" --tokenizer_name "$TOKENIZER_NAME"  \
    --clean_start --truncation "$TRUNCATION" --log_level "$LOG_LEVEL" --output_path "$OUTPUT_PATH" \
    --entity "$ENTITY" --input_data "$INPUT_FOLDER" --available_percentage "$AVAILABLE_PERCENTAGE" \
    --epochs "$EPOCHS" --seed "$SEED" --batch_size "$BATCH_SIZE" --lr "$LR" \
    --eval_size "$EVAL_SIZE" --max_chunk_length "$MAX_CHUNK_LENGTH"