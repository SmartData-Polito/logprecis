
#Experiment details
EXPERIMENT_IDENTIFIER=("secureBERT_LR_00005" "secureBERT_LR_00001" "secureBERT_LR_000005" "secureBERT_LR_000001") 
TASK="entity_classification"
DEVICES=0 #if running on cpu, add --no_cuda below
LOG_LEVEL="info"
OUTPUT_PATH="./results/"

#Input info
INPUT_DATA="../../1.Dataset/Training/Supervised/Partition/777/sample_train_corpus.parquet"
EVAL_SIZE=0.2
SEED=777

#Model info
MODEL_NAME="ehsanaghaei/SecureBERT" #Chosen model
FINETUNED_PATH="../../2.Training/self_supervised_training/results/self_supervision/token/ehsanaghaei_SecureBERT/secureBERT_LR_00005/seed_1/best_model" #Path, on your filesystem, to the finetuned model (e.g., if any domain-adapted) or online models
TOKENIZER_NAME="ehsanaghaei/SecureBERT" #if you use a finetuned tokenizer, specify the path 
MAX_CHUNK_LENGTH=512

#Training details
BATCH_SIZE=8
EPOCHS=50 #Training epochs
LR=(0.00005 0.00001 0.000005 0.000001) #Or smaller, 0.000001
TRUNCATION="context_chunking" #"default", "simple_chunking", "context_chunking"
ENTITY="token"
AVAILABLE_PERCENTAGE=1

for i in 0 1 2 3
do
    CUDA_VISIBLE_DEVICES="$DEVICES" python ./train.py --identifier "${EXPERIMENT_IDENTIFIER[i]}" --task "$TASK" \
        --model_name "$MODEL_NAME" --finetuned_path "$FINETUNED_PATH" --tokenizer_name "$TOKENIZER_NAME"  \
        --clean_start --truncation "$TRUNCATION" --log_level "$LOG_LEVEL" --output_path "$OUTPUT_PATH" \
        --entity "$ENTITY" --input_data "$INPUT_DATA" --available_percentage "$AVAILABLE_PERCENTAGE" \
        --epochs "$EPOCHS" --seed "$SEED" --batch_size "$BATCH_SIZE" --lr "${LR[i]}" \
        --eval_size "$EVAL_SIZE" --max_chunk_length "$MAX_CHUNK_LENGTH"
done