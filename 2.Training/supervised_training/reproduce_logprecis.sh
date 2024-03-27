########## RUN THIS FILE TO TRAIN THE EQUIVALENT OF `SmartDataPolito/logprecis`

SEED=(5 102 118 130 777)

for j in 0 1 2 3 4
do
    #Experiment details
    EXPERIMENT_IDENTIFIER=("logprecis_LR_00005" "logprecis_LR_00001" "logprecis_LR_000005" "logprecis_LR_000001")  
    TASK="entity_classification"
    DEVICES=0 #if running on cpu, add --no_cuda below
    LOG_LEVEL="info"
    OUTPUT_PATH="./results/"

    #Input info
    INPUT_DATA="../../1.Dataset/Training/Supervised/Partition/${SEED[j]}/sample_train_corpus.parquet"
    
    #Model info
    MODEL_NAME="microsoft/codebert-base" #Chosen model
    FINETUNED_PATH="SmartDataPolito/SecureShellBert" #Path, on your filesystem, to the finetuned model (e.g., if any domain-adapted) or online models
    TOKENIZER_NAME="microsoft/codebert-base" #if you use a finetuned tokenizer, specify the path 
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
            --epochs "$EPOCHS" --seed "${SEED[j]}" --batch_size "$BATCH_SIZE" --lr "${LR[i]}" \
            --max_chunk_length "$MAX_CHUNK_LENGTH"
    done
done