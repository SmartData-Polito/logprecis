
SEED=(5 102 118 130 777)

#Experiment info
DEVICES=0 #if running on cpu, specify no_cuda below
TASK="entity_classification"
LOG_LEVEL="info"

#Model info
MODEL_NAME="ehsanaghaei/SecureBERT"
TOKENIZER_NAME="ehsanaghaei/SecureBERT"
MAX_CHUNK_LENGTH=512

#Input info
TRUNCATION="context_chunking" #"default", "simple_chunking", "context_chunking"
ENTITY="token"
BATCH_SIZE=32

#Output info > Used only if no finetuned path is specified
OUTPUT_PATH="./results/" 

for j in 0 1 2 3 4
do
    IDENTIFIER="inference_SecureBERT__seed_${SEED[j]}"
    
    FINETUNED_PATH="../../2.Training/supervised_training/results/entity_classification/token/ehsanaghaei_SecureBERT/secureBERT_LR_000001/seed_${SEED[j]}/best_model" #"Path_to_your_trained_model"
    
    INPUT_FOLDER="../../1.Dataset/Training/Supervised/Partition/${SEED[j]}/sample_test_corpus.parquet"


    CUDA_VISIBLE_DEVICES="$DEVICES" python inference.py  --task "$TASK" \
        --model_name "$MODEL_NAME" --tokenizer_name "$TOKENIZER_NAME"  \
        --truncation "$TRUNCATION" --log_level "$LOG_LEVEL" \
        --entity "$ENTITY" --batch_size "$BATCH_SIZE" --max_chunk_length "$MAX_CHUNK_LENGTH" \
        --input_data "$INPUT_FOLDER" --finetuned_path "$FINETUNED_PATH" \
        --output_path "$OUTPUT_PATH" --identifier "$IDENTIFIER" --no_cuda
done