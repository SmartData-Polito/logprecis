#Experiment info
DEVICES=0 #if running on cpu, specify no_cuda below
LOG_LEVEL="info"
BATCH_SIZE=16
#Model info
MODEL_NAME="SmartDataPolito/logprecis"
FINETUNED_PATH="" #"Path_to_your_trained_model"
TOKENIZER_NAME="SmartDataPolito/logprecis"
#Input info
INPUT_FOLDER="../../1.Dataset/Training/Supervised/test_corpus.parquet"
TRUNCATION="context_chunking" #"default", "simple_chunking", "context_chunking"
ENTITY="token"
#Output info > Used only if no finetuned path is specified
OUTPUT_PATH="./experiments/" 
IDENTIFIER="inference"

CUDA_VISIBLE_DEVICES="$DEVICES" python inference.py  \
    --model_name "$MODEL_NAME" --tokenizer_name "$TOKENIZER_NAME"  \
    --truncation "$TRUNCATION" --log_level "$LOG_LEVEL" \
    --entity "$ENTITY" --batch_size "$BATCH_SIZE" \
    --input_data "$INPUT_FOLDER" --finetuned_path "$FINETUNED_PATH" \
    --output_path "$OUTPUT_PATH" --identifier "$IDENTIFIER"