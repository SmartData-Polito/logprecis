########## RUN THIS FILE TO USE `SmartDataPolito/logprecis` FOR INFERENCE AND REPRODUCE THE PREDICTIONS FOR 3.Characterization

#Experiment info
IDENTIFIER="cyberlab_test2"
DEVICES=0 #if running on cpu, specify no_cuda below
TASK="entity_classification"
LOG_LEVEL="debug"

#Model info
MODEL_NAME="microsoft/codebert-base"
FINETUNED_PATH="./results/entity_classification/statement/microsoft_codebert-base/statement_codebert/seed_5/best_model" #"Path_to_your_trained_model"
TOKENIZER_NAME="microsoft/codebert-base"
MAX_CHUNK_LENGTH=512

#Input info
INPUT_FOLDER="../../1.Dataset/Inference/sample_cyberlab.parquet"
TRUNCATION="context_chunking" #"default", "simple_chunking", "context_chunking"
ENTITY="word"
BATCH_SIZE=64

#Output info > Used only if no finetuned path is specified
OUTPUT_PATH="./results/" 

CUDA_VISIBLE_DEVICES="$DEVICES" python ./inference.py  --task "$TASK" \
    --model_name "$MODEL_NAME" --tokenizer_name "$TOKENIZER_NAME"  \
    --truncation "$TRUNCATION" --log_level "$LOG_LEVEL" \
    --entity "$ENTITY" --batch_size "$BATCH_SIZE" --max_chunk_length "$MAX_CHUNK_LENGTH" \
    --input_data "$INPUT_FOLDER" --finetuned_path "$FINETUNED_PATH" \
    --output_path "$OUTPUT_PATH" --identifier "$IDENTIFIER"