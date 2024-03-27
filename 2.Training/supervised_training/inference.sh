########## RUN THIS FILE TO USE `SmartDataPolito/logprecis` FOR INFERENCE AND REPRODUCE THE PREDICTIONS FOR 3.Characterization

#Experiment info
IDENTIFIER="inference"
DEVICES=0 #if running on cpu, specify no_cuda below
TASK="entity_classification"
LOG_LEVEL="info"

#Model info
MODEL_NAME="ehsanaghaei/SecureBERT"
FINETUNED_PATH="" #"Path_to_your_trained_model"
TOKENIZER_NAME="ehsanaghaei/SecureBERT"
MAX_CHUNK_LENGTH=512

#Input info
INPUT_FOLDER="../../1.Dataset/Inference/cyberlab_data.csv"
TRUNCATION="context_chunking" #"default", "simple_chunking", "context_chunking"
ENTITY="token"
BATCH_SIZE=32

#Output info > Used only if no finetuned path is specified
OUTPUT_PATH="./results/" 

CUDA_VISIBLE_DEVICES="$DEVICES" python inference.py  --task "$TASK" \
    --model_name "$MODEL_NAME" --tokenizer_name "$TOKENIZER_NAME"  \
    --truncation "$TRUNCATION" --log_level "$LOG_LEVEL" \
    --entity "$ENTITY" --batch_size "$BATCH_SIZE" --max_chunk_length "$MAX_CHUNK_LENGTH" \
    --input_data "$INPUT_FOLDER" --finetuned_path "$FINETUNED_PATH" \
    --output_path "$OUTPUT_PATH" --identifier "$IDENTIFIER"