from transformers import XLMRobertaTokenizerFast

# Path to your manually downloaded model folder
model_dir = "model"

# Load tokenizer from SentencePiece model
tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_dir, unk_token="<unk>", model_max_length=512)

# Save tokenizer files to your local model folder
tokenizer.save_pretrained(model_dir)

print("Tokenizer files successfully generated in 'model/' folder.")
