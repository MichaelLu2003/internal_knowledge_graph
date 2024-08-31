from transformers import AutoTokenizer

# Load your tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')

# Define your prompt and subject
prompt = "{} was born in"
subject = "Max Jacob"

# Format the sentence by inserting the subject into the prompt
sentence = prompt.format(subject)

# Tokenize the sentence
tokenized_output = tokenizer(sentence, return_tensors="pt")

# Print the tokenized input IDs and their corresponding tokens
input_ids = tokenized_output["input_ids"][0]
tokens = [tokenizer.decode([token_id]) for token_id in input_ids]

print(f"Original Sentence: {sentence}")
print(f"Tokenized Input IDs: {input_ids}")
print(f"Tokens: {tokens}")
