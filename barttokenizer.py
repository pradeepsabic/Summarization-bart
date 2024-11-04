from transformers import BartTokenizer

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
text = "Transformers are amazing for NLP tasks!"
inputs = tokenizer(text, return_tensors='pt')
print(inputs)  # Outputs tokenized input in tensor format
