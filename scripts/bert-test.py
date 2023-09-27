import torch
from transformers import BertTokenizer, BertForMultipleChoice

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMultipleChoice.from_pretrained('bert-base-uncased')

context = "On that day, two people came into the lawyer's office, one holding a notebook and the other maneuvering his walker through the door. The man with the walker started off the meeting by laying out his legal case, while the other man struggled to keep up with all the jargon and kept asking lots of questions."
question = "Which of them is not disabled?"
choices = ["The man with the notebook", "The man holding a walking stick"]

input_ids = []
attention_masks = []

# Tokenizing each choice with the question and context
for choice in choices:
    encoding = tokenizer.encode_plus(question, text_pair=f"{context} {choice}", return_tensors='pt', max_length=512, padding='max_length', truncation=True)
    input_ids.append(encoding['input_ids'])
    attention_masks.append(encoding['attention_mask'])

input_ids = torch.stack(input_ids).squeeze(dim=2)  # Adjust dimensions here
attention_masks = torch.stack(attention_masks).squeeze(dim=2)

outputs = model(input_ids=input_ids, attention_mask=attention_masks)
logits = outputs.logits
prediction = torch.argmax(logits).item()  # Getting the single prediction as Python int
print("Correct Choice:", choices[prediction])  # Directly printing the correct choice