# Roberta-Base-Fine-Tuned-for-Spam-Ham-Detection
Base Transformer: Roberta Base 

Fine Tuned on Spam Ham SMS Dataset

**Evaluation Metrics:**

1) Cross Entropy Loss
2)  F1 Score

**Metrics Results:**

1) Loss -> 0.13
2) F1 -> 77.99


**How To Use?**
In your python code simply use the following program statements to import the tokenizer and the model

```
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("SalehAhmad/roberta-base-finetuned-sms-spam-ham-detection")
model = AutoModelForSequenceClassification.from_pretrained("SalehAhmad/roberta-base-finetuned-sms-spam-ham-detection")
```

After the tokenizer and the model have been initiated then to use the model simply follow the following statements

```
inputSentence = input().lower()
inputs = tokenizer(inputSentence, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
logits = model(inputs['input_ids'],inputs['attention_mask']).logits
predicted_class_id = logits.argmax().item()
print('Prediction:',model.config.id2label[predicted_class_id])
```
