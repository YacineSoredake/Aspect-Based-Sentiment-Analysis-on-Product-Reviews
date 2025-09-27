import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_PATH = "../models/ate_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

id2label = model.config.id2label

def predict_aspects(text):
   
    tokens = tokenizer(text.split(), is_split_into_words=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**tokens)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)

    word_ids = tokens.word_ids()
    results = []
    for idx, word_id in enumerate(word_ids):
        if word_id is None:  
            continue
        token = tokens.tokens()[idx]
        label = id2label[predictions[0][idx].item()]
        results.append((token, label))
    return results

if __name__ == "__main__":
    sentence = "The keyboard is okay, not great but not bad either."
    results = predict_aspects(sentence)

    print("\nSentence:", sentence)
    print("Predictions:")
    for token, label in results:
        print(f"{token:15} -> {label}")
