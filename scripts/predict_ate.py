import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Path where your fine-tuned model was saved
MODEL_PATH = "../models/ate_model"

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

# Labels (same as training)
id2label = model.config.id2label

def predict_aspects(text):
    # Tokenize input
    tokens = tokenizer(text.split(), is_split_into_words=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**tokens)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)

    # Map back to tokens
    word_ids = tokens.word_ids()
    results = []
    for idx, word_id in enumerate(word_ids):
        if word_id is None:  # skip special tokens
            continue
        token = tokens.tokens()[idx]
        label = id2label[predictions[0][idx].item()]
        results.append((token, label))
    return results

if __name__ == "__main__":
    sentence = "The laptop battery lasts all day, but the keyboard feels cheap, the screen brightness is disappointing outdoors, and the speakers are surprisingly clear for this price."
    results = predict_aspects(sentence)

    print("\nSentence:", sentence)
    print("Predictions:")
    for token, label in results:
        print(f"{token:15} -> {label}")
