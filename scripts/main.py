import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification

# -------------------------------
# Load ATE model
MODEL_ATE_PATH = "../models/ate_model"
ate_tokenizer = AutoTokenizer.from_pretrained(MODEL_ATE_PATH)
ate_model = AutoModelForTokenClassification.from_pretrained(MODEL_ATE_PATH)
id2label = ate_model.config.id2label

# Load ASC model
ASC_MODEL_PATH = "../models/asc_model/checkpoint-885"
asc_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
asc_model = AutoModelForSequenceClassification.from_pretrained(ASC_MODEL_PATH)

# -------------------------------
# Helper functions
def extract_aspects(text: str):
    tokens = ate_tokenizer(text.split(), is_split_into_words=True, return_tensors="pt")
    with torch.no_grad():
        outputs = ate_model(**tokens)
    predictions = torch.argmax(outputs.logits, dim=2)

    word_ids = tokens.word_ids()
    aspects, current = [], []

    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        token = tokens.tokens()[idx]
        label = id2label[predictions[0][idx].item()]

        if label.startswith("B-Aspect"):
            if current:
                aspects.append(" ".join(current))
                current = []
            current.append(token)
        elif label.startswith("I-Aspect"):
            current.append(token)
        else:
            if current:
                aspects.append(" ".join(current))
                current = []

    if current:
        aspects.append(" ".join(current))

    aspects = [asp.replace("##", "") for asp in aspects]
    return aspects

def classify_sentiment(sentence: str, aspect: str):
    inputs = asc_tokenizer(
        sentence,
        aspect,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    with torch.no_grad():
        outputs = asc_model(**inputs)
    predicted_class = outputs.logits.argmax(dim=-1).item()

    label_map = {1: "negative", 2: "neutral", 3: "positive"}
    return label_map.get(predicted_class, "unknown")

def analyze_sentence(sentence: str):
    aspects = extract_aspects(sentence)
    results = {asp: classify_sentiment(sentence, asp) for asp in aspects}
    return {"sentence": sentence, "analysis": results}

# -------------------------------
# FastAPI setup
app = FastAPI(title="Aspect-Based Sentiment Analysis API")

class RequestText(BaseModel):
    sentence: str

@app.post("/analyze")
def analyze_text(request: RequestText):
    sentence = request.sentence
    aspects = extract_aspects(sentence)

    results = {}
    for aspect in aspects:
        sentiment = classify_sentiment(sentence, aspect)
        results[aspect] = sentiment

    return {"sentence": sentence, "analysis": results}
