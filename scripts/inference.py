from transformers import pipeline

model_path = "./models/checkpoint-final"  # change to your best checkpoint
classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)

review = "The laptop battery is terrible but the screen is amazing."
print(classifier(review))
