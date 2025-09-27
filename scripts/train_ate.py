# train_ate.py
import json
import os
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
import evaluate


MODEL_NAME = "bert-base-uncased"   
DATA_PATH = "../data/processed/ate_train.json"
OUTPUT_DIR = "../models/ate_model"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

label_list = ["O", "B-Aspect", "I-Aspect"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align_labels(example):
    tokenized = tokenizer(example["tokens"], is_split_into_words=True, truncation=True)

    word_ids = tokenized.word_ids()
    labels = []
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100) 
        else:
            labels.append(label2id[example["labels"][word_idx]])
    tokenized["labels"] = labels
    
    return {k: v for k, v in tokenized.items() if k != "polarity"}


tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=False)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [label_list[l] for l in label if l != -100]
        for label in labels
    ]
    true_preds = [
        [label_list[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    do_eval=True,
    eval_steps=100,
    save_steps=100,
    logging_dir="./logs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset, 
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f" Model saved at {OUTPUT_DIR}")
