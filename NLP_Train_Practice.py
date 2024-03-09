# hello git
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    get_scheduler,
    AdamW,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
import evaluate


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


num_epochs = 3
raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(
    ["sentence1", "sentence2", "idx"]
)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)


"""#check batch shape
for batch in train_dataloader:
    break
print({k: v.shape for k, v in batch.items()})
"""
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
num_training_steps = num_epochs * len(train_dataloader)
progress_bar = tqdm(range(num_training_steps))
optimizer = AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
"""#check model output
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)
"""
if torch.has_mps:
    device = torch.device("mps")  # for mac use
else:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print("Training device: ", device)

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])
print("/n")
print(metric.compute())
