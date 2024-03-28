"""
In this program, I try to use a verilog dataset to fine-tune codegen
"""

import torch
import re
import datasets
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)


def tokenize_function(example):
    outputs = tokenizer(
        example["clean_text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)

    return {"input_ids": input_batch}


def clean_comment(example):
    return {"clean_text": re.sub(r"//(.)*", "", example["text"])}


def extrach_module(example):
    # module_pattern = r"module\s+(\w+)\s*\((.*?)\);(.*?)endmodule"
    module_pattern = r"module(.)*;(.*?)endmodule"
    module_match = re.search(module_pattern, example["clean_text"], re.DOTALL)
    if module_match:
        prompt_pattern = r"(module (.)*)"
        promptRegex = re.compile(prompt_pattern)
        prompt = promptRegex.search(module_match.group())
        if prompt:
            return {"clean_text": "//" + prompt.group() + "\n" + module_match.group()}
        else:
            return {"clean_text": None}
    else:
        return {"clean_text": None}


def calculate_text_length(example):
    return {"clean_text_length": len(example["clean_text"])}


def dataPreprocess(raw_datasets, small_data=False):
    unique_datasets = raw_datasets.unique("text")
    unique_datasets["text"] = unique_datasets["train"]
    del unique_datasets["train"]
    unique_datasets = datasets.Dataset.from_dict(unique_datasets)
    unique_datasets = unique_datasets.map(clean_comment)
    unique_datasets = unique_datasets.map(extrach_module)
    unique_datasets = unique_datasets.filter(lambda x: x["clean_text"] is not None)
    if small_data:
        print("Warning: using the small dataset!!!")
        unique_datasets = unique_datasets.map(calculate_text_length)
        unique_datasets = unique_datasets.filter(
            lambda x: x["clean_text_length"] <= 500
        )
    return unique_datasets


if torch.backends.mps.is_built():
    device = torch.device("mps")  # for Apple silicon
else:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device = {}".format(device))

pretrained_checkpoint = "Salesforce/codegen-350M-mono"
checkpoint = "vgen-ds"
epoch_num = 10
batch_size = 4
small_data = True  # just for quick try
context_length = 128

raw_datasets = load_dataset("shailja/Verilog_GitHub")
unique_datasets = dataPreprocess(raw_datasets, small_data=small_data)
train_test_ds = unique_datasets.train_test_split(test_size=0.4)
test_val_ds = train_test_ds["test"].train_test_split(test_size=0.5)
ttv_ds = datasets.DatasetDict(
    {
        "train": train_test_ds["train"],
        "validation": test_val_ds["train"],
        "test": test_val_ds["test"],
    }
)

tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint)

tokenized_datasets = ttv_ds.map(
    tokenize_function, batched=True, remove_columns=ttv_ds["train"].column_names
)


model = AutoModelForCausalLM.from_pretrained(pretrained_checkpoint)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
args = TrainingArguments(
    output_dir=checkpoint,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=5_000,
    gradient_accumulation_steps=8,
    num_train_epochs=epoch_num,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_000,
    fp16=False,
    push_to_hub=False,
    use_mps_device=device == "mps",
)
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)
trainer.train()

# test prediction
text = "//module 2 to 1 mux"
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
generated_ids = model.generate(input_ids, max_length=context_length).to(device)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
