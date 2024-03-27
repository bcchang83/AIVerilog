"""
In this program, I try to use a verilog dataset to fine-tune model codegen2-1B.
"""

import torch
import re
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def tokenize_function(example):
    return tokenizer(example["prompt"], example["clean_text"], truncation=True)


def clean_comment(example):
    return {"clean_text": re.sub(r"//(.)*", "", example["text"])}


def clean_repeat(example):
    return {"unique": example.unique("text")}


def get_prompt(example):
    # pattern = r"(module (.)*)|(always @(.)*)"
    pattern = r"(module (.)*)"
    promptRegex = re.compile(pattern)
    prompt = promptRegex.search(example["text"])
    if prompt:
        return {"prompt": "//" + prompt.group()}
    else:
        return {"prompt": None}


def extrach_module(example):
    # module_pattern = r"module\s+(\w+)\s*\((.*?)\);(.*?)endmodule"
    module_pattern = r"module(.)*;(.*?)endmodule"
    module_match = re.search(module_pattern, example["clean_text"], re.DOTALL)
    if module_match:
        return {"clean_text": module_match.group()}
    else:
        return {"clean_text": None}


def calculate_text_length(example):
    return {"clean_text_length": len(example["clean_text"])}


if torch.backends.mps.is_built():
    device = torch.device("mps")  # for mac use
else:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

raw_datasets = load_dataset("shailja/Verilog_GitHub")
unique_datasets = raw_datasets.unique("text")
unique_datasets["text"] = unique_datasets["train"]
del unique_datasets["train"]
unique_datasets = datasets.Dataset.from_dict(unique_datasets)
unique_datasets = unique_datasets.map(clean_comment)
unique_datasets = unique_datasets.map(extrach_module)
unique_datasets = unique_datasets.filter(lambda x: x["clean_text"] is not None)
unique_datasets = unique_datasets.map(get_prompt)
unique_datasets = unique_datasets.filter(lambda x: x["prompt"] is not None)
unique_datasets = unique_datasets.map(calculate_text_length)
train_test_ds = unique_datasets.train_test_split(test_size=0.4)
test_val_ds = train_test_ds["test"].train_test_split(test_size=0.5)
ttv_ds = datasets.DatasetDict(
    {
        "train": train_test_ds["train"],
        "validation": test_val_ds["train"],
        "test": test_val_ds["test"],
    }
)
checkpoint = "Salesforce/codegen2-1B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# tokenized_datasets = ttv_ds.map(tokenize_function, batched=True)


# model = AutoModelForCausalLM.from_pretrained(
#     checkpoint, trust_remote_code=True, revision="main"
# )


"""
text = "def hello_world():"
input_ids = tokenizer(text, return_tensors="pt").input_ids
generated_ids = model.generate(input_ids, max_length=128)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
"""
