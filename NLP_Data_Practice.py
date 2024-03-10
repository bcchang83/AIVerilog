from transformers import AutoTokenizer
from datasets import load_dataset
import html
import wget
import unzip
import os


def lowercase_condition(example):
    return {"condition": example["condition"].lower()}


def filter_nones(x):
    return x["condition"] is not None


def compute_review_length(example):
    return {"review_length": len(example["review"].split())}


def tokenize_function(examples):
    return tokenizer(examples["review"], truncation=True)


def tokenize_and_split(examples):
    return tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )


def tokenize_and_split_v2(examples):
    result = tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
    # Extract mapping between new and old indices
    sample_map = result.pop("overflow_to_sample_mapping")
    for key, values in examples.items():
        result[key] = [values[i] for i in sample_map]
    return result


curpath = os.curdir
abspath = os.path.abspath(curpath)
if (os.path.exists(os.path.join(abspath, "drugsComTrain_raw.tsv")) == False) and (
    os.path.exists(os.path.join(abspath, "drugsComTest_raw.tsv") == False)
):
    print("cannot find data, start download")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip"
    zipfile = wget.download(url)
    zippath = os.path.join(curpath, "drugsCom_raw.zip")
    os.system("tar -xvf {}".format(zippath))


data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
# \t is the tab character in Python
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
for split in drug_dataset.keys():
    assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0"))

drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0", new_column_name="patient_id"
)
print("Clean condition is None")
drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)
print("Lowercase condition")
drug_dataset = drug_dataset.map(lowercase_condition)
print("Compute review length")
drug_dataset = drug_dataset.map(compute_review_length)
print("Filter short review, <30")
drug_dataset = drug_dataset.filter(lambda x: x["review_length"] > 30)
print("Clean html characters")
drug_dataset = drug_dataset.map(lambda x: {"review": html.unescape(x["review"])})

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)
tokenized_dataset = drug_dataset.map(
    tokenize_and_split, batched=True, remove_columns=drug_dataset["train"].column_names
)
print(len(tokenized_dataset["train"]), len(drug_dataset["train"]))
