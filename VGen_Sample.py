import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


if torch.backends.mps.is_built():
    device = torch.device("mps")  # for mac use
else:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Prompt
prompt = "//module half adder "

# Load model and tokenizer
model_name = "shailja/CodeGen_2B_Verilog"
tokenizer = AutoTokenizer.from_pretrained("shailja/fine-tuned-codegen-2B-Verilog")
model = AutoModelForCausalLM.from_pretrained(
    "shailja/fine-tuned-codegen-2B-Verilog"
).to(device)
# model.save_pretrained('/Users/bcchang/Desktop/CS/Computer Organization VHDL/proj/')
# Sample
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
sample = model.generate(input_ids, max_length=128, temperature=0.5, top_p=0.9)
print(tokenizer.decode(sample[0], truncate_before_pattern=[r"endmodule"]) + "endmodule")

prompt2 = "//module 2to1 multiplexer "  # cool! seems workable
input_ids2 = tokenizer(prompt2, return_tensors="pt").input_ids.to(device)
sample2 = model.generate(input_ids2, max_length=128, temperature=0.5, top_p=0.9)
print(
    tokenizer.decode(sample2[0], truncate_before_pattern=[r"endmodule"]) + "endmodule"
)

prompt3 = "//module mux 2-1"
input_ids3 = tokenizer(prompt3, return_tensors="pt").input_ids.to(device)
sample3 = model.generate(input_ids3, max_length=128, temperature=0.5, top_p=0.9)
print(
    tokenizer.decode(sample3[0], truncate_before_pattern=[r"endmodule"]) + "endmodule"
)
