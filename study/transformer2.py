# %%
# env: hf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2", output_hidden_states=True)

# %%
text = "The Shawshank"
input1 = tokenizer.encode(text, return_tensors="pt")
output = model.generate(input1, max_length=5, do_sample=False)

print("\n", tokenizer.decode(output[0]))
# %%
model.transformer.wte
# %%

r1 = model.transformer.wte(torch.tensor(464))
r1.shape
# %%
