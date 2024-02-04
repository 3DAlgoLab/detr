# %%
from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

sentence = (
    "He didnt want to talk about cells on the cell phone "
    "because he considered it boring"
)
inputs = tokenizer.encode(
    sentence,
    return_tensors="pt",
    add_special_tokens=True,
)
print(inputs)
print(inputs.shape, inputs.dtype)

tokens = tokenizer.convert_ids_to_tokens(inputs[0])
print(tokens)

# %%
from bertviz import head_view  # noqa: E402


def show_head_view(model, tokenizer, sentence):
    input_ids = tokenizer.encode(sentence, return_tensors="pt", add_special_tokens=True)
    attention = model(input_ids)[-1]
    tokens = tokenizer.convert_ids_to_tokens(list(input_ids[0]))
    head_view(attention, tokens)


show_head_view(model, tokenizer, sentence)
# %%
