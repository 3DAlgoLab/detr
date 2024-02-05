# %%
import pathlib
import pickle
import random
import re
import unicodedata

import keras
import tensorflow as tf
from keras.layers import TextVectorization

# from tensorflow.keras.layers import TextVectorization

# %%

# download dataset provided by Anki: https://www.manythings.org/anki/
text_file = keras.utils.get_file(
    fname="fra-eng.zip",
    origin="http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip",
    extract=True,
)

# %%
# show where the file is located now
text_file = pathlib.Path(text_file).parent / "fra.txt"
print(text_file)

# %%


def normalize(line):
    line = unicodedata.normalize("NFKC", line.strip().lower())
    line = re.sub(r"^([^ \w])(?!\s)", r"\1 ", line)
    line = re.sub(r"(\s[^ \w])(?!\s)", r"\1 ", line)
    line = re.sub(r"(?!\s)([^ \w])$", r" \1", line)
    line = re.sub(r"(?!\s)([^ \w]\s)", r" \1", line)
    eng, fra = line.split("\t")
    fra = "[start] " + fra + " [end]"
    return eng, fra


# normalize each line
with open(text_file) as fp:
    text_pairs = [normalize(line) for line in fp]

for _ in range(5):
    print(random.choice(text_pairs))

with open("text_pairs.pickle", "wb") as fp:
    pickle.dump(text_pairs, fp)

# %%

with open("text_pairs.pickle", "rb") as fp:
    text_pairs = pickle.load(fp)

# split it
random.shuffle(text_pairs)
n_val = int(0.15 * len(text_pairs))
n_train = len(text_pairs) - 2 * n_val
train_pairs = text_pairs[:n_train]
val_pairs = text_pairs[n_train : n_train + n_val]
test_pairs = text_pairs[n_train + n_val :]

# %%
# parameter
vocab_size_en = 10000
vocab_size_fr = 20000
seq_length = 20

# Create Vectorizer
eng_vectorizer = TextVectorization(
    max_tokens=vocab_size_en,
    standardize="None",
    split="whitespace",
    output_mode="int",
    output_sequence_length=seq_length,
)

fra_vectorizer = TextVectorization(
    max_tokens=vocab_size_fr,
    standardize="None",
    split="whitespace",
    output_mode="int",
    output_sequence_length=seq_length + 1,
)

# train the vectorization layer using training dataset
train_eng_texts = [pair[0] for pair in train_pairs]
train_fra_texts = [pair[1] for pair in train_pairs]
eng_vectorizer.adapt(train_eng_texts)
fra_vectorizer.adapt(train_fra_texts)


# %%
with open("vectorize.pickle", "wb") as fp:
    data = {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs,
        "engvec_config": eng_vectorizer.get_config(),
        "engvec_weights": eng_vectorizer.get_weights(),
        "fravec_config": fra_vectorizer.get_config(),
        "fravec_weights": fra_vectorizer.get_weights(),
    }
    pickle.dump(data, fp)


# %%
with open("vectorize.pickle", "rb") as fp:
    data = pickle.load(fp)

train_pairs = data["train"]
val_pairs = data["val"]
test_pairs = data["test"]
# %%
eng_vectorizer = TextVectorization.from_config(data["engvec_config"])
eng_vectorizer.set_weights(data["engvec_weights"])
fra_vectorizer = TextVectorization.from_config(data["fravec_config"])
fra_vectorizer.set_weights(data["fravec_weights"])


# %%
def format_dataset(eng, fra):
    eng = eng_vectorizer(eng)
    fra = fra_vectorizer(fra)
    source = {"encoder_inputs": eng, "decoder_inputs": fra[:, :-1]}
    target = fra[:, 1:]
    return (source, target)


def make_dataset(pairs, batch_size=64):
    eng_texts, fra_texts = zip(*pairs)
    dataset = tf.data.Dataset.from_tensor_slices((list(eng_texts), list(fra_texts)))
    return (
        dataset.shuffle(2048).batch(batch_size).map(format_dataset).prefetch(16).cache()
    )


train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)
# %%
for inputs, targets in train_ds.take(1):
    print(inputs["encoder_inputs"].shape)
    print(inputs["encoder_inputs"])
    print(inputs["decoder_inputs"].shape)
    print(inputs["decoder_inputs"])
    print(f"targets.shape: {targets.shape}")
    print(f"targets[0]: {targets[0]}")

# %%
