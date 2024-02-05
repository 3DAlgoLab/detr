# %%
import matplotlib.pyplot as plt
import numpy as np

# %%


def positional_encoding(max_length, d_model):
    pos_enc_matrix = np.zeros((max_length, d_model))
    for pos in range(max_length):
        for i in range(d_model):
            angle = pos / np.power(10000, 2 * i / d_model)
            pos_enc_matrix[pos, i] = np.sin(angle) if i % 2 == 0 else np.cos(angle)

    return pos_enc_matrix


max_length = 10
d_model = 12  # dimension of the model


pos_enc_matrix = positional_encoding(max_length, d_model)
print(pos_enc_matrix)
