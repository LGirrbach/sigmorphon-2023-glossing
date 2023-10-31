# Make dummy word encodings
src_word_encodings = torch.randn(2, char_encodings.shape[0], 512)

# Word to batch mapping
batch_indices = torch.arange(char_encodings.shape[0], device=char_encodings.device)
batch_indices = batch_indices.reshape(-1, 1, 1)
batch_indices = batch_indices.expand(-1, char_encodings.shape[1], 1)
batch_indices = batch_indices.reshape(-1, 1)
batch_indices[0, 0] = -1

# Map chars to batch ids
num_words, chars_per_word = batch.word_extraction_index.shape
word_extraction_index_flat = batch.word_extraction_index.flatten()
batch_indices = torch.index_select(
    batch_indices, dim=0, index=word_extraction_index_flat
)
batch_indices = batch_indices.reshape(
    num_words, chars_per_word, 1
)

# Take mode of batch indices in each word
# Trick: mean of non-padding indices is mode of indices as all batch indices are equal
batch_indices = batch_indices.float()
padding_mask = (batch_indices != -1).float()
batch_indices = (batch_indices * padding_mask).sum(dim=1) / padding_mask.sum(dim=1)
batch_indices = batch_indices.long()

batch_indices = batch_indices.expand(-1, best_path_matrix.shape[-1])
morpheme_mask = torch.any(best_path_matrix, dim=1)
batch_indices = torch.masked_select(batch_indices, mask=morpheme_mask)

# Select word encodings for each morpheme
src_word_encodings = src_word_encodings.transpose(0, 1).to(morpheme_encodings.device)
src_word_encodings = src_word_encodings.reshape(src_word_encodings.shape[0], -1)
src_word_encodings = src_word_encodings[batch_indices]
print(src_word_encodings.shape)
print(morpheme_encodings.shape)
