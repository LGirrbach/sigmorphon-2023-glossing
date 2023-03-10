import torch
import torch.nn as nn

from torch import Tensor
from utils import make_mask_2d
from utils import make_mask_3d
from torch_scatter import scatter_sum
from torch.nn.functional import one_hot


class UnsupervisedMorphemeSegmenter(nn.Module):
    neg_inf_val = -1e9

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.morpheme_end_classifier = nn.Linear(self.hidden_size, 1)
        self.log_sigmoid = nn.LogSigmoid()
        self.cross_entropy = nn.CrossEntropyLoss()

    @staticmethod
    def get_best_paths(scores: Tensor, word_lengths: Tensor, num_morphemes: Tensor):
        # scores: shape [#words x #chars]
        num_words, num_chars = scores.shape

        # Compute Character -> Morpheme Mask
        max_num_morphemes = torch.max(num_morphemes).cpu().item()

        index_mask = make_mask_2d(num_morphemes - 1)
        index_mask_padding = torch.ones(
            index_mask.shape[0], 1, dtype=torch.bool, device=index_mask.device
        )
        index_mask = torch.cat([index_mask, index_mask_padding], dim=1)
        index_mask = index_mask.to(scores.device)

        best_separators = torch.topk(scores, dim=1, k=max_num_morphemes).indices
        best_separators = best_separators + 1
        best_separator_indices = torch.masked_fill(
            best_separators, mask=index_mask, value=0
        )

        best_separators_one_hot = torch.zeros(
            num_words, num_chars+1, dtype=torch.long, device=scores.device
        )
        best_separators_one_hot = torch.scatter(
            best_separators_one_hot, dim=1, index=best_separator_indices, value=1
        )
        best_separators_one_hot = best_separators_one_hot[:, 1:]
        best_separators_one_hot = torch.roll(best_separators_one_hot, shifts=1, dims=1)
        character_to_morpheme = best_separators_one_hot.cumsum(dim=1)

        best_path_mask = make_mask_3d(word_lengths, num_morphemes)
        best_path_matrix = one_hot(character_to_morpheme, num_classes=max_num_morphemes)
        best_path_matrix = torch.masked_fill(best_path_matrix, mask=best_path_mask, value=0)
        best_path_matrix = best_path_matrix.bool()

        return best_path_matrix, best_separators_one_hot

    def get_marginals(self, scores: Tensor, word_lengths: Tensor, num_morphemes: Tensor):
        batch_size = scores.shape[0]
        max_num_chars = scores.shape[1]
        max_num_morphemes = torch.max(num_morphemes).cpu().item()

        log_sigmoid = self.log_sigmoid(scores)
        log_one_minus_sigmoid = self.log_sigmoid(-scores)

        beta_prior = torch.full(
            (batch_size, max_num_chars, max_num_morphemes), fill_value=self.neg_inf_val, device=scores.device
        )
        beta_prior[torch.arange(batch_size), word_lengths - 1, num_morphemes - 1] = 0.
        padding = torch.full((batch_size, 1), fill_value=self.neg_inf_val, device=scores.device)

        def pad_left(score_row: Tensor):
            return torch.cat([padding, score_row[:, :-1]], dim=1)

        def pad_right(score_row: Tensor):
            return torch.cat([score_row[:, 1:], padding], dim=1)

        # Forward-Backward Algorithm
        # Compute Alpha (Forward Scores)
        alpha = [torch.full((batch_size, max_num_morphemes), fill_value=self.neg_inf_val, device=scores.device)]
        alpha[0][:, 0] = 0.

        for t in range(max_num_chars - 1):
            prev_alpha = alpha[-1]
            alpha_t_stay = prev_alpha + log_one_minus_sigmoid[:, t:t+1]
            alpha_t_shift = pad_left(prev_alpha) + log_sigmoid[:, t:t+1]
            alpha_t = torch.logaddexp(alpha_t_stay, alpha_t_shift)
            alpha.append(alpha_t)

        # Compute Beta (Backward Scores)
        beta = [torch.full((batch_size, max_num_morphemes), fill_value=self.neg_inf_val, device=scores.device)]

        for t in range(max_num_chars):
            t = max_num_chars - 1 - t
            next_beta = beta[0]
            beta_t_stay = next_beta + log_one_minus_sigmoid[:, t:t+1]
            beta_t_shift = pad_right(next_beta) + log_sigmoid[:, t:t+1]
            beta_t = torch.logaddexp(beta_t_stay, beta_t_shift)
            beta_t = torch.logaddexp(beta_t, beta_prior[:, t])
            beta.insert(0, beta_t)

        # Compute Marginals
        alpha = torch.stack(alpha, dim=1)
        beta = torch.stack(beta[:-1], dim=1)
        z = alpha[torch.arange(batch_size), word_lengths-1, num_morphemes-1]
        z = z.reshape(batch_size, 1, 1)

        marginal_mask = make_mask_3d(word_lengths, num_morphemes)
        marginals = (alpha + beta - z).exp()
        marginals = torch.masked_fill(marginals, mask=marginal_mask, value=0.0)

        return marginals

    def predict_morphemes(self, word_encodings: Tensor, word_lengths: Tensor, morpheme_end_scores: Tensor):
        # word_encodings: shape [#words x #chars x features]
        # morpheme_end_scores: shape [#words x #chars]

        # Get #morphemes per word
        character_mask = make_mask_2d(word_lengths)
        word_encodings_pooled = torch.masked_fill(
            word_encodings, mask=character_mask.unsqueeze(-1), value=0.0
        )
        word_encodings_pooled = word_encodings_pooled.sum(dim=1)
        num_morpheme_scores = self.num_morpheme_classifier(word_encodings_pooled)
        num_morphemes = torch.argmax(num_morpheme_scores, dim=-1)

        # Get Morpheme Separators
        _, is_morpheme_separator = self.get_best_paths(morpheme_end_scores, word_lengths, num_morphemes)

        # Compute Char -> Morpheme Mapping
        character_mask_flat = character_mask.flatten()

        # character_to_morpheme = torch.roll(is_morpheme_separator, shifts=1, dims=1)
        character_to_morpheme = is_morpheme_separator
        character_to_morpheme = character_to_morpheme.flatten()
        character_to_morpheme = character_to_morpheme.cumsum(dim=0)
        character_to_morpheme = character_to_morpheme + 1
        character_to_morpheme = torch.masked_fill(
            character_to_morpheme, mask=character_mask_flat, value=0
        )

        # Compute Morpheme Encodings
        word_encodings = word_encodings.reshape(-1, self.hidden_size)
        morpheme_encodings = scatter_sum(
            word_encodings, index=character_to_morpheme, dim=0
        )
        morpheme_encodings = morpheme_encodings[1:, :]

        # Compute Morpheme -> Word Mapping
        num_morphemes = torch.sum(is_morpheme_separator, dim=1)
        morpheme_mask = make_mask_2d(num_morphemes)
        morpheme_mask_flat = morpheme_mask.flatten()
        morpheme_mask_flat = torch.logical_not(morpheme_mask_flat)

        morpheme_to_word = torch.arange(num_morphemes.shape[0])
        morpheme_to_word = morpheme_to_word.unsqueeze(1)
        morpheme_to_word = morpheme_to_word.expand(morpheme_mask.shape)
        morpheme_to_word = morpheme_to_word.flatten()
        morpheme_to_word = torch.masked_select(
            morpheme_to_word, mask=morpheme_mask_flat
        )
        morpheme_to_word = morpheme_to_word.cpu().tolist()

        return {
            "morpheme_encodings": morpheme_encodings,
            "morpheme_word_mapping": morpheme_to_word,
            "morpheme_extraction_index": character_to_morpheme
        }

    def forward(self, word_encodings: Tensor, word_lengths: Tensor, num_morphemes: Tensor = None):
        # word_encodings: shape [#words x #chars x features]
        batch_size = word_encodings.shape[0]
        max_num_chars = word_encodings.shape[1]

        # Compute Morpheme End Scores
        score_mask = torch.ones(batch_size, max_num_chars, dtype=torch.bool)
        score_mask[:, :max_num_chars-1] = make_mask_2d(word_lengths-1)
        score_mask = score_mask.to(word_encodings.device)

        morpheme_end_scores = self.morpheme_end_classifier(word_encodings).squeeze(2)
        morpheme_end_scores = torch.masked_fill(morpheme_end_scores, score_mask, value=self.neg_inf_val)

        # Get Best Scores
        best_path_matrix, _ = self.get_best_paths(morpheme_end_scores, word_lengths, num_morphemes)
        best_path_matrix = best_path_matrix.to(morpheme_end_scores.device)
        # best_path_matrix: shape [#words x #chars x #morphemes]

        # Get Marginals
        marginals = self.get_marginals(morpheme_end_scores, word_lengths, num_morphemes)
        # marginals: shape [#words x #chars x #morphemes]

        # Compute Soft Morpheme Representations
        word_encodings = word_encodings.transpose(1, 2)
        morpheme_encodings = torch.bmm(word_encodings, marginals)
        # morpheme_encodings: shape [#words x features x #morphemes]
        morpheme_encodings = morpheme_encodings.transpose(1, 2)
        # morpheme_encodings: shape [#words x #morphemes x features]

        # Compute Residuals
        residual_scores = torch.where(best_path_matrix, marginals - 1., marginals)
        residuals = torch.bmm(word_encodings, residual_scores)
        residuals = residuals.transpose(1, 2)
        residuals = residuals.detach()

        # Compute Hard Morpheme Representations
        morpheme_encodings = morpheme_encodings - residuals

        # Select Relevant Morphemes
        morpheme_encodings = morpheme_encodings.reshape(-1, self.hidden_size)
        morpheme_mask = make_mask_2d(num_morphemes).flatten()
        morpheme_mask = torch.logical_not(morpheme_mask)
        all_morpheme_indices = torch.arange(morpheme_encodings.shape[0], device=morpheme_mask.device)
        morpheme_indices = torch.masked_select(all_morpheme_indices, mask=morpheme_mask)
        morpheme_encodings = torch.index_select(morpheme_encodings, index=morpheme_indices, dim=0)

        # Compute Entropy of Margins
        margin_entropy = marginals * torch.log(marginals + 1e-8)
        margin_entropy = 0.001 * margin_entropy.sum()

        # Compute Score Sum Loss
        score_sums = torch.sigmoid(morpheme_end_scores).sum(dim=1)
        score_sum_loss = torch.pow(score_sums - num_morphemes.float() + 1, 2)
        score_sum_loss = -0.001 * score_sum_loss.sum()

        return morpheme_encodings, best_path_matrix
