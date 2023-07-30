import torch
import torch.nn as nn

from torch import Tensor
from utils import make_mask_2d
from utils import make_mask_3d
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

        # Mask Separator Indices that Belong to Padding Chars
        index_mask = make_mask_2d(num_morphemes - 1)
        index_mask_padding = torch.ones(
            index_mask.shape[0], 1, dtype=torch.bool, device=index_mask.device
        )
        index_mask = torch.cat([index_mask, index_mask_padding], dim=1)
        index_mask = index_mask.to(scores.device)

        # Select Number of Separators (with the Highest Scores) according to Number of Morphemes
        # Because of Padding, Indices Start with 1 (We Remove 0 Later)
        best_separators = torch.topk(scores, dim=1, k=max_num_morphemes).indices
        best_separators = best_separators + 1
        best_separator_indices = torch.masked_fill(
            best_separators, mask=index_mask, value=0
        )

        # Convert Ordinal Indices to One-Hot Representations
        # e.g. [1, 4] -> [0, 1, 0, 0, 1, 0, 0] corresponds to 3 morphemes
        best_separators_one_hot = torch.zeros(
            num_words, num_chars + 1, dtype=torch.long, device=scores.device
        )
        best_separators_one_hot = torch.scatter(
            best_separators_one_hot, dim=1, index=best_separator_indices, value=1
        )
        # Remove Padding Indices
        best_separators_one_hot = best_separators_one_hot[:, 1:]
        # New Morpheme Starts at Next Character
        # -> Shift before cumsum
        best_separators_one_hot = torch.roll(best_separators_one_hot, shifts=1, dims=1)
        character_to_morpheme = best_separators_one_hot.cumsum(dim=1)

        # Mask Padding Characters (Avoid Appending to Last Morpheme)
        best_path_mask = make_mask_3d(word_lengths, num_morphemes)
        best_path_matrix = one_hot(character_to_morpheme, num_classes=max_num_morphemes)
        best_path_matrix = torch.masked_fill(
            best_path_matrix, mask=best_path_mask, value=0
        )
        best_path_matrix = best_path_matrix.bool()

        return best_path_matrix, best_separators_one_hot

    def get_marginals(
        self, scores: Tensor, word_lengths: Tensor, num_morphemes: Tensor
    ):
        batch_size = scores.shape[0]
        max_num_chars = scores.shape[1]
        max_num_morphemes = torch.max(num_morphemes).cpu().item()

        # Log Sigmoid:
        # Exploit 1 - sigmoid(z) = sigmoid(-z) => log(1 - sigmoid(z)) = log(sigmoid(-z))
        log_sigmoid = self.log_sigmoid(scores)
        log_one_minus_sigmoid = self.log_sigmoid(-scores)

        # Beta Prior defines where to Start Calculation of Beta Scores
        # (Needed for Batch Vectorisation)
        beta_prior = torch.full(
            (batch_size, max_num_chars, max_num_morphemes),
            fill_value=self.neg_inf_val,
            device=scores.device,
        )
        beta_prior[torch.arange(batch_size), word_lengths - 1, num_morphemes - 1] = 0.0

        # Helpers for Padding Alpha and Beta Scores
        padding = torch.full(
            (batch_size, 1), fill_value=self.neg_inf_val, device=scores.device
        )

        def pad_left(score_row: Tensor):
            return torch.cat([padding, score_row[:, :-1]], dim=1)

        def pad_right(score_row: Tensor):
            return torch.cat([score_row[:, 1:], padding], dim=1)

        # Forward-Backward Algorithm
        # Compute Alpha (Forward Scores)
        alpha = [
            torch.full(
                (batch_size, max_num_morphemes),
                fill_value=self.neg_inf_val,
                device=scores.device,
            )
        ]
        alpha[0][:, 0] = 0.0

        for t in range(max_num_chars - 1):
            prev_alpha = alpha[-1]
            alpha_t_stay = prev_alpha + log_one_minus_sigmoid[:, t : t + 1]
            alpha_t_shift = pad_left(prev_alpha) + log_sigmoid[:, t : t + 1]
            alpha_t = torch.logaddexp(alpha_t_stay, alpha_t_shift)
            alpha.append(alpha_t)

        # Compute Beta (Backward Scores)
        beta = [
            torch.full(
                (batch_size, max_num_morphemes),
                fill_value=self.neg_inf_val,
                device=scores.device,
            )
        ]

        for t in range(max_num_chars):
            t = max_num_chars - 1 - t
            next_beta = beta[0]
            beta_t_stay = next_beta + log_one_minus_sigmoid[:, t : t + 1]
            beta_t_shift = pad_right(next_beta) + log_sigmoid[:, t : t + 1]
            beta_t = torch.logaddexp(beta_t_stay, beta_t_shift)
            beta_t = torch.logaddexp(beta_t, beta_prior[:, t])
            beta.insert(0, beta_t)

        # Compute Marginals
        alpha = torch.stack(alpha, dim=1)
        beta = torch.stack(beta[:-1], dim=1)
        z = alpha[torch.arange(batch_size), word_lengths - 1, num_morphemes - 1]
        z = z.reshape(batch_size, 1, 1)

        marginal_mask = make_mask_3d(word_lengths, num_morphemes)
        marginals = (alpha + beta - z).exp()
        marginals = torch.masked_fill(marginals, mask=marginal_mask, value=0.0)

        return marginals

    def _select_relevant_morphemes(
        self, morpheme_encodings: Tensor, num_morphemes: Tensor
    ) -> Tensor:
        """Select Morphemes that do not correspond to Padding"""
        morpheme_encodings = morpheme_encodings.reshape(-1, self.hidden_size)
        morpheme_mask = make_mask_2d(num_morphemes).flatten()
        morpheme_mask = torch.logical_not(morpheme_mask)
        all_morpheme_indices = torch.arange(
            morpheme_encodings.shape[0], device=morpheme_mask.device
        )
        morpheme_indices = torch.masked_select(all_morpheme_indices, mask=morpheme_mask)
        morpheme_encodings = torch.index_select(
            morpheme_encodings, index=morpheme_indices, dim=0
        )
        return morpheme_encodings

    def forward(
        self,
        word_encodings: Tensor,
        word_lengths: Tensor,
        num_morphemes: Tensor = None,
        training: bool = False,
    ):
        # word_encodings: shape [#words x #chars x features]
        batch_size = word_encodings.shape[0]
        max_num_chars = word_encodings.shape[1]

        # Compute Morpheme End Scores
        score_mask = torch.ones(batch_size, max_num_chars, dtype=torch.bool)
        score_mask[:, : max_num_chars - 1] = make_mask_2d(word_lengths - 1)
        score_mask = score_mask.to(word_encodings.device)

        morpheme_end_scores = self.morpheme_end_classifier(word_encodings).squeeze(2)

        # Add Gaussian Noise to Push Scores towards Discreteness
        if training:
            morpheme_end_scores = morpheme_end_scores + torch.randn_like(
                morpheme_end_scores
            )

        morpheme_end_scores = torch.masked_fill(
            morpheme_end_scores, score_mask, value=self.neg_inf_val
        )

        # Get Best Scores
        best_path_matrix, _ = self.get_best_paths(
            morpheme_end_scores, word_lengths, num_morphemes
        )
        best_path_matrix = best_path_matrix.to(morpheme_end_scores.device)
        # best_path_matrix: shape [#words x #chars x #morphemes]

        # In Inference Mode, We can Use Hard Attention Directly
        if not self.training:
            word_encodings = word_encodings.transpose(1, 2)
            morpheme_encodings = torch.bmm(word_encodings, best_path_matrix.float())
            morpheme_encodings = morpheme_encodings.transpose(1, 2)
            morpheme_encodings = self._select_relevant_morphemes(
                morpheme_encodings, num_morphemes
            )
            return morpheme_encodings, best_path_matrix

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
        residual_scores = torch.where(best_path_matrix, marginals - 1.0, marginals)
        residuals = torch.bmm(word_encodings, residual_scores)
        residuals = residuals.transpose(1, 2)
        residuals = residuals.detach()

        # Compute Hard Morpheme Representations
        morpheme_encodings = morpheme_encodings - residuals

        # Select Relevant Morphemes
        morpheme_encodings = self._select_relevant_morphemes(
            morpheme_encodings, num_morphemes
        )
        return morpheme_encodings, best_path_matrix
