import torch


def compute_pearson_correlation(cosine_similarities, gold_standard):
    """
    Compute the Pearson correlation coefficient between cosine similarities and gold standard values.

    Args:
        cosine_similarities (torch.Tensor): A tensor containing cosine similarity values.
        gold_standard (torch.Tensor): A tensor containing gold standard values.

    Returns:
        pearson_correlation (float): Pearson correlation coefficient.
    """
    # Ensure input tensors have the same length
    if cosine_similarities.shape != gold_standard.shape:
        raise ValueError("Input tensors must have the same shape")

    # Mean of the two tensors
    mean_cosine = torch.mean(cosine_similarities)
    mean_gold = torch.mean(gold_standard)

    # Compute the covariance
    covariance = torch.sum((cosine_similarities - mean_cosine) * (gold_standard - mean_gold))

    # Compute the standard deviations
    std_cosine = torch.sqrt(torch.sum((cosine_similarities - mean_cosine) ** 2))
    std_gold = torch.sqrt(torch.sum((gold_standard - mean_gold) ** 2))

    # Pearson correlation coefficient
    pearson_correlation = covariance / (std_cosine * std_gold)

    return pearson_correlation.item()


def compute_spearman_correlation(values1, values2):
    """
    Compute the Spearman correlation coefficient between two sets of values.

    Args:
        values1 (torch.Tensor): A tensor containing the first set of values.
        values2 (torch.Tensor): A tensor containing the second set of values.

    Returns:
        spearman_correlation (float): Spearman correlation coefficient.
    """
    # Ensure input tensors have the same length
    if values1.shape != values2.shape:
        raise ValueError("Input tensors must have the same shape")

    # Rank the values using argsort twice to get ranks
    def rank_tensor(tensor):
        _, indices = torch.sort(tensor)  # Sort the tensor
        ranks = torch.empty_like(indices, dtype=torch.float)
        ranks[indices] = torch.arange(1, len(tensor) + 1, dtype=torch.float)
        return ranks

    # Compute ranks for both tensors
    ranks1 = rank_tensor(values1)
    ranks2 = rank_tensor(values2)

    # Compute Pearson correlation on the ranks
    mean_rank1 = torch.mean(ranks1)
    mean_rank2 = torch.mean(ranks2)
    covariance = torch.sum((ranks1 - mean_rank1) * (ranks2 - mean_rank2))
    std_rank1 = torch.sqrt(torch.sum((ranks1 - mean_rank1) ** 2))
    std_rank2 = torch.sqrt(torch.sum((ranks2 - mean_rank2) ** 2))

    spearman_correlation = covariance / (std_rank1 * std_rank2)

    return spearman_correlation.item()