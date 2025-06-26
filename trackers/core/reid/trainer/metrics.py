import torch


def top_k_accuracy(
    logits: torch.Tensor, true_labels: torch.Tensor, top_k: int = 1
) -> torch.Tensor:
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    top_k_predicted_indices = torch.t(
        torch.topk(logits, top_k, dim=1, largest=True, sorted=True)[1]
    )
    correct_matches = torch.eq(
        top_k_predicted_indices,
        true_labels.view(1, -1).expand_as(top_k_predicted_indices),
    )
    num_correct_in_top_k = correct_matches[:top_k].view(-1).float().sum(0, keepdim=True)
    return num_correct_in_top_k.mul_(100.0 / true_labels.size(0))
