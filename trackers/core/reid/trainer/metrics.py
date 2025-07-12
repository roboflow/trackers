import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


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


def compute_feature(
    data_loader: DataLoader, feature_extractor: nn.Module, device: torch.device
):
    features, identities, camera_ids = [], [], []
    with torch.inference_mode():
        for data in tqdm(data_loader, total=len(data_loader)):
            image_batch = data["image"].to(device)
            identity_batch = data["identity"]
            camera_id_batch = data["camera_id"]

            extracted_features = feature_extractor.forward_features(image_batch).cpu()
            image_batch = image_batch.cpu()
            torch.cuda.empty_cache()

            features.append(extracted_features)
            identities.extend(identity_batch)
            camera_ids.extend(camera_id_batch)

        features = torch.cat(features, dim=0)
    return features, identities, camera_ids


def compute_distance_matrix(feature_1: torch.Tensor, feature_2: torch.Tensor):
    num_query_features, num_test_features = feature_1.size(0), feature_2.size(0)
    query_squared_norms = (
        torch.pow(feature_1, 2)
        .sum(dim=1, keepdim=True)
        .expand(num_query_features, num_test_features)
    )
    test_squared_norms = (
        torch.pow(feature_2, 2)
        .sum(dim=1, keepdim=True)
        .expand(num_test_features, num_query_features)
        .t()
    )
    distance_matrix = query_squared_norms + test_squared_norms
    distance_matrix.addmm_(feature_1, feature_2.t(), beta=1, alpha=-2)
    return distance_matrix


def evaluate_rank(
    query_dataloader: DataLoader,
    test_dataloader: DataLoader,
    feature_extractor: nn.Module,
    device: torch.device,
    max_rank: int = 50,
):
    query_features, query_identities, query_camera_ids = compute_feature(
        query_dataloader, feature_extractor, device
    )
    test_features, test_identities, test_camera_ids = compute_feature(
        test_dataloader, feature_extractor, device
    )
    distance_matrix = compute_distance_matrix(query_features, test_features)
    num_queries, num_test_samples = distance_matrix.shape

    if num_test_samples < max_rank:
        max_rank = num_test_samples

    ranked_test_indices = np.argsort(distance_matrix, axis=1)
    identity_matches = (
        test_identities[ranked_test_indices] == query_identities[:, np.newaxis]
    ).astype(np.int32)

    query_cmc_curves = []
    query_average_precisions = []
    num_valid_queries = 0.0

    for query_idx in range(num_queries):
        query_person_id = query_identities[query_idx]
        query_camera_id = query_camera_ids[query_idx]

        sorted_test_indices = ranked_test_indices[query_idx]
        same_camera_same_identity_mask = (
            test_identities[sorted_test_indices] == query_person_id
        ) & (test_camera_ids[sorted_test_indices] == query_camera_id)
        valid_test_mask = np.invert(same_camera_same_identity_mask)

        # compute cmc curve
        raw_cmc_curve = identity_matches[query_idx][valid_test_mask]
        if not np.any(raw_cmc_curve):
            continue

        cmc_curve = raw_cmc_curve.cumsum()
        cmc_curve[cmc_curve > 1] = 1

        query_cmc_curves.append(cmc_curve[:max_rank])
        num_valid_queries += 1.0

        # compute average precision
        num_relevant_matches = raw_cmc_curve.sum()
        precision_at_rank = raw_cmc_curve.cumsum()
        precision_at_rank = [x / (i + 1.0) for i, x in enumerate(precision_at_rank)]
        precision_at_rank = np.asarray(precision_at_rank) * raw_cmc_curve
        average_precision = precision_at_rank.sum() / num_relevant_matches
        query_average_precisions.append(average_precision)

    assert num_valid_queries > 0, "Error: all query identities do not appear in test"

    query_cmc_curves = np.asarray(query_cmc_curves).astype(np.float32)
    query_cmc_curves = query_cmc_curves.sum(0) / num_valid_queries
    mean_average_precision = np.mean(query_average_precisions)

    return query_cmc_curves, mean_average_precision
