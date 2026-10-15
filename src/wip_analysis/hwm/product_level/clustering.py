from typing import Any, Literal, Optional

import numpy as np
import numpy.typing as npt
import polars as pl
from kneed import KneeLocator
from loguru import logger
from rapidfuzz.distance import Levenshtein
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors


def _jaccard_distance(seq1: list[str], seq2: list[str]) -> float:
    """
    Calculate Jaccard distance between two sequences, treating them as sets.

    Jaccard distance ignores the order of elements and only considers the set of unique
    stages present in each sequence. It is defined as:
        Jaccard distance = 1 - Jaccard similarity
        Jaccard similarity = |A ∩ B| / |A ∪ B|

    Args:
        seq1: First recipe sequence (list of stage names).
        seq2: Second recipe sequence (list of stage names).

    Returns:
        Jaccard distance between 0.0 (identical sets) and 1.0 (completely different sets).

    """
    set1 = set(seq1)
    set2 = set(seq2)

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0.0  # Both sequences are empty

    jaccard_similarity = intersection / union
    return 1.0 - jaccard_similarity


def cluster_recipe_seq(
    df: pl.DataFrame,
    clustering_method: Literal["dbscan", "hierarchical", "hdbscan"],
    distance_metric: Literal["levenshtein", "jaccard"] = "jaccard",
    eps: Optional[float] = None,
    min_samples: int = 2,
    min_samples_for_clustering: int = 5,
    primary_cluster_ratio: Optional[float] = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Clustering based on distance metrics between recipe sequences.
    Scope: all recipe sequences of **many different** `product_id`.

    Args:
        df (pl.DataFrame): DataFrame containing recipe sequences of many products. expected columns:
            - `lot_id`: Identifier for the lot.
            - `product_id`: Identifier for the product.
            - `recipe_seq`: The recipe sequence as a list of stage names.
        clustering_method: Clustering algorithm to use. Options:
            - "dbscan": Density-based clustering with noise detection.
            - "hierarchical": Agglomerative hierarchical clustering.
            - "hdbscan": Hierarchical DBSCAN with varying density clusters.
        distance_metric: Distance metric to use for comparing sequences. Options:
            - "levenshtein": Edit distance that considers both element differences and order.
              Suitable when sequence order matters (default).
            - "jaccard": Set-based distance that ignores order and only considers which stages
              are present. Suitable when only the set of stages matters, not their order.
        eps (float, optional): The maximum distance between two samples for one to be considered
            as in the neighborhood of the other. If None, it will be auto-determined.
            Note: For Jaccard distance, eps should be in range [0, 1].
        primary_cluster_ratio (float, optional): The required ratio of samples in the main cluster.
        min_samples (int): The DBSCAN neighborhood threshold used by clustering algorithms
            that require it. Defaults to 2.
        min_samples_for_clustering (int): Minimum number of sequences required to run an actual
            clustering algorithm. If a product has fewer sequences than this value, all sequences
            are grouped into one direct cluster (cluster_id=1). Defaults to 2.

    Returns:
        Dict: product_id -> List of clusters, where each cluster is a dict with:
            - `cluster_id`: The ID of the cluster.
            - `cluster_lots`: A dict mapping lot IDs to their recipe sequences in the cluster
    """
    results: dict[str, list[dict[str, Any]]] = {}
    product_ids = df["product_id"].unique().to_list()

    for pid in product_ids:
        logger.debug("Clustering Over product_id: {}", pid)
        product_df = df.filter(pl.col("product_id") == pid)

        seqs = product_df["recipe_seq"].to_list()
        lot_ids = product_df["lot_id"].to_list()

        n_seqs = len(seqs)
        if n_seqs < max(1, min_samples_for_clustering):
            logger.debug(
                "  Too few lots for clustering for product_id {}: {} < {}. "
                "Grouping into a single direct cluster.",
                pid,
                n_seqs,
                min_samples_for_clustering,
            )
            results[pid] = [
                {
                    "cluster_id": 1,
                    "cluster_lots": dict(zip(lot_ids, seqs, strict=True)),
                }
            ]
            continue

        unique_seqs, seq_to_idx = [], {}
        inv_indices = np.empty(n_seqs, dtype=np.intp)

        for i, s in enumerate(seqs):
            key = tuple(s) if isinstance(s, list) else s
            if key not in seq_to_idx:
                seq_to_idx[key] = len(unique_seqs)
                unique_seqs.append(s)
            inv_indices[i] = seq_to_idx[key]

        n_unique = len(unique_seqs)

        unique_dist_matrix = np.zeros((n_unique, n_unique))
        for i in range(n_unique):
            for j in range(i, n_unique):
                if distance_metric == "levenshtein":
                    distance = Levenshtein.distance(unique_seqs[i], unique_seqs[j])
                elif distance_metric == "jaccard":
                    distance = _jaccard_distance(unique_seqs[i], unique_seqs[j])
                else:
                    raise ValueError(f"Unsupported distance metric: {distance_metric}")
                unique_dist_matrix[i, j] = distance
                unique_dist_matrix[j, i] = distance

        dist_matrix = unique_dist_matrix[np.ix_(inv_indices, inv_indices)]

        if clustering_method == "hierarchical":
            logger.debug("  Using hierarchical clustering.")
            assert primary_cluster_ratio is not None, (
                "main_cluster_ratio must be provided for hierarchical clustering."
            )
            labels = _cluster_hierarchical(dist_matrix, primary_cluster_ratio=primary_cluster_ratio)
        elif clustering_method == "dbscan":
            logger.debug("  Using DBSCAN clustering.")
            pid_eps = eps
            labels = _cluster_dbscan(
                dist_matrix,
                pid_eps=pid_eps,
                min_samples=min_samples,
                primary_cluster_ratio=primary_cluster_ratio,
            )
        elif clustering_method == "hdbscan":
            logger.debug("  Using HDBSCAN clustering.")
            labels = _cluster_hdbscan(dist_matrix)
        else:
            raise ValueError(f"Unsupported clustering method: {clustering_method}")

        if np.all(labels == 0):
            logger.warning("  All samples are classified as noise. Grouping into a single cluster.")
            labels[:] = 1

        # Output
        product_results: list[dict[str, Any]] = []
        unique_labels = set(labels)

        for k in unique_labels:
            indices = np.where(labels == k)[0]
            cluster_lot_info: dict[str, list[str]] = {lot_ids[i]: seqs[i] for i in indices}
            product_results.append(
                {
                    "cluster_id": int(k),
                    "cluster_lots": cluster_lot_info,
                }
            )

            logger.debug("  Cluster {}: {} lots", k, len(indices))
            logger.opt(lazy=True).trace(
                "{}",
                lambda c=cluster_lot_info: "\n".join(
                    f"    Lot {lot_id}: {seq}" for lot_id, seq in c.items()
                ),
            )

        results[pid] = product_results

    return results


def _cluster_dbscan(
    dist_mat: npt.NDArray,
    pid_eps: Optional[float] = None,
    min_samples: int = 2,
    primary_cluster_ratio: Optional[float] = None,
) -> npt.NDArray:
    if pid_eps is None:
        logger.info("    Eps is not set, trying to auto-determine it.")

        neighbours = NearestNeighbors(n_neighbors=min_samples, metric="precomputed")
        neighbours.fit(dist_mat)
        k_distances, _ = neighbours.kneighbors(dist_mat)
        k_distances = np.sort(k_distances[:, -1])

        try:
            kneedle = KneeLocator(
                range(len(k_distances)), k_distances, curve="convex", direction="increasing"
            )
            pid_eps = kneedle.elbow_y
            assert pid_eps is not None, "Keeneedle did not find an elbow."
            logger.info(f"    Auto-determined eps: {pid_eps}")
        except Exception as e:
            logger.error(f"    Failed to auto-determine eps: {e}")
            # Fallback: use a heuristic based on distance matrix statistics
            max_dist = np.max(dist_mat)
            pid_eps = max_dist * 0.5
            logger.warning(f"    Using default eps of 0.5 * max distance: {pid_eps}")
    else:
        logger.info(f"    Using provided eps: {pid_eps}")

    if primary_cluster_ratio is None:
        logger.info("    Primary cluster ratio is not set, use auto-determined eps")
        dbscan = DBSCAN(eps=pid_eps, min_samples=min_samples, metric="precomputed")
        labels = dbscan.fit_predict(dist_mat) + 1
        return labels

    logger.info(f"    Finding eps satisfies: primary cluster ratio >= {primary_cluster_ratio}")
    max_attempts, increase_factor = 20, 1.2

    for attempt in range(max_attempts):
        logger.info(f"    Attempt {attempt + 1}/{max_attempts}, eps={pid_eps:.2f}")

        dbscan = DBSCAN(eps=pid_eps, min_samples=min_samples, metric="precomputed")
        labels = dbscan.fit_predict(dist_mat)

        unique_labels, counts = np.unique(labels, return_counts=True)
        non_noise_mask = unique_labels != -1
        if np.any(non_noise_mask):
            max_cluster_size = np.max(counts[non_noise_mask])
            ratio = max_cluster_size / dist_mat.shape[0]
        else:
            max_cluster_size = 0
            ratio = 0.0
        logger.info(f"    Max cluster size: {max_cluster_size} ({ratio:.2f})")
        if ratio >= primary_cluster_ratio:
            logger.info(f"    Found a valid clustering with eps={pid_eps:.2f}.")
            return labels + 1
        pid_eps *= increase_factor

    logger.warning("    Couldn't satisfy the primary cluster ratio.")
    logger.warning("    Grouping all samples into a single cluster.")
    return np.ones(dist_mat.shape[0], dtype=int)


def _cluster_hierarchical(dist_mat: npt.NDArray, primary_cluster_ratio: float) -> npt.NDArray:
    n_samples = dist_mat.shape[0]

    for k in range(n_samples, 1, -1):
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
        labels = model.fit_predict(dist_mat)
        _, counts = np.unique(labels, return_counts=True)
        if max(counts) / n_samples >= primary_cluster_ratio:
            logger.info(f"    Found a valid clustering with {k} clusters.")
            return labels + 1

    logger.warning("    Couldn't satisfy the primary cluster ratio.")
    logger.warning("    Grouping all samples into a single cluster.")
    return np.ones(n_samples, dtype=int)


def _cluster_hdbscan(dist_mat: npt.NDArray) -> npt.NDArray:
    n_samples = int(dist_mat.shape[0])
    min_cluster_size = max(5, round(n_samples * 0.1))
    hdbscan = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min(2, n_samples),
        metric="precomputed",
        allow_single_cluster=True,
    )
    labels = hdbscan.fit_predict(dist_mat)
    return labels + 1
