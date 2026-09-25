import torch


def batch_kmeans_clustering(
    features,
    n_clusters,
    n_init=10,
    max_iter=100,
    tol=1e-4,
    seed=42,
    use_gpu=False,
):
    if not use_gpu:
        try:
            from sklearn.cluster import KMeans
        except ModuleNotFoundError:
            use_gpu = True

        if not use_gpu:
            B, L, D = features.shape
            centroids = torch.zeros(B, n_clusters, D, device=features.device, dtype=features.dtype)

            with torch.no_grad():
                for i in range(B):
                    feat_i = features[i].detach().cpu().numpy()
                    kmeans = KMeans(
                        n_clusters=n_clusters,
                        n_init=n_init,
                        max_iter=max_iter,
                        random_state=seed,
                    )
                    kmeans.fit(feat_i)
                    centroids[i] = torch.from_numpy(kmeans.cluster_centers_).float()

            return centroids.to(features.device)

    B, L, D = features.shape
    device = features.device
    dtype = features.dtype

    features_fp = features.float() if features.dtype not in (torch.float32, torch.float64) else features

    def init_centroids():
        indices = torch.randint(L, (B, n_clusters), device=device, generator=generator)
        return torch.gather(
            features_fp,
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, D),
        )

    def kmeans_single_init():
        centroids = init_centroids()
        for _ in range(max_iter):
            distances = torch.cdist(features_fp, centroids)
            labels = distances.argmin(dim=-1)

            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(B, n_clusters, device=device, dtype=centroids.dtype)
            new_centroids.scatter_add_(
                1, labels.unsqueeze(-1).expand(-1, -1, D), features_fp
            )
            counts.scatter_add_(1, labels, torch.ones(B, L, device=device, dtype=centroids.dtype))

            mask = counts > 0
            new_centroids = torch.where(
                mask.unsqueeze(-1),
                new_centroids / counts.clamp_min(1.0).unsqueeze(-1),
                centroids,
            )

            shift = (centroids - new_centroids).pow(2).sum(dim=-1).mean()
            centroids = new_centroids
            if shift < tol:
                break

        distances = torch.cdist(features_fp, centroids)
        min_dist = distances.min(dim=-1).values
        inertia = min_dist.sum(dim=-1)
        return centroids, inertia

    with torch.no_grad():
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        best_centroids = None
        best_inertia = None
        for _ in range(n_init):
            centroids, inertia = kmeans_single_init()
            if best_centroids is None:
                best_centroids = centroids
                best_inertia = inertia
                continue

            better = inertia < best_inertia
            best_inertia = torch.where(better, inertia, best_inertia)
            best_centroids = torch.where(better.view(B, 1, 1), centroids, best_centroids)

    return best_centroids.to(dtype)
