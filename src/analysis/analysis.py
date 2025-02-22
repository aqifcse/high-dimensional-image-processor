import numpy as np
from sklearn.decomposition import PCA


def perform_pca(image_data, n_components: int = 2):
    """
    Perform PCA on a 5D image (Z, Time, Channel, X, Y).
    Here we reshape the data so each sample corresponds
    to a flattened (X,Y) pixel vector
    for each Z, Time, and Channel combination.
    The result is reshaped to (Z, Time, Channel, n_components).
    """
    if image_data.ndim != 5:
        raise ValueError(
            "Expected image_data to be 5D (Z, Time, Channel, X, Y)"
        )
    Z, T, C, X, Y = image_data.shape
    # Flatten spatial dimensions: each sample is one (X,Y) pixel vector.
    samples = image_data.reshape(Z * T * C, X * Y)  # shape: (Z*T*C, X*Y)
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(samples)  # shape: (Z*T*C, n_components)
    transformed = transformed.reshape(Z, T, C, n_components)
    return transformed


def calculate_statistics(image_data):
    """
    Calculate basic statistics from image data.
    """
    stats = {
        "mean": float(np.mean(image_data)),
        "std_dev": float(np.std(image_data)),
        "min": float(np.min(image_data)),
        "max": float(np.max(image_data)),
    }
    return stats
