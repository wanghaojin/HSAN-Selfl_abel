import numpy as np
def get_center(features,predict_labels,cluster_num):
    centers = []
    for i in range(cluster_num):
        indices = np.where(predict_labels == i)[0]
        cluster_features = features[indices]
        center = cluster_features.mean(axis=0)
        centers.append(center)

    return np.array(centers)