import os
import cv2
import numpy as np
import face_recognition
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances
from collections import defaultdict

IMAGE_DIR = "/home/anusri/sdp/images"
OUTPUT_PLOT = "cluster_plot.png"
OUTPUT_GRID = "cluster_grid.png"

def extract_face_embedding(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image, model="cnn")
    if not face_locations:
        print(f"No face found in {image_path}")
        return None
    embeddings = face_recognition.face_encodings(image, face_locations)
    return embeddings[0] if embeddings else None

def load_embeddings(image_dir):
    features, paths = [], []
    for fname in os.listdir(image_dir):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(image_dir, fname)
            embedding = extract_face_embedding(path)
            if embedding is not None:
                features.append(embedding)
                paths.append(path)
    return np.array(features), paths

def custom_cluster(embeddings, min_group_size=3, distance_threshold=0.6):
    distance_matrix = pairwise_distances(embeddings, metric="euclidean")
    N = len(embeddings)
    labels = [-1] * N
    cluster_id = 0
    assigned = set()

    for i in range(N):
        if i in assigned:
            continue
        neighbors = np.where(distance_matrix[i] < distance_threshold)[0].tolist()
        neighbors = [n for n in neighbors if n != i and n not in assigned]

        if len(neighbors) >= min_group_size - 1:
            group = [i] + neighbors
            for idx in group:
                labels[idx] = cluster_id
                assigned.add(idx)
            cluster_id += 1

    return labels

def save_cluster_plot(image_paths, labels, embeddings):
    embeddings_scaled = StandardScaler().fit_transform(embeddings)
    reduced = PCA(n_components=2).fit_transform(embeddings_scaled)

    unique_labels = sorted(set(labels))
    colors = plt.get_cmap("tab10")

    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        cluster_points = reduced[indices]
        color = "gray" if label == -1 else colors(label % 10)
        label_name = "Noise" if label == -1 else f"Cluster {label}"
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, s=100, label=label_name)

    plt.title("Face Clustering with PCA")
    plt.legend(loc="upper right", fontsize="small", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    plt.close()
    print(f"Plot saved to {OUTPUT_PLOT}")

def save_cluster_grid(image_paths, labels):
    cluster_dict = defaultdict(list)
    for path, label in zip(image_paths, labels):
        if label != -1:
            cluster_dict[label].append(path)

    cluster_rows = []
    max_width = 0
    for cluster in sorted(cluster_dict.keys()):
        images = []
        for path in cluster_dict[cluster]:
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, (100, 100))
                images.append(img)
        if images:
            row = np.hstack(images)
            cluster_rows.append(row)
            max_width = max(max_width, row.shape[1])

    for i in range(len(cluster_rows)):
        if cluster_rows[i].shape[1] < max_width:
            pad_width = max_width - cluster_rows[i].shape[1]
            pad = np.zeros((100, pad_width, 3), dtype=np.uint8)
            cluster_rows[i] = np.hstack([cluster_rows[i], pad])

    if cluster_rows:
        pad = np.zeros((20, max_width, 3), dtype=np.uint8)
        grid = np.vstack([np.vstack([r, pad]) for r in cluster_rows[:-1]] + [cluster_rows[-1]])
        cv2.imwrite(OUTPUT_GRID, grid)
        print(f"Grid saved to {OUTPUT_GRID}")

def main():
    embeddings, image_paths = load_embeddings(IMAGE_DIR)
    if len(embeddings) > 0:
        labels = custom_cluster(embeddings)
        save_cluster_plot(image_paths, labels, embeddings)
        save_cluster_grid(image_paths, labels)
    else:
        print("No face embeddings found.")

if __name__ == "__main__":
    main()
