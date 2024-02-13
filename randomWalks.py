from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
import random

def extract_evenly_spaced_black_points(image_path, num_points=100, threshold=10):
    image = Image.open(image_path)
    bw_image = image.convert('L')
    image_array = np.array(bw_image)
    black_points_indices = np.where(image_array < threshold)
    black_points_list = list(zip(black_points_indices[1], black_points_indices[0]))
    black_points_array = np.array(black_points_list)
    kmeans = KMeans(n_clusters=num_points, random_state=0).fit(black_points_array)
    evenly_spaced_points = kmeans.cluster_centers_
    evenly_spaced_points_int = np.array([point for point in evenly_spaced_points], dtype=np.int32)
    return evenly_spaced_points_int

def is_inside_boundary(points, boundary_points):
    hull = ConvexHull(boundary_points)
    hull_path = Path(boundary_points[hull.vertices])
    points = np.atleast_2d(points)  # Ensure points is at least 2-dimensional
    return hull_path.contains_points(points)

def generate_random_walk_from_interior(boundary_points):
    center = np.mean(boundary_points, axis=0)
    start_point = center #boundary_points[np.random.randint(len(boundary_points))] * 0.9 + center * 0.1
    
    walk_points = [start_point]
    current_point = start_point
    
    while True:
        angle = random.uniform(0, 2 * np.pi)
        step_size = random.uniform(1, 5)  # Adjust step size as needed
        new_point = current_point + np.array([step_size * np.cos(angle), step_size * np.sin(angle)])
        
        # Correctly format the new_point as a 2D array before checking
        if not is_inside_boundary(new_point[np.newaxis, :], boundary_points):
            break
        
        walk_points.append(new_point)
        current_point = new_point
    
    return np.array(walk_points)

def plot_random_walk(boundary_points, walk_points):
    plt.figure(figsize=(8, 6))
    hull = ConvexHull(boundary_points)
    for simplex in hull.simplices:
        plt.plot(boundary_points[simplex, 0], boundary_points[simplex, 1], 'k-')
    #plt.plot(walk_points[:, 0], walk_points[:, 1], '-d', marker='.')
    plt.plot(walk_points[:, 0], walk_points[:, 1], '-k', marker='.', color='0.2', label='Random Walk')
    plt.axis('equal')
    plt.show()

# Replace 'path_to_image.jpeg' with the path to your actual image file.
image_path = 'MonteCarlo.jpeg'
boundary_points = extract_evenly_spaced_black_points(image_path)
walk_points = generate_random_walk_from_interior(boundary_points)
plot_random_walk(boundary_points, walk_points)
