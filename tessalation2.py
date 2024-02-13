from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.spatial import Delaunay, ConvexHull
from sklearn.cluster import KMeans

def extract_evenly_spaced_black_points(image_path, num_points=100, threshold=10):
    image = Image.open(image_path)
    bw_image = image.convert('L')
    image_array = np.array(bw_image)
    black_points_indices = np.where(image_array < threshold)
    black_points_list = list(zip(black_points_indices[1], black_points_indices[0]))
    black_points_array = np.array(black_points_list)
    kmeans = KMeans(n_clusters=num_points, random_state=0).fit(black_points_array)
    evenly_spaced_points = kmeans.cluster_centers_
    evenly_spaced_points_int = [tuple(map(int, point)) for point in evenly_spaced_points]
    return evenly_spaced_points_int

def is_inside_boundary(points, boundary_points):
    hull = ConvexHull(boundary_points)
    hull_path = Path(boundary_points[hull.vertices])
    return hull_path.contains_points(points)

def add_internal_points_and_triangulate(boundary_points, num_internal_points=50):
    min_x, max_x = np.min(boundary_points[:, 0]), np.max(boundary_points[:, 0])
    min_y, max_y = np.min(boundary_points[:, 1]), np.max(boundary_points[:, 1])
    np.random.seed(0)
    internal_points = np.random.rand(num_internal_points, 2)
    internal_points[:, 0] = internal_points[:, 0] * (max_x - min_x) + min_x
    internal_points[:, 1] = internal_points[:, 1] * (max_y - min_y) + min_y
    internal_points_inside = internal_points[is_inside_boundary(internal_points, boundary_points)]
    all_points = np.vstack((boundary_points, internal_points_inside))
    tri = Delaunay(all_points)
    return all_points, tri.simplices

def plot_triangulation(all_points, simplices):
    plt.triplot(all_points[:, 0], all_points[:, 1], simplices)
    plt.plot(all_points[:, 0], all_points[:, 1], 'o')
    plt.axis('equal')  # Adjust the axis to equal scale for a better view of the mesh
    plt.show()

# Main execution flow for mesh generation
def generate_mesh(image_path, num_boundary_points=100, num_internal_points=50, threshold=10):
    boundary_points = extract_evenly_spaced_black_points(image_path, num_boundary_points, threshold)
    boundary_points_array = np.array(boundary_points)
    all_points, triangles = add_internal_points_and_triangulate(boundary_points_array, num_internal_points)
    plot_triangulation(all_points, triangles)

# Replace 'path_to_image.jpeg' with the path to your actual image file.
image_path = 'MonteCarlo.jpeg'
generate_mesh(image_path)
