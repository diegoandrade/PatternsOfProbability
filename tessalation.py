from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path  # Corrected import for Path
from scipy.spatial import Delaunay, ConvexHull
from sklearn.cluster import KMeans

def extract_evenly_spaced_black_points(image_path, num_points=100, threshold=10):
    # Load the image and convert to black and white
    image = Image.open(image_path)
    bw_image = image.convert('L')
    image_array = np.array(bw_image)
    
    # Identify black points
    black_points_indices = np.where(image_array < threshold)
    black_points_list = list(zip(black_points_indices[1], black_points_indices[0]))
    
    # Cluster the black points and find the centroids
    black_points_array = np.array(black_points_list)
    kmeans = KMeans(n_clusters=num_points, random_state=0).fit(black_points_array)
    evenly_spaced_points = kmeans.cluster_centers_
    
    # Convert centroids to integer tuples
    evenly_spaced_points_int = [tuple(map(int, point)) for point in evenly_spaced_points]
    return evenly_spaced_points_int

def is_inside_boundary(points, boundary_points):
    hull = ConvexHull(boundary_points)
    hull_path = Path(boundary_points[hull.vertices])
    return hull_path.contains_points(points)

def add_internal_points_and_triangulate(boundary_points, num_internal_points=50):
    # Determine the bounding box
    min_x, max_x = np.min(boundary_points[:, 0]), np.max(boundary_points[:, 0])
    min_y, max_y = np.min(boundary_points[:, 1]), np.max(boundary_points[:, 1])
    
    # Generate random internal points
    np.random.seed(0)
    internal_points = np.random.rand(num_internal_points, 2)
    internal_points[:, 0] = internal_points[:, 0] * (max_x - min_x) + min_x
    internal_points[:, 1] = internal_points[:, 1] * (max_y - min_y) + min_y
    
    # Filter out points that are outside the boundary
    internal_points_inside = internal_points[is_inside_boundary(internal_points, boundary_points)]
    
    # Combine boundary and internal points
    all_points = np.vstack((boundary_points, internal_points_inside))
    
    # Perform Delaunay triangulation
    tri = Delaunay(all_points)
    return all_points, tri.simplices

def plot_triangulation(all_points, simplices):
    # Plot the triangulation
    plt.triplot(all_points[:, 0], all_points[:, 1], simplices)
    plt.plot(all_points[:, 0], all_points[:, 1], 'o')
    plt.show()

def place_bubbles_and_triangulate(points, num_bubbles=1):
    """
    Places bubbles (additional points) at the centroid of existing points and performs Delaunay triangulation.

    Parameters:
        points (np.array): An array of points to be triangulated.
        num_bubbles (int): The number of bubbles (additional points) to add.

    Returns:
        np.array: An array of points including the bubbles.
        np.array: An array of simplices from the Delaunay triangulation.
    """
    # Calculate the centroid of the existing points
    centroid = np.mean(points, axis=0)
    
    # Place bubbles at the centroid
    bubbles = np.tile(centroid, (num_bubbles, 1))
    
    # Combine the original points with the bubbles
    all_points = np.vstack((points, bubbles))
    
    # Perform Delaunay triangulation
    tri = Delaunay(all_points)
    
    return all_points, tri.simplices

# Main execution
image_path = 'MonteCarlo.jpeg'
boundary_points = extract_evenly_spaced_black_points(image_path)
boundary_points_array = np.array(boundary_points)
#all_points, triangles = add_internal_points_and_triangulate(boundary_points_array)
all_points, triangles = place_bubbles_and_triangulate(boundary_points_array)
plot_triangulation(all_points, triangles)
