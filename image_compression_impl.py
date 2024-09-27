import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

# Function to load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path)
    return np.array(image)

# Function to perform KMeans clustering for image quantization
def image_compression(image_np, n_colors):
    pixels = image_np.reshape(-1, 3)

    # Apply KMeans clustering to reduce the number of colors
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)

    # Replace each pixel with its corresponding cluster center (compressed color)
    compressed_pixels = kmeans.cluster_centers_[kmeans.labels_].astype(np.uint8)

    # Reshape the compressed pixels back to the original image shape
    compressed_image = compressed_pixels.reshape(image_np.shape)

    return compressed_image

# Function to concatenate and save the original and quantized images side by side
def save_result(original_image_np, quantized_image_np, output_path):
    # Convert NumPy arrays back to PIL images
    original_image = Image.fromarray(original_image_np)
    quantized_image = Image.fromarray(quantized_image_np)
    
    # Get dimensions
    width, height = original_image.size
    
    # Create a new image that will hold both the original and quantized images side by side
    combined_image = Image.new('RGB', (width * 2, height))
    
    # Paste original and quantized images side by side
    combined_image.paste(original_image, (0, 0))
    combined_image.paste(quantized_image, (width, 0))
    
    # Save the combined image
    combined_image.save(output_path)

def __main__():
    # Load and process the image
    image_path = './joe.jpg'
    output_path = './compressed_image.png'
    image_np = load_image(image_path)

    # Perform image quantization using KMeans
    n_colors = 2  # Number of colors to reduce the image to, you may change this to experiment
    quantized_image_np = image_compression(image_np, n_colors)

    # Save the original and quantized images side by side
    save_result(image_np, quantized_image_np, output_path)
