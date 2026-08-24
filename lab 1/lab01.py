# Lab 01 Tasks Implementation

# Part 1: Environment Setup & Validation
# 1.1 Installation
# Uncomment and run the following lines if you are in a Jupyter Notebook:
# %pip install opencv-python
# %pip install matplotlib
# %pip install pandas numpy

# 1.2 Import & Validate
import cv2
import matplotlib.pyplot as plt

# Patch plt.show to save images instead of hanging in the background
_old_show = plt.show
_fig_counter = 1
def _new_show(*args, **kwargs):
    global _fig_counter
    filename = f"task_output_fig_{_fig_counter}.png"
    plt.savefig(filename)
    print(f"Saved figure to {filename}")
    _fig_counter += 1
plt.show = _new_show

import numpy as np
import pandas as pd

def validate_environment():
    """Validates if the required libraries are imported successfully."""
    print("Environment Validation:")
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"NumPy Version: {np.__version__}")
    print(f"Pandas Version: {pd.__version__}")
    print("Matplotlib is ready.")
    print("-" * 30)

# ==========================================
# Part 2: Python Data Structures & Logic
# ==========================================

# Task 1: Object-Oriented Grocery Manager
class GroceryManager:
    """Manages a grocery list with items, quantities, and prices."""
    
    def __init__(self):
        # Store items as a dictionary where key is item name, 
        # and value is a dictionary with 'quantity' and 'price'.
        self.grocery_list = {}

    def add_item(self, item, quantity, price):
        """Adds an item to the grocery list or updates it if it exists."""
        if item in self.grocery_list:
            self.grocery_list[item]['quantity'] += quantity
            self.grocery_list[item]['price'] = price
        else:
            self.grocery_list[item] = {'quantity': quantity, 'price': price}
        print(f"Added {quantity}x {item} at ${price:.2f} each.")

    def remove_item(self, item):
        """Removes an item from the list, handling errors if the item doesn't exist."""
        try:
            del self.grocery_list[item]
            print(f"Removed {item} from the grocery list.")
        except KeyError:
            print(f"Error: Cannot remove '{item}'. It does not exist in the list.")

    def view_list(self):
        """Displays all items in the grocery list."""
        if not self.grocery_list:
            print("The grocery list is empty.")
            return
        
        print("\n--- Grocery List ---")
        for item, details in self.grocery_list.items():
            qty = details['quantity']
            price = details['price']
            print(f"{item.capitalize()}: {qty} units @ ${price:.2f} each")
        print("--------------------")

    def calculate_total(self):
        """Computes and returns the total cost of all items."""
        total = 0.0
        for details in self.grocery_list.values():
            total += details['quantity'] * details['price']
        print(f"Total Cost: ${total:.2f}")
        return total


# Task 2: Advanced Student Record System
# Build a nested dictionary-based record system
student_records = {
    101: {'Name': 'Alice', 'Major': 'Computer Science', 'Grades': [85, 90, 92]},
    102: {'Name': 'Bob', 'Major': 'Mathematics', 'Grades': [78, 81, 79]},
    103: {'Name': 'Charlie', 'Major': 'Computer Science', 'Grades': [95, 98, 99]},
    104: {'Name': 'Diana', 'Major': 'Physics', 'Grades': [88, 85, 91]}
}

def get_top_student(records):
    """Returns the name of the student with the highest average grade."""
    top_student = None
    highest_avg = -1

    for student_id, info in records.items():
        grades = info['Grades']
        avg_grade = sum(grades) / len(grades) if grades else 0
        if avg_grade > highest_avg:
            highest_avg = avg_grade
            top_student = info['Name']
            
    return top_student

def search_by_major(records, major):
    """Prints all students enrolled in the given major."""
    print(f"\nStudents majoring in {major}:")
    found = False
    for student_id, info in records.items():
        if info['Major'].lower() == major.lower():
            print(f"- {info['Name']}")
            found = True
    if not found:
        print("No students found in this major.")


# ==========================================
# Part 3: Core Computer Vision & Matplotlib
# ==========================================

# NOTE: For these tasks to work, you need to provide valid image paths.
# We will use placeholder paths like 'image1.jpg'. Please replace them with actual images.

# Task 3: Safe Image Loading & RGB Visualization
def safe_image_loading(image_path='Sukuna.jpeg'):
    """Loads an image safely and visualizes it in RGB."""
    # Load an image
    image = cv2.imread(image_path)

    # Implement a safety check
    if image is None:
        print(f"Error: Image not found at path '{image_path}'. Skipping Task 3.")
        return

    # Convert from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Set up plot and hide axes
    plt.figure(figsize=(8, 6))
    plt.axis('off')

    # Display image with formatted title
    plt.imshow(image_rgb)
    plt.title('King of Curses - Ryomen Sukuna', fontsize=16, color='darkred', pad=15)
    plt.show()


# Task 4: Interactive Geometry & Blank Canvas
def draw_geometry():
    """Creates a blank canvas and draws concentric circles and a bounding box."""
    # Create an 800x800 blank black image
    canvas = np.zeros((800, 800, 3), dtype=np.uint8)
    
    center = (400, 400)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    
    # Draw 5 concentric circles of alternating colors
    max_radius = 250
    for i in range(5):
        radius = max_radius - (i * 50)
        cv2.circle(canvas, center, radius, colors[i % len(colors)], thickness=-1)
        
    # Add a bounding box around the outermost circle
    top_left = (center[0] - max_radius, center[1] - max_radius)
    bottom_right = (center[0] + max_radius, center[1] + max_radius)
    cv2.rectangle(canvas, top_left, bottom_right, (255, 255, 255), thickness=5)
    
    # Calculate and print the exact center coordinates
    # Center formula: ((x1 + x2) / 2, (y1 + y2) / 2)
    center_calc = ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2)
    print(f"\nTask 4: The exact center coordinates of the canvas bounding box are {center_calc}")
    
    # Display the result
    plt.figure(figsize=(6, 6))
    plt.imshow(canvas) # Canvas is already BGR and colors given as RGB for pyplot
    plt.axis('off')
    plt.title('Target Board with Bounding Box')
    plt.show()


# Task 5: Advanced Blurring & NumPy ROI Extraction
def blur_and_roi(image_path='high_res.jpg'):
    """Applies heavy blur and extracts ROIs from original and blurred images."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image not found at path '{image_path}'. Skipping Task 5.")
        return
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply heavy Gaussian Blur
    blurred_image = cv2.GaussianBlur(image_rgb, (25, 25), 0)
    
    h, w, _ = image_rgb.shape
    
    # Ensure image is large enough for a 300x300 ROI
    if h < 300 or w < 300:
        print("Image is too small for a 300x300 ROI.")
        return
        
    # Extract 300x300 ROI from the exact center
    cy, cx = h // 2, w // 2
    roi_original = image_rgb[cy - 150:cy + 150, cx - 150:cx + 150]
    roi_blurred = blurred_image[cy - 150:cy + 150, cx - 150:cx + 150]
    
    # Plot side-by-side using Matplotlib subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(roi_original)
    axes[0].axis('off')
    axes[0].set_title('Original ROI', fontsize=14, color='navy')
    
    axes[1].imshow(roi_blurred)
    axes[1].axis('off')
    axes[1].set_title('Blurred ROI', fontsize=14, color='darkorange')
    
    plt.tight_layout()
    plt.show()


# Task 6: Alpha Blending & Typography
def alpha_blending(image_path='background.jpg'):
    """Creates a stylized image caption using alpha blending."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image not found at path '{image_path}'. Skipping Task 6.")
        return
        
    # Resize for consistency
    image = cv2.resize(image, (800, 600))
    overlay = image.copy()
    
    h, w, _ = overlay.shape
    # Calculate bottom 20%
    bottom_20_start = int(h * 0.8)
    
    # Create a solid color overlay (e.g., blue rectangle)
    # BGR format for OpenCV
    cv2.rectangle(overlay, (0, bottom_20_start), (w, h), (255, 0, 0), -1)
    
    # Blend original and overlay
    alpha = 0.5
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    # Write a title over the transparent box
    text = "Stylized Image Caption"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 3
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    
    # Center text in the overlay box
    text_x = (w - text_size[0]) // 2
    text_y = bottom_20_start + (h - bottom_20_start + text_size[1]) // 2
    
    cv2.putText(blended, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(blended_rgb)
    plt.axis('off')
    plt.title('Alpha Blending & Typography')
    plt.show()


# Task 7: Matrix-Based Rotation & Adaptive Thresholding
def thresholding_and_rotation(image_path='document.jpg'):
    """Compares thresholding methods and applies a rotation matrix."""
    # Load a grayscale document or high-contrast image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Image not found at path '{image_path}'. Skipping Task 7.")
        return
        
    # Apply Global Binary Thresholding
    _, global_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Apply Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
                                            
    # Create a rotation matrix: rotate by 45 degrees, scale by 0.8
    h, w = adaptive_thresh.shape
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 0.8)
    
    # Apply rotation
    rotated_adaptive = cv2.warpAffine(adaptive_thresh, rotation_matrix, (w, h))
    
    # Compare side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(global_thresh, cmap='gray')
    axes[0].axis('off')
    axes[0].set_title('Global Thresholding')
    
    axes[1].imshow(adaptive_thresh, cmap='gray')
    axes[1].axis('off')
    axes[1].set_title('Adaptive Thresholding')
    
    axes[2].imshow(rotated_adaptive, cmap='gray')
    axes[2].axis('off')
    axes[2].set_title('Rotated Adaptive (45 deg, 0.8x scale)')
    
    plt.tight_layout()
    plt.show()


# Task 8: Masking with Bitwise Logic
def bitwise_masking(image1_path='img1.jpg', image2_path='img2.jpg'):
    """Applies bitwise logic to cut out and combine regions from two images."""
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    if img1 is None or img2 is None:
        print("Error: One or both images not found for Task 8. Skipping.")
        return
        
    # Resize both to 500x500
    img1 = cv2.resize(img1, (500, 500))
    img2 = cv2.resize(img2, (500, 500))
    
    # Create a binary mask (black image with a thick white circle in center)
    mask = np.zeros((500, 500), dtype=np.uint8)
    cv2.circle(mask, (250, 250), 150, 255, -1)
    
    # Use cv2.bitwise_and to cut out the polygon shape from the first image
    fg = cv2.bitwise_and(img1, img1, mask=mask)
    
    # Use cv2.bitwise_not to invert the mask
    mask_inv = cv2.bitwise_not(mask)
    
    # Use it to cut out the background from the second image
    bg = cv2.bitwise_and(img2, img2, mask=mask_inv)
    
    # Combine them using cv2.bitwise_or
    combined = cv2.bitwise_or(fg, bg)
    
    combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(combined_rgb)
    plt.axis('off')
    plt.title('Masking with Bitwise Logic')
    plt.show()


# Task 9: Statistical Image Profiling with Pandas
def image_profiling(image_path='profile_img.jpg'):
    """Profiles the statistics of an image's color channels using Pandas."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image not found at path '{image_path}'. Skipping Task 9.")
        return
        
    # Convert to RGB (OpenCV loads in BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Flatten the 2D pixel arrays for Red, Green, and Blue channels into 1D arrays
    # Splitting the channels
    r, g, b = cv2.split(image_rgb)
    
    r_flat = r.flatten()
    g_flat = g.flatten()
    b_flat = b.flatten()
    
    # Load these into a Pandas DataFrame
    df = pd.DataFrame({
        'Red': r_flat,
        'Green': g_flat,
        'Blue': b_flat
    })
    
    # Analyze and print the summary using describe()
    print("\nTask 9: Statistical Image Profiling Summary")
    summary = df.describe()
    print(summary)


# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    print("Running Lab 01 Tasks...\n")
    
    validate_environment()
    
    # Task 1 Test
    print("--- Task 1: Grocery Manager ---")
    gm = GroceryManager()
    gm.add_item("Apples", 5, 1.20)
    gm.add_item("Bread", 2, 2.50)
    gm.view_list()
    gm.calculate_total()
    gm.remove_item("Bread")
    gm.remove_item("Milk") # Testing error handling
    
    # Task 2 Test
    print("\n--- Task 2: Student Record System ---")
    top_student = get_top_student(student_records)
    print(f"The top student is: {top_student}")
    search_by_major(student_records, "Computer Science")
    
    # Task 3-9 Tests (Require images)
    print("\n--- Computer Vision Tasks ---")
    print("Please provide valid image paths to see the visualizations.")
    # Uncomment to run the visual tasks if you have images in the directory
    safe_image_loading('Sukuna.jpeg')
    draw_geometry()
    blur_and_roi('high_res.jpg')
    alpha_blending('background.jpg')
    thresholding_and_rotation('document.jpg')
    bitwise_masking('img1.jpg', 'img2.jpg')
    image_profiling('profile_img.jpg')
    
    # We will just run task 4 as it does not require an external image
    print("\nRunning Task 4 (Geometry)...")
    draw_geometry()
    
    print("\nAll implemented tasks executed.")
