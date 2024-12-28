import os
import threading
from PIL import Image, ImageOps
import svgwrite
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
from concurrent.futures import ThreadPoolExecutor
import cv2

# Global event to stop the process
stop_event = threading.Event()

def process_chunk(start, end, data, threshold):
    """Process a chunk of the image for binarization using Canny edge detection."""
    chunk = data[start:end]
    # Apply Canny edge detection to detect edges in the chunk
    edges = cv2.Canny(chunk, 100, 200)
    return np.where(edges > threshold, 255, 0)

def png_to_vector_parallel(input_path, output_path, threshold=128, progress_callback=None):
    """
    Convert a PNG image to a vector SVG file with parallel processing using edge detection for better handling of text.
    
    Parameters:
        input_path (str): Path to the input PNG file.
        output_path (str): Path to save the output SVG file.
        threshold (int): Brightness threshold for binarization (0-255).
        progress_callback (function): Callback function to update progress.
    """
    # Open the image and convert to grayscale
    image = Image.open(input_path).convert("L")
    
    # Resize image if it's too large (reduce width to 1000px)
    max_width = 1000
    if image.width > max_width:
        ratio = max_width / float(image.width)
        new_height = int((float(image.height) * float(ratio)))
        image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)  # Use LANCZOS instead of ANTIALIAS
    
    # Convert image to numpy array
    data = np.array(image)

    # Create an SVG drawing
    dwg = svgwrite.Drawing(output_path, profile="tiny")

    # Split the image into chunks for parallel processing
    height, width = data.shape
    chunk_size = height // 4  # Divide the image into 4 chunks for 4 threads
    total_pixels = height * width
    pixel_count = 0

    # Create thread pool for parallel processing
    with ThreadPoolExecutor() as executor:
        # Adjust chunk size for the last part to ensure it does not go out of bounds
        chunk_ranges = [(start, min(start + chunk_size, height)) for start in range(0, height, chunk_size)]
        results = list(executor.map(lambda r: process_chunk(r[0], r[1], data, threshold), chunk_ranges))
        
    # Combine the results from all chunks
    binary_data = np.vstack(results)

    # Trace paths in the binary image
    for y in range(height):
        for x in range(width):
            if stop_event.is_set():  # Check if stop event is set
                print("Process stopped by user.")
                return
            if binary_data[y, x] == 0:  # Black pixel (edge detected)
                dwg.add(dwg.rect(insert=(x, y), size=(1, 1), fill="black"))
            pixel_count += 1
            # Update progress
            if pixel_count % (total_pixels // 100) == 0 and progress_callback:
                progress_callback(pixel_count / total_pixels)

    # Save the SVG file
    dwg.save()
    print(f"SVG saved to {output_path}")

def update_progress(progress):
    progress_bar['value'] = progress * 100
    root.update_idletasks()

def select_file():
    input_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if not input_path:
        return

    input_file_name = os.path.splitext(input_path)[0]
    output_file = f"{input_file_name}.svg"

    try:
        # Show progress bar and stop button
        progress_bar.pack(pady=10)
        stop_button.pack(pady=10)
        stop_button.config(state=tk.NORMAL)

        # Start the conversion in a separate thread
        stop_event.clear()  # Clear any previous stop event
        threading.Thread(target=convert_image, args=(input_path, output_file)).start()
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def convert_image(input_path, output_file):
    try:
        # Start the conversion with progress tracking
        png_to_vector_parallel(input_path, output_file, progress_callback=update_progress)
        if not stop_event.is_set():
            messagebox.showinfo("Success", f"SVG successfully saved to: {output_file}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
    finally:
        # Re-enable the button for further conversions
        button.config(state=tk.NORMAL)
        progress_bar['value'] = 0  # Reset progress bar

        # Hide progress bar and stop button after conversion
        progress_bar.pack_forget()
        stop_button.pack_forget()

def stop_conversion():
    stop_event.set()  # Set the stop event to terminate the process
    button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)  # Disable stop button when process is stopped

def reset_interface():
    label.config(text="Select an image file to convert:")
    button.config(state=tk.NORMAL)

def on_button_click():
    label.config(text="Processing...")
    button.config(state=tk.DISABLED)
    progress_bar['value'] = 0  # Reset progress bar
    root.after(100, select_file)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Image to SVG Converter")
    
    # Center the window on the screen
    window_width = 300
    window_height = 250
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)
    root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

    label = tk.Label(root, text="Select an image file to convert:")
    label.pack(pady=10)

    button = tk.Button(root, text="Select File", command=on_button_click)
    button.pack(pady=10)

    # Progress bar (hidden initially)
    progress_bar = Progressbar(root, orient="horizontal", length=200, mode="determinate")
    
    # Stop button (hidden initially)
    stop_button = tk.Button(root, text="Stop", command=stop_conversion, state=tk.DISABLED)

    root.mainloop()
