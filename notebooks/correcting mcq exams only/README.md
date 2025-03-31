# Bubble Sheet Processing and Answer Extraction

This project processes scanned images of multiple-choice bubble sheets to detect filled bubbles, visualize the detection results, and extract the selected answers for each row.

## How It Works

1. **Preprocessing:**
   - **Image Reading:**  
     The input image is loaded from a given file path.
   - **Grayscale Conversion:**  
     The image is converted to grayscale for easier processing.
   - **Noise Reduction:**  
     Gaussian blur is applied to reduce noise.
   - **Adaptive Thresholding:**  
     Adaptive thresholding is used to convert the image to a binary format, making it easier to detect bubble shapes.
   - **Morphological Operations:**  
     Morphological closing and opening are applied to clean up the binary image.

2. **Bubble Detection:**
   - **Contour Detection:**  
     The script finds contours in the preprocessed image.
   - **Bubble Filtering:**  
     Contours are filtered based on area, circularity (how close to a perfect circle), and aspect ratio to determine which ones likely represent bubbles.
   - **Centroid Calculation:**  
     The centroid (center point) of each bubble is calculated and stored along with other bubble properties.

3. **Organizing Bubbles:**
   - **Row Grouping:**  
     Detected bubbles are sorted by their vertical position and grouped into rows based on their y-coordinates.
   - **Column Sorting:**  
     Within each row, bubbles are sorted by their x-coordinates.

4. **Determining Filled Bubbles:**
   - **Fill Percentage Calculation:**  
     For each bubble, a mask is created, and the average grayscale value inside the bubble is measured. The fill percentage is calculated inversely to the average value (darker bubbles have higher fill percentages).
   - **Thresholding:**  
     A threshold is applied to determine if a bubble is considered filled. The most filled bubble in each row is marked as the selected answer and assigned an option label (e.g., A, B, C, D).

5. **Visualization:**
   - **Drawing Bubbles:**  
     The script draws contours around detected bubbles. Filled bubbles are highlighted in green, while unfilled ones are drawn in red.
   - **Labeling Options:**  
     Each bubble's option label is drawn near its centroid.
   - **Display and Save:**  
     The resulting image is displayed using Matplotlib and saved to disk as `result.jpg`.

6. **Answer Extraction:**
   - **Row-wise Answer Selection:**  
     For each row, the script extracts the option label of the filled bubble (if any) and compiles a list of answers corresponding to each row.

7. **Execution Flow:**
   - **Process Multiple Images:**  
     The `main` function processes multiple bubble sheet images. It prints the detected answers for each row of each image and handles exceptions gracefully.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- SciPy
- Pillow

## Setup and Usage

1. **Install Dependencies:**
   ```bash
   pip install opencv-python numpy matplotlib scipy pillow

2. **Prepare Your Images:**

- Place the bubble sheet images (e.g., correct_mcq_sheet.png, student_mcq_sheet.png) in the working directory.
- Modify the image_paths list in the script if you have different filenames or paths.

3. **Output:**

- The script will display the processed images with detected bubbles and option labels.
- It will save a visual result image (result.jpg) and print the extracted answers for each row in the console.

## Conclusion

This project provides a complete pipeline to preprocess, detect, and interpret multiple-choice bubble sheets, making it useful for automated exam grading and survey processing.