Streamlit Image Processing Application
Welcome to the Streamlit Image Processing Application! This tool allows users to upload an image and apply a variety of image processing filters using OpenCV and other image processing libraries. The processed image and its histogram are displayed, and users have the option to download the processed image.

Features
Upload an image (JPG, JPEG, PNG, BMP, GIF)
Display the original image and its histogram
Apply various image processing filters
Display the processed image and its histogram
Option to download the processed image
Image Processing Filters Available
Laplacian Filter
Gaussian Blur
Adaptive Threshold (Grayscale)
Adaptive Threshold (RGB)
Minimum Filter
Maximum Filter
Histogram Stretching
Histogram Equalization
Split Color Channels (Red, Green, Blue)
Swap Color Channels (RG, RB, GB)
Median Blur
Image Complement
Solarization
Average Blur
Prewitt Operator
Sobel Operator
Roberts Operator
Dilation
Erosion
Opening
Closing
Internal Boundary
External Boundary
Morphological Gradient


Usage
Open the app in your web browser (default is http://localhost:8501).
Upload an image using the file uploader.
Select a filter from the dropdown menu.
Adjust the kernel size or other parameters using the sliders, if applicable.
Choose the display mode (RGB, Grayscale, Binary).
View the processed image and its histogram.
Download the processed image using the download button.
Code Structure
app.py: Main application script with all Streamlit components and image processing functions.
requirements.txt: List of Python packages required to run the application.
Dependencies
Streamlit
OpenCV
NumPy
Matplotlib
Pillow
Contributing
Contributions are welcome! If you find any bugs or have suggestions for improvements, please open an issue or create a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgements
Streamlit
OpenCV
NumPy
Matplotlib
Pillow
