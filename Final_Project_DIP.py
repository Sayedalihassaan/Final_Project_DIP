import streamlit as st
import cv2
import numpy as np

#Python Imaging Library (PIL)
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt




# Image processing functions
def plot_histogram(image, title):
    plt.figure()
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    
    # Check if the image is RGB
    if len(image.shape) == 3:
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
    else:
        # If the image is grayscale
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.plot(hist, color='k')
        plt.xlim([0, 256])
    
    plt.grid(True)
    return plt

def display_with_histogram_original_image(original_image, title):
    original_array = np.array(original_image)
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    
    axes.imshow(original_array)
    axes.set_title("Original Image")
    axes.axis('off')
    
    hist_plt = plot_histogram(original_array, f"{title} Histogram")
    plt.show()
    st.pyplot(hist_plt)


def display_with_histogram_processed_image(processed_image, title):
    processed_array = np.array(processed_image)
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))

    axes.imshow(processed_array)
    axes.set_title("Processed Image")
    axes.axis('off')
    
    hist_plt = plot_histogram(processed_image, f"{title} Histogram")
    plt.show()
    st.pyplot(hist_plt)





def complement_image(img):
    return 255 - img


def stretch_histogram(img_gray): 
    min_val = np.min(img_gray)
    max_val = np.max(img_gray)
    stretched = (img_gray - min_val) * (255.0 / (max_val - min_val))
    return stretched.astype(np.uint8)  


def histogram_equalization(img_gray): # contrast
    return cv2.equalizeHist(img_gray)

def split_color_channels(img):
    b, g, r = cv2.split(img)
    zero_channel = np.zeros_like(b)
    return {
        "red": cv2.merge([r, zero_channel, zero_channel]),
        "green": cv2.merge([zero_channel, g, zero_channel]),
        "blue": cv2.merge([zero_channel, zero_channel, b]),
    }

# averaging filter 
def avg_blur(img, kernel_size=5):
    return cv2.blur(img, (kernel_size, kernel_size))


# Weighted Average Filter 
def g_blur(img, kernel_size=5):                           # sigma
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)



def adaptive_thresholding(gray_image, kernel_size=3):
    old_threshold = 128
    new_threshold = 0
    while True:
 
        blurred_image = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
        
        
        ret, binary1 = cv2.threshold(blurred_image, old_threshold, 255, cv2.THRESH_BINARY)
        ret, binary2 = cv2.threshold(blurred_image, old_threshold, 255, cv2.THRESH_BINARY_INV)
        
        
        m1 = np.mean(gray_image[binary1 > 0])
        m2 = np.mean(gray_image[binary2 > 0])
        
        # Update threshold value
        new_threshold = 0.5 * (m1 + m2)  # 75
        
        # Check convergence 
        if abs(new_threshold - old_threshold) > 1:
            break
        
        old_threshold = new_threshold
    
    
    ret, binary = cv2.threshold(gray_image, new_threshold, 255, cv2.THRESH_BINARY)
    return binary


def adaptive_thresholding_rgb(rgb_image, kernel_size=3):
    # Split the RGB image into its respective channels
    channels = cv2.split(rgb_image)
    
    # Apply adaptive thresholding to each channel
    thresholded_channels = [adaptive_thresholding(channel, kernel_size) for channel in channels]
    
    # Merge the thresholded channels back into a single image
    thresholded_rgb = cv2.merge(thresholded_channels)
    
    return thresholded_rgb



def laplacian_filter(img, kernel_size=3):
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)
    laplacian_8u = cv2.convertScaleAbs(laplacian)
    return laplacian_8u 


def solarize(img_gray, threshold=128):
    solarized_image = img_gray.copy()     # X = 255 - X
    solarized_image[solarized_image < threshold] = 255 - solarized_image[solarized_image < threshold]
    return solarized_image



def min_filter(img_gray, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8) 
    return cv2.erode(img_gray, kernel, iterations=1)



def max_filter(img_gray, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(img_gray, kernel, iterations=1)


def swap_color_channels(img):
    b, g, r = cv2.split(img)
    return {                #B   G   R
        "swap_rg": cv2.merge([b, r, g]),  # Swap red and green channels
        "swap_rb": cv2.merge([r, g, b]),  # Swap red and blue channels
        "swap_gb": cv2.merge([g, b, r]),  # Swap green and blue channels
    }


def median_blur(img, kernel_size=5):
    return cv2.medianBlur(img, kernel_size)


def roberts_operator(img):
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    # Roberts cross kernels
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    # Convolve with the Roberts kernels
    img_x = cv2.filter2D(img_gray,-1 , kernel_x)
    img_y = cv2.filter2D(img_gray, -1, kernel_y)

    # Compute the magnitude of the gradients
    img_roberts = np.sqrt(np.square(img_x) + np.square(img_y))

    # Normalize to 8-bit range
    img_roberts = np.clip(img_roberts, 0, 255).astype(np.uint8)

    return cv2.convertScaleAbs(img_roberts)   # uint8 range [0, 255]



def prewitt_operator(img):
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    # Prewitt kernels for horizontal and vertical gradients
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)

    # Convolve with Prewitt kernels
    img_x = cv2.filter2D(img_gray, -1, kernel_x)
    img_y = cv2.filter2D(img_gray, -1, kernel_y)

    # Calculate the magnitude
    img_prewitt = np.sqrt(np.square(img_x) + np.square(img_y))

    # Replace NaN or infinite values with zero to ensure valid data
    img_prewitt[np.isnan(img_prewitt)] = 0
    img_prewitt[np.isinf(img_prewitt)] = 0

    # Convert to 8-bit unsigned integers
    img_prewitt_8bit = np.clip(img_prewitt, 0, 255).astype(np.uint8)

    # Return the converted scale absolute
    return cv2.convertScaleAbs(img_prewitt_8bit)



def sobel_operator(img):
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    # Apply Sobel kernels for gradients in X and Y
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in X
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in Y

    # Calculate the magnitude
    sobel_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))

    # Replace NaN or infinite values with zero
    sobel_magnitude[np.isnan(sobel_magnitude)] = 0 #Replaces any NaN (Not a Number) or 
                                                    #infinite values in the magnitude array with zero to ensure valid pixel values.
    sobel_magnitude[np.isinf(sobel_magnitude)] = 0

    # Convert to 8-bit unsigned integers and return
    return cv2.convertScaleAbs(np.clip(sobel_magnitude, 0, 255))



def dilate_image(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)

def erode_image(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(img, kernel, iterations=1)

def opening_image(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def closing_image(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)



def internal_boundary(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_img = cv2.erode(img, kernel, iterations=1)
    internal_boundary = img - eroded_img
    return internal_boundary


def external_boundary(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_img = cv2.dilate(img, kernel, iterations=1)
    external_boundary = dilated_img - img
    return external_boundary


def morphological_gradient(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_img = cv2.dilate(img, kernel, iterations=1)
    eroded_img = cv2.erode(img, kernel, iterations=1)
    morph_gradient = dilated_img - eroded_img
    return morph_gradient




def ensure_grayscale(image):
    """Ensure the input image is grayscale."""
    if len(image.shape) == 3:  # 3-channel color image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:  # Already grayscale
        return image



st.title("Image Processing with Streamlit")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "gif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    st.image(image, caption='Uploaded Image', use_column_width=True)


    display_with_histogram_original_image(img_cv, "orignal")


    st.write("Select a filter to apply to the image:")

    display_mode = st.radio("Display mode", ["RGB", "Grayscale", "Binary"])

    
 

    filter_choice = st.selectbox("Select a filter", [
        "Laplacian",
        "Gaussian Blur",
        "Adaptive Threshold",
        "adaptive_thresholding_rgb",
        "Min Filter",
        "Max Filter",
        "Stretch Histogram",
        "Histogram Equalization",
        "Split Color Channels",
        "Swap Color Channels",
        "Median Blur",
        "Complement",
        "Solarization",
        "Average Filter",
        "Prewitt Operator",
         "Sobel Operator",
        "Roberts Operator",
        "dilate image",
        "erode image",
        "opening image",
        "closing image",
        "internal boundary",
        "external boundary",
        "morphological gradient"

    ])

    if filter_choice == "Roberts Operator":
        processed_image = roberts_operator(img_cv)
    

## Applying the chosen filter

    if filter_choice == "Laplacian":
        kernel_size = st.slider("Kernel size for Laplacian", 3, 7, step=2)
        processed_image = laplacian_filter(img_cv, kernel_size)

    elif filter_choice == "Sobel Operator":
        processed_image = sobel_operator(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY))


    elif filter_choice == "Prewitt Operator":
        processed_image = prewitt_operator(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY))


    elif filter_choice == "Gaussian Blur":
        kernel_size = st.slider("Kernel size for Gaussian Blur", 3, 11, step=2)
        processed_image = g_blur(img_cv, kernel_size)


    elif filter_choice == "Complement":
        processed_image = complement_image(img_cv)


    elif filter_choice == "Adaptive Threshold":
        kernel_size = st.slider("Kernel size for Adaptive Threshold", 3, 11, step=2)
        processed_image = adaptive_thresholding(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) , kernel_size)


    elif filter_choice == "adaptive_thresholding_rgb":
        kernel_size = st.slider("Kernel size for Adaptive Threshold", 3, 11, step=2)
        processed_image = adaptive_thresholding_rgb(img_cv , kernel_size)


    elif filter_choice == "Min Filter":
        kernel_size = st.slider("Kernel size for Min Filter", 3, 11, step=2)
        processed_image = min_filter(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), kernel_size)


    elif filter_choice == "dilate image":
        kernel_size = st.slider("Kernel size for dilate image", 3, 11, step=2)
        processed_image = dilate_image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), kernel_size)

    elif  filter_choice== "erode image":
        kernel_size = st.slider("Kernel size for erode image", 3, 11, step=2)
        processed_image = erode_image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), kernel_size)

    elif  filter_choice== "opening image":
        kernel_size = st.slider("Kernel size for opening image", 3, 11, step=2)
        processed_image = opening_image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), kernel_size)

    elif  filter_choice== "closing image":
        kernel_size = st.slider("Kernel size for closing image", 3, 11, step=2)
        processed_image = closing_image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), kernel_size)

    elif  filter_choice== "internal boundary":
        kernel_size = st.slider("Kernel size for internal boundary", 3, 11, step=2)
        processed_image = internal_boundary(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), kernel_size)

    elif  filter_choice== "external boundary":
        kernel_size = st.slider("Kernel size for external boundary", 3, 11, step=2)
        processed_image = external_boundary(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), kernel_size)

    elif  filter_choice== "morphological gradient":
        kernel_size = st.slider("Kernel size for morphological gradient", 3, 11, step=2)
        processed_image = morphological_gradient(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), kernel_size)



    elif filter_choice == "Max Filter":
        kernel_size = st.slider("Kernel size for Max Filter", 3, 11, step=2)
        processed_image = max_filter(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), kernel_size)


    elif filter_choice == "Stretch Histogram":
        processed_image = stretch_histogram(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY))


    elif filter_choice == "Histogram Equalization":
        processed_image = histogram_equalization(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY))


    elif filter_choice == "Solarization":
        threshold = st.slider("Threshold for Solarization", 0, 255, 128)
        processed_image = solarize(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), threshold)


    elif filter_choice == "Split Color Channels":
        color_channel = st.radio("Select color channel", ["red", "green", "blue"])
        split_channels = split_color_channels(img_cv)
        processed_image = split_channels[color_channel]


    elif filter_choice == "Swap Color Channels":
        channel_swap = st.radio("Select channel swap", ["swap_rg", "swap_rb", "swap_gb"])
        channel_swaps = swap_color_channels(img_cv)
        processed_image = channel_swaps[channel_swap]


    elif filter_choice == "Median Blur":
        kernel_size = st.slider("Kernel size for Median Blur", 3, 11, step=2)
        processed_image = median_blur(img_cv, kernel_size)


    elif filter_choice == "Average Filter":
        kernel_size = st.slider("Kernel size for Average Filter", 3, 11, step=2)
        processed_image = avg_blur(img_cv, kernel_size)


    if display_mode == "Grayscale":
        processed_image = ensure_grayscale(processed_image)  # Ensure it's grayscale
        

    if  display_mode == "Binary":
        processed_image = cv2.threshold(ensure_grayscale(processed_image), 128, 255, cv2.THRESH_BINARY)[1]
    


    # Display the processed image
    st.image(processed_image, caption=f'Image after {filter_choice}', use_column_width=True)
    
    display_with_histogram_processed_image(processed_image, 'processed')


    # Option to download the processed image
    buffered = BytesIO()
    Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)).save(buffered, format="PNG")
    st.download_button(
        label="Download Image",
        data=buffered.getvalue(),
        file_name="processed_image.png",
        mime="image/png"
    )

### END