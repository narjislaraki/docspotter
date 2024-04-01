from craft_text_detector import Craft
import cv2
import numpy as np
import pytesseract


# Helper functions
def _process_image(image):
    """
    Process the input image by converting it to grayscale and applying dilation.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Processed image.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_image = cv2.dilate(gray_image, kernel, iterations=1)
    return dilated_image

def _resize_and_process_image(image, target_width=300):
    """
    Resize the input image while maintaining the aspect ratio and process it.

    Args:
        image (numpy.ndarray): Input image.
        target_width (int): Width to which the image will be resized.

    Returns:
        numpy.ndarray: Processed image.
    """
    ratio = target_width / float(image.shape[1])
    target_height = int(float(image.shape[0]) * float(ratio))
    resized_img = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
    processed_img = _process_image(resized_img)
    return processed_img


# Main functions
def create_craft_detector(crop_type="poly", cuda=False, text_threshold=0.8, link_threshold=0.4, low_text=0.25):
    """
    Create a Craft text detector object.

    Args:
        crop_type (str): Type of cropping method to be used by Craft detector.
        cuda (bool): Whether to use GPU acceleration with CUDA.
        text_threshold (float): Text confidence threshold.
        link_threshold (float): Link confidence threshold.
        low_text (float): Low text threshold.

    Returns:
        Craft: Craft text detector object.
    """
    craft_detector = Craft(crop_type=crop_type, cuda=cuda, text_threshold=text_threshold,
                           link_threshold=link_threshold, low_text=low_text)
    return craft_detector


def detect_text(craft_detector, image):
    """
    Detect text in the input image using Craft detector.

    Args:
        craft_detector (Craft): Craft text detector object.
        image (numpy.ndarray): Input image.

    Returns:
        list: Detected text boxes.
    """
    prediction_result = craft_detector.detect_text(image)['boxes']
    return prediction_result

def extract_text(image):
    """
    Extract text from the input image using Tesseract OCR.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        str: Extracted text.
    """
    text = pytesseract.image_to_string(image)
    return text

def extract_data(image):
    """
    Extract data from the input image using Tesseract OCR.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        dict: Extracted data.
    """
    prediction_result = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    return prediction_result

def skew_and_extract_text(image, prediction_result):
    """
    Skew the input image based on prediction results ROIs and extract text from it.

    Args:
        image (numpy.ndarray): Input image.
        prediction_result (list): Detected text boxes.

    Returns:
        tuple: Extracted words and corresponding bounding boxes.
    """
    text_rois = prediction_result
    values = []
    rois = []
    for roi in text_rois:
        if len(roi) >= 3:  # at least 3 points for a polygon
            pts1 = np.float32([roi[0], roi[1], roi[2], roi[3]])

            # Calculate width and height based on the roi coordinates
            width = int(np.linalg.norm(roi[1] - roi[0]))
            height = int(np.linalg.norm(roi[2] - roi[1]))

            # Destination coordinates
            pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

            # Calculate the perspective transform matrix and apply it
            M = cv2.getPerspectiveTransform(pts1, pts2)
            warped_roi = cv2.warpPerspective(image, M, (width, height))
            processed_warped = _resize_and_process_image(warped_roi)

            # Extract text from warped_roi using pytesseract
            extracted = extract_text(processed_warped)
            split_extracted = [word.strip() for word in extracted.split()]
            if split_extracted:
                box_points = np.array(roi, dtype=np.int32).reshape(-1, 2)

                # Calculate the minimum bounding rectangle
                rotated_rect = cv2.minAreaRect(box_points)

                # Get vertices of box
                box_vertices = cv2.boxPoints(rotated_rect)
                box_vertices = np.int0(box_vertices)

                # append box_vertices on angled_rois list
                for word in split_extracted:
                    values.append(word)
                    rois.append(box_vertices)

    return values, rois

