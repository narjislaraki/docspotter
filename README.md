# DocSpotter

DocSpotter is a Python library designed to extract specific information from document images by combining text detection and extraction technologies.

## Installation

You can install DocSpotter via pip:

```
pip install docspotter
```

## Dependencies 

Before your tests, you might need to add the following line to your code in order to use tesseract :

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\PATH\TO\tesseract.exe'
```
## Usage

```python
from docspotter import create_craft_detector, detect_text, extract_data, skew_and_extract_text
import cv2

# Load your image
image = cv2.imread('your_image.jpg')

# Create a Craft text detector
craft_detector = create_craft_detector()

# Detect text in the image
text_boxes = detect_text(craft_detector, image)

# Extract text from the image
extracted_text = extract_data(image)

# Skew the image and extract text
values, rois = skew_and_extract_text(image, text_boxes)

```

## Contributing

Contributions are welcome ! If you have any ideas, enhancements or bug fixes, please open an issue or submit a pull request.

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE]() file for details.
