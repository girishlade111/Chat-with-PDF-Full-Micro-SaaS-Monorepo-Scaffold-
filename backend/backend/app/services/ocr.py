import pytesseract
from PIL import Image


def ocr_image(img_path: str) -> str:
return pytesseract.image_to_string(Image.open(img_path))
