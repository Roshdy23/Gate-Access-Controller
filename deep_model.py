import cv2
import easyocr
import re

arabic_regex = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'

arabic_to_word = {
    "٠": "zero", "١": "one", "٢": "two", "٣": "three", "٤": "four",
    "٥": "five", "٦": "six", "٧": "seven", "٨": "eight", "٩": "nine",
    "ق": "kaf", "ل": "lam", "و": "waw", "ه": "heh", "م": "mem", "ا": "alf", "ب": "beh",
    "ج": "gem", "د": "dal", "ر": "reh", "ز": "zay", "س": "sen", "ص": "sad", "ط": "tah",
    "ف": "fih", "ع": "ein", "غ": "ghayn", "خ": "kha", "ش": "sheen", "ت": "teh",
    "ظ": "zah", "ة": "hatah", "ي": "yeh", "ى": "yeh", "ن": "non",
}


def is_arabic(char):
    return re.match(arabic_regex, char) is not None


def is_arabic_digit(char):
    return bool(re.search(r'[٠-٩]', char))


def preprocess_image(image_path):
    """Load and preprocess the image."""
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def check_as_one_block(text):
    """
       Check if the text is a valid license plate block.
       A valid block has:
       - At least 1 Arabic digits and At most 4.
       - At least 1 non-digit Arabic characters and At most 3.
       - All it's characters are either Arabic digit ,non-digit Arabic characters, or ' ' char.
    """
    cnt_non_digit_char = sum(1 for char in text if char != ' ' and is_arabic(char) and not is_arabic_digit(char))
    cnt_digit = sum(1 for char in text if is_arabic_digit(char))
    cnt_space = sum(1 for char in text if char == ' ')

    if 1 <= cnt_digit <= 4 and 1 <= cnt_non_digit_char <= 3 and cnt_space >= 1:
        return True
    
    return False


def filter_arabic_text(results):
    """Filter OCR results to extract valid Arabic text."""

    plate_text = ""
    num_is_found = False

    for (bbox, text, prob) in results:

        if not all(is_arabic(char) or char == ' ' for char in text):
            continue

        if text == "مصر":
            continue

        if check_as_one_block(text):
            plate_text = text[::-1]
            break

        if len(text) > 4:
            continue

        if all(is_arabic_digit(char) for char in text):
            plate_text += text
            num_is_found = True
            continue

        if num_is_found:
            plate_text += " " + text[::-1]
            break

    return plate_text


def run_deep_model(image_path):

    image = preprocess_image(image_path)
    reader = easyocr.Reader(['ar'])
    results = reader.readtext(image)
    plate_text = filter_arabic_text(results)

    return plate_text


def map_output_to_plate_format(output_text):
    return '-'.join([arabic_to_word[char] for char in output_text if char in arabic_to_word])


def run_easy_OCR(image_path):
    text = run_deep_model(image_path)
    return map_output_to_plate_format(text)
