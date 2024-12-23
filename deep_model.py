import cv2
import easyocr
import re

arabic_regex = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'


def run_deep_model(path):
    image_path = path
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    reader = easyocr.Reader(['ar'])  # 'ar' for Arabic

    results = reader.readtext(image)
    plate_text = ""
    num = False
    for (bbox, text, prob) in results:
        # Draw the bounding box
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

        cnt_non_space = 0
        for char in text:
            if char == ' ': continue
            cnt_non_space += 1
        # Use regular expression to find Arabic digits
        if (cnt_non_space > 4 or text == "مصر"): continue
        match = True
        for i in range(len(text)):
            match = match and (re.search(r'[٠-٩]+', text[i]))
        if match:
            plate_text += text
            num = True
            continue

        if (num):
            plate_text += " " + text
            break

    # Extract and display the detected text
    # plate_text = ' '.join([text for (_, text, _) in results])
    print("Detected License Plate:", plate_text)

    return plate_text


arabic_to_word = {
    "٠": "zero", "١": "one", "٢": "two", "٣": "three", "٤": "four",
    "٥": "five", "٦": "six", "٧": "seven", "٨": "eight", "٩": "nine",
    "ق": "qaf", "ل": "lam", "و": "waw", "ه": "heh", "م": "meem", "ا": "alif", "ب": "beh",
    "ج": "jeem", "د": "dal", "ر": "reh", "ز": "zay", "س": "seen", "ص": "sad", "ط": "tah",
    "ف": "feh", "ع": "ain", "غ": "ghayn", "خ": "kha", "ش": "sheen", "ص": "sad", "ت": "teh",
    "ظ": "zah", "ة": "hatah", "ي": "yeh", "ن": "noon", "ف": "feh", "ج": "jeem"
}


def map_output_to_plate_format(output_text):
    formatted_text = []

    for char in output_text:
        if (not char in arabic_to_word): continue
        formatted_text.append(arabic_to_word[char])

    # Return the space-separated string
    return '-'.join(formatted_text)


def run_easy_OCR(image_path):
    text = run_deep_model(image_path)
    return map_output_to_plate_format(text)
