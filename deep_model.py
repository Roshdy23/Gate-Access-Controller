import cv2
import easyocr
import re

arabic_regex = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'


def run_deep_model(index):
    image_path = f"./images/plate{index}.jpg"
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

    # Save the result to a text file
    with open('output_deepLearning.txt', 'w', encoding='utf-8') as file:
        file.write(plate_text)
