from detect_characters import GateAccessController
from plateDetection import plateDetection
from preProcessing import imagePreprocessing
from segmentPlate import segment_plate

features = []
labels = []
controller = GateAccessController()


# UNCOMMENT THIS TO TRAIN THE MODEL
# controller.train_model()

def run_OCR(input_image_path):
    preprocessed_image, original_image = imagePreprocessing(input_image_path)

    # buildCharacterDB(features, labels)

    license_plate = plateDetection(preprocessed_image, original_image)
    plate_text = ""

    if license_plate is not None:
        chars = segment_plate(license_plate)
        for i, char in enumerate(chars):
            prediction = controller.predict_plate_text(char)
            print(f"Character {i}: {prediction}")

            plate_text += prediction[0]
            if i != len(chars) - 1:
                plate_text += "-"
    else:
        print("No license plate detected.")

    return plate_text
