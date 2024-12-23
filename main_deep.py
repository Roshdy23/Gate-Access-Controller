import cv2
import easyocr
import matplotlib.pyplot as plt
import argparse
from deep_model import run_deep_model

parser = argparse.ArgumentParser()
parser.add_argument("index", type=int)
args = parser.parse_args()

# Run the deep model with the provided index
run_deep_model(args.index)
