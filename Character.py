import cv2
import numpy as np
charWidth = 60
charHeight = 60
class Character:
    def __init__(self, char, template='', width=charWidth, height=charHeight, img=None):
        self.char = char    
        if img is None:
            self.template = cv2.imread(template, 0)
        else:
            self.template = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        self.col_sum = np.zeros(shape=(height, width))
        self.corr = 0
        self.shape = self.template.shape 
        self.resize_and_calculate(height, width)

    def resize_and_calculate(self, height, width):
  
        dim = (height, width)
        self.template = cv2.resize(self.template, dim, interpolation=cv2.INTER_AREA)
        self.shape = self.template.shape 