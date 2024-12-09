import cv2
import numpy as np
from Character import Character
from HOG import HOG
def load_and_process_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is not None and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img, (60, 60)) 
    return img_resized

def buildCharacterDB(features, labels):
    
    def calc_features_labels(path, label):
        img = cv2.imread(path)
        if img is not None and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img, (60, 60)) 
        hog = HOG()
        features.append(hog.compute_hog_features(img_resized))
        labels.append(label)
 

   
    calc_features_labels('images/data-set/alf_1.jpg', "alf")
    calc_features_labels('images/data-set/alf_2.jpg', "alf")
    calc_features_labels('images/data-set/alf_3.jpg', "alf")
    calc_features_labels('images/data-set/alf_4.jpg', "alf")
    calc_features_labels('images/data-set/alf_5.jpg', "alf")
    calc_features_labels('images/data-set/alf_6.jpg', "alf")
    calc_features_labels('images/data-set/alf_7.png', "alf")
    calc_features_labels('images/data-set/alf_8.jpg', "alf")
    calc_features_labels('images/data-set/alf_9.jpg', "alf")
    calc_features_labels('images/data-set/alf_10.jpg', "alf")
    calc_features_labels('images/data-set/beh_1.jpg', "beh")
    calc_features_labels('images/data-set/beh_2.jpg', "beh")
    calc_features_labels('images/data-set/beh_3.jpg', "beh")
    calc_features_labels('images/data-set/beh_4.jpg', "beh")
    calc_features_labels('images/data-set/beh_5.jpg', "beh")
    calc_features_labels('images/data-set/dal_1.jpg', "dal")
    calc_features_labels('images/data-set/dal_2.jpg', "dal")
    calc_features_labels('images/data-set/dal_3.jpg', "dal")
    calc_features_labels('images/data-set/dal_4.jpg', "dal")
    calc_features_labels('images/data-set/dal_5.jpg', "dal")
    calc_features_labels('images/data-set/dal_6.jpg', "dal")
    calc_features_labels('images/data-set/ein_1.png', "ein")
    calc_features_labels('images/data-set/ein_2.png', "ein")
    calc_features_labels('images/data-set/ein_3.png', "ein")
    calc_features_labels('images/data-set/fih_1.jpg', "fih")
    calc_features_labels('images/data-set/fih_2.png', "fih")
    calc_features_labels('images/data-set/gem_1.jpg', "gem")
    calc_features_labels('images/data-set/gem_2.jpg', "gem")
    calc_features_labels('images/data-set/gem_3.jpg', "gem")
    calc_features_labels('images/data-set/gem_4.jpg', "gem")
    calc_features_labels('images/data-set/gem_5.jpg', "gem")
    calc_features_labels('images/data-set/heh_1.jpg', "heh")
    calc_features_labels('images/data-set/heh_2.png', "heh")
    calc_features_labels('images/data-set/heh_3.png', "heh")
    calc_features_labels('images/data-set/kaf_1.jpg', "kaf")
    calc_features_labels('images/data-set/kaf_2.jpg', "kaf")
    calc_features_labels('images/data-set/kaf_3.jpg', "kaf")
    calc_features_labels('images/data-set/kaf_4.jpg', "kaf")
    calc_features_labels('images/data-set/kaf_5.jpg', "kaf")
    calc_features_labels('images/data-set/kaf_6.jpg', "kaf")
    calc_features_labels('images/data-set/kaf_7.png', "kaf")
    calc_features_labels('images/data-set/lam_1.png', "lam")
    calc_features_labels('images/data-set/lam_2.png', "lam")
    calc_features_labels('images/data-set/lam_3.jpg', "lam")
    calc_features_labels('images/data-set/mem_1.jpg', "mem")
    calc_features_labels('images/data-set/mem_2.jpg', "mem")
    calc_features_labels('images/data-set/mem_3.jpg', "mem")
    calc_features_labels('images/data-set/mem_4.jpg', "mem")
    calc_features_labels('images/data-set/mem_5.jpg', "mem")
    calc_features_labels('images/data-set/non_1.png', "non")
    calc_features_labels('images/data-set/non_2.png', "non")
    calc_features_labels('images/data-set/reh_1.png', "reh")
    calc_features_labels('images/data-set/reh_2.jpg', "reh")
    calc_features_labels('images/data-set/reh_3.jpg', "reh")
    calc_features_labels('images/data-set/reh_4.jpg', "reh")
    calc_features_labels('images/data-set/reh_5.jpg', "reh")
    calc_features_labels('images/data-set/sad_1.jpg', "sad")
    calc_features_labels('images/data-set/sad_2.jpg', "sad")
    calc_features_labels('images/data-set/sad_3.jpg', "sad")
    calc_features_labels('images/data-set/sad_4.jpg', "sad")
    calc_features_labels('images/data-set/sad_5.jpg', "sad")
    calc_features_labels('images/data-set/sad_6.jpg', "sad")
    calc_features_labels('images/data-set/sen_1.jpg', "sen")
    calc_features_labels('images/data-set/sen_2.png', "sen")
    calc_features_labels('images/data-set/tah_1.jpg', "tah")
    calc_features_labels('images/data-set/tah_2.jpg', "tah")
    calc_features_labels('images/data-set/tah_3.jpg', "tah")
    calc_features_labels('images/data-set/waw_1.jpg', "waw")
    calc_features_labels('images/data-set/waw_2.jpg', "waw")
    calc_features_labels('images/data-set/waw_3.jpg', "waw")
    calc_features_labels('images/data-set/waw_4.jpg', "waw")
    calc_features_labels('images/data-set/waw_5.jpg', "waw")
    calc_features_labels('images/data-set/waw_6.jpg', "waw")
    calc_features_labels('images/data-set/waw_7.jpg', "waw")
    calc_features_labels('images/data-set/waw_8.jpg', "waw")
    calc_features_labels('images/data-set/waw_9.jpg', "waw")
    calc_features_labels('images/data-set/yeh_1.jpg', "yeh")
    calc_features_labels('images/data-set/yeh_2.jpg', "yeh")


    # Numbers
    calc_features_labels('images/data-set/one_1.jpg', "1")
    calc_features_labels('images/data-set/one_2.jpg', "1")
    calc_features_labels('images/data-set/one_3.jpg', "1")
    calc_features_labels('images/data-set/one_4.jpg', "1")
    calc_features_labels('images/data-set/one_5.jpg', "1")
    calc_features_labels('images/data-set/two_1.jpg', "2")
    calc_features_labels('images/data-set/two_2.jpg', "2")
    calc_features_labels('images/data-set/two_3.jpg', "2")
    calc_features_labels('images/data-set/two_4.jpg', "2")
    calc_features_labels('images/data-set/two_5.jpg', "2")
    calc_features_labels('images/data-set/three_1.jpg', "3")
    calc_features_labels('images/data-set/three_2.jpg', "3")
    calc_features_labels('images/data-set/three_3.jpg', "3")
    calc_features_labels('images/data-set/three_4.jpg', "3")
    calc_features_labels('images/data-set/three_5.jpg', "3")
    calc_features_labels('images/data-set/four_1.jpg', "4")
    calc_features_labels('images/data-set/four_2.jpg', "4")
    calc_features_labels('images/data-set/four_3.jpg', "4")
    calc_features_labels('images/data-set/four_4.jpg', "4")
    calc_features_labels('images/data-set/four_5.jpg', "4")
    calc_features_labels('images/data-set/five_1.jpg', "5")
    calc_features_labels('images/data-set/five_2.jpg', "5")
    calc_features_labels('images/data-set/five_3.jpg', "5")
    calc_features_labels('images/data-set/five_4.jpg', "5")
    calc_features_labels('images/data-set/five_5.jpg', "5")
    calc_features_labels('images/data-set/six_1.jpg', "6")
    calc_features_labels('images/data-set/six_2.jpg', "6")
    calc_features_labels('images/data-set/six_3.jpg', "6")
    calc_features_labels('images/data-set/six_4.jpg', "6")
    calc_features_labels('images/data-set/seven_1.jpg', "7")
    calc_features_labels('images/data-set/seven_2.jpg', "7")
    calc_features_labels('images/data-set/seven_3.jpg', "7")
    calc_features_labels('images/data-set/seven_4.jpg', "7")
    calc_features_labels('images/data-set/seven_5.jpg', "7")
    calc_features_labels('images/data-set/eight_1.jpg', "8")
    calc_features_labels('images/data-set/eight_2.jpg', "8")
    calc_features_labels('images/data-set/eight_3.jpg', "8")
    calc_features_labels('images/data-set/eight_4.jpg', "8")
    calc_features_labels('images/data-set/nine_1.jpg', "9")
    calc_features_labels('images/data-set/nine_2.jpg', "9")
    calc_features_labels('images/data-set/nine_3.jpg', "9")
    calc_features_labels('images/data-set/nine_4.jpg', "9")
    calc_features_labels('images/data-set/nine_5.jpg', "9")
	