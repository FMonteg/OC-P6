import numpy as np
from PIL import Image, ImageOps
import os
import tensorflow as tf
from keras.applications.vgg16 import VGG16
import pickle


#On cree une fonction qui fera toutes les transformations d'image necessaires
def traitement_image(img, contrast_cutoff = 2, size = 150):
    #Conversion de l'image de RGBA vers RGB si necessaire (ImageOps ne supporte pas le format RGBA)
    if img.mode == 'RGBA':
        img = convert_rgba_rgb(img)
    #Redimension de l'image
    img = ImageOps.fit(img, (size, size), method = Image.BILINEAR, bleed = 0.0, centering =(0.5, 0.5))
    #Etirement de l'histogramme si necessaire
    img = ImageOps.autocontrast(img, cutoff = contrast_cutoff, ignore = contrast_cutoff)
    #Egalisation de l'histogramme
    img = ImageOps.equalize(img, mask = None)
    return img

#Certaines images utilisent le format de couleurs RGBA, qu'il faut convertir en RGB en remplacant la transparence par du blanc
def convert_rgba_rgb(img):
    img.load() # required for png.split()
    background = Image.new("RGB", img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3]) # 3 is the alpha channel
    return background


#On charge et prepare l'image
img = Image.open("image.jpg")
img = traitement_image(img, size=150)
data = np.array([np.array(img)])
data = data / 255.0


#On charge le detecteur de features VGG16
model = VGG16(weights='imagenet', include_top=False)
#On charge le classifieur entraine
keras_model_path = "./keras_save"
model2 = tf.keras.models.load_model(keras_model_path)
#On applique tout ca a l'image formatee
features = model.predict(data)
prediction = model2.predict(features)


#On charge le dictionnaire races <-> labels
with open('classes.pi', 'rb') as f1:
    class_names_label = pickle.load(f1)
#On identifie quelle race est la plus probable d'apres notre classifieur
label = np.argmax(prediction, axis = 1)
race = list(class_names_label.keys())[list(class_names_label.values()).index(label)]
#On l'affiche
print(race)

