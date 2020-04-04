from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
import pickle
import os
import math
import detect_test
import timeit
import tkinter 
from tkinter import *
from PIL import ImageTk,Image
from tkinter.filedialog import askopenfilename

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1))) 
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))
model.load_weights('../data/vgg_face_weights.h5')

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
accuracy = 0.36
mylist_image = []
mylist_value = []
name_list= [
            'Arya Stark \n 23 ans',
            'Leonardo Dicaprio \n 49 ans',
            'Kit Harington \n 31 ans',
            'Christopher Nolan \n 56 ans',
            'Soufian Mejdaoui\n 22 ans',
            'Travis_Fimmel \n 40 ans'
             ]

def load():
    with open("../data/images.txt", 'rb') as f:
        mylist_image1 = pickle.load(f)

    with open("../data/values.txt", 'rb') as g:
        mylist_value1 = pickle.load(g)
    return mylist_value1, mylist_image1

def verifyFace(img):
    start = timeit.default_timer()
    list_index = []
    img_representation = vgg_face_descriptor.predict(preprocess_image('%s' % (img)))[0,:]
    for i in range(0,len(mylist_image)):
        cosine_similarity = findCosineSimilarity(img_representation, mylist_value[i])
        print("Accuracy: ", cosine_similarity,"Pic: ", mylist_image[i])
        list_index.append(cosine_similarity)
    tmp = min(list_index)
    stop = timeit.default_timer()
    if tmp <= accuracy:
        print("Le temps ecoulé est : ",stop-start)
        return list_index.index(tmp),img
    else:
        print("Le temps ecoulé est : ",stop-start)
        return -1,img

def choose_video():
    root = Tk()
    root.withdraw()
    root.update()
    flux = askopenfilename(filetypes=[("Video files","*.mp4")])
    if flux!='':
        list_images_captures = detect_test.open_flux(flux)
        recognize_persons(list_images_captures)
    root.destroy()

def choose_camera():
    list_images_captures = detect_test.open_flux('0')
    recognize_persons(list_images_captures)

def quit(window):
    window.destroy()

def recognize_persons(list_images_captures):
    i = 0
    while i < len(list_images_captures):
        f = plt.figure()
        indice,img = verifyFace(list_images_captures[i]) ##########################################################
        if indice != -1:
            print("you are ", name_list[indice])
            f.add_subplot(1, 2, 1)
            plt.imshow(image.load_img('%s' % (img)))
            plt.text(30, -12, name_list[indice], bbox=dict(facecolor='blue', alpha=0.1))
            plt.xticks([])
            plt.yticks([])
            f.add_subplot(1, 2, 2)
            plt.imshow(image.load_img('%s' % (mylist_image[indice])))
            plt.xticks([])
            plt.yticks([])
            plt.show(block=True)
            i += 1
        else:
            print("not in the database")
            i += 1
#end recognize_persons
#---------------debut du main program----------------  
top = tkinter.Tk()
mylist_value,mylist_image  = load()
top.title('Face Recognition')
top.geometry("500x500") 
top.resizable(0, 0) 
can = Canvas(top)
can.pack(expand=True, fill=BOTH)
filename = PhotoImage(file = "facial.png")
can.img=filename
can.create_image(0, 0, anchor=NW, image=filename)
var = StringVar()
label = Label( top, textvariable=var, relief=RAISED,font='Helvetica',fg='blue',height=1,width=50)
var.set("Welcom to our application,Make your choice:")
B =  tkinter.Button(top, text ="open Camera",height=3,width=30,fg='white',bg='blue',command=choose_camera)
C = tkinter.Button(top, text ="open Video",height=3,width=30,fg='white',bg='blue',command=choose_video)
D = tkinter.Button(top,text="Quit",height=3,width=30,fg='white',bg='blue',command=lambda:quit(top))
can.create_window(70,120,window=label,anchor=NW)
can.create_window(200,190, window=B, anchor=NW)
can.create_window(200,260, window=C, anchor=NW)
can.create_window(200,330, window=D, anchor=NW)
top.mainloop()

#prog_main()
'''
for filename in os.listdir("../captured_pictures"):
    if filename.endswith('.jpg'):
        #print(filename)
        os.unlink("../captured_pictures/"+filename)
'''