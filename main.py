import sys
from inception_blocks_v2 import *
from fr_utils import *
from pathlib import Path
import cv2 as cv
from mtcnn.mtcnn import MTCNN
import numpy as np
from tensorflow.keras import backend as K
import argparse
import pickle


def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(np.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(np.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(0.0, basic_loss))
    return loss


def face_detector():
    print('loading FD - ', K.image_data_format())
    model = MTCNN()
    print('Done loading FD - ', K.image_data_format())
    return model


def face_recognizer():
    print('loading FR - ', K.image_data_format())
    K.set_image_data_format('channels_first')
    model = faceRecoModel(input_shape=(3, 96, 96))
    model.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
    load_weights_from_FaceNet(model)
    print('Done loading FR - ', K.image_data_format())
    return model


def get_faces_mtcnn(image, detector, f=6):
    shape = image.shape
    face_points = detector.detect_faces(image)
    faces = np.empty((len(face_points), 96, 96, 3), dtype='uint8')
    for i, pt in enumerate(face_points):
        x, y, w, h = pt['box']
        ZX = int(w/f)
        ZY = int(h/f)
        xa = max(x - ZX, 0)
        xb = min(x + w + ZX, shape[1])
        ya = max(y - ZY, 0)
        yb = min(y + h + ZY, shape[0])
        face = image[ya: yb, xa: xb]
        face = cv.resize(face, (96, 96), interpolation=cv.INTER_AREA)
        faces[i] = face
    return faces

def get_faces(image, classifier, f=6):
    shape = image.shape
    face_points = classifier.detectMultiScale(image)
    faces = np.empty((len(face_points), 96, 96, 3), dtype='uint8')
    for i, pt in enumerate(face_points):
        x, y, w, h = pt
        ZX = int(w/f)
        ZY = int(h/f)
        xa = max(x - ZX, 0)
        xb = min(x + w + ZX, shape[1])
        ya = max(y - ZY, 0)
        yb = min(y + h + ZY, shape[0])
        face = image[ya: yb, xa: xb]
        face = cv.resize(face, (96, 96), interpolation=cv.INTER_AREA)
        faces[i] = face
    return faces


def get_dist(image1, image2, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".

    Arguments:
    image_path -- path to an image
    identity -- string, name of the person you'd like to verify the identity. Has to be an employee who works in the office.
    database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
    model -- your Inception model instance in Keras

    Returns:
    dist -- distance between the image_path and the image of "identity" in the database.
    door_open -- True, if the door should open. False otherwise.
    """
    # K.set_image_data_format('channels_first')
    encoding1 = img_to_encoding(image1, model)
    encoding2 = img_to_encoding(image2, model)
    dist = np.linalg.norm(np.subtract(encoding1, encoding2))
#     dist = 1 - np.dot(encoding1.reshape(-1), encoding2.reshape(-1))
    if dist < 0.7:
        print('Welcome in!')
        door_open = True
    else:
        print('Please go away')
        door_open = False

    return dist, door_open


def check(image, model, database, threshold=0.7):
    min_dist = threshold
    candidate = None
    encoding = img_to_encoding(image, model)
    for name, against in database.items():
        dist = np.linalg.norm(np.subtract(encoding, against))
#         dist = 1 - np.dot(encoding1.reshape(-1), encoding2.reshape(-1))

    if dist < min_dist:
        min_dist = dist
        candidate = name
            
    return name, min_dist



def check_up(image, model, database, threshold=0.7):
    min_dist = threshold
    candidates = []
    encoding = img_to_encoding(image, model)
    for name, againsts in database.items():
        dist = np.linalg.norm(np.subtract(encoding, against))
#         dist = 1 - np.dot(encoding1.reshape(-1), encoding2.reshape(-1))

    if dist < min_dist:
        min_dist = dist
        candidate = name
            
    return name, min_dist

    
def face_recognizer():
    K.set_image_data_format('channels_first')
    model = faceRecoModel(input_shape=(3, 96, 96))
    model.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
    load_weights_from_FaceNet(model)
    return model


def output(text):
    print(f'\n\n~~~~ {text}')

  
def db():  
    global db    
    while True:
        print('='*120)
        try:
            output('Current database:')
            print(db.keys())
        except:
            output('No database found!')
        print()
        print('Add                   - enter(1)')
        print('Delete                - enter(2)')
        print('Go back to main menu  - enter(b)')
        print('Exit the program      - enter(e)')
        inp = input('Action: ')
        
        if inp == '1':
            print('='*120)
            print('Open webcam          - enter(1)')
            print('Provide image path   - enter(2)')
            print('Go back to main menu - enter(b)')
            inp = input('Action: ')            
            if inp == '1':                
                print('start webcam')
            elif inp == '2':
                print('enter input in following format - {name}{single-space}{path1}{single-space}{path2}{more images} (following the same pattern, if available)')
                print('Example: xyz /home/images/img1.jpg /home/images/img2.jpg /home/images/img3.jpg')
                inp = input('Enter: ')
                data = inp.split()
                try:
                    for path in data[1:]:
                        encodings = []
                        img = cv.imread(path)[:, :, ::-1]
                        img_fliped = cv.flip(img, 1)                        
                        encodings.append(img_to_encoding(img, model))
                        encodings.append(img_to_encoding(img_fliped, model))
                    db[data[0]] = encodings
                    output('Done!')
                except:
                    output('Invalid input!')
            elif inp == 'b':
                main_menu()
            else:
                output('Invalid input!')
                    
        elif inp == '2':
            inp = input('Enter name to delete: ')
            try:
                del(db[inp])
                output(f'Removed {inp}')
            except:
                output('No such name exist!')
        elif inp == 'b':
            main_menu()
        elif inp == 'e':
            sys.exit('Program exited')
        else:
            output('Invalid input!')


def attendance():
    global detector
    global recognizer

    try:
        detector
    except:
        print('Loading ...')
        detector = cv.CascadeClassifier(str(Path(cv.data.haarcascades) / 'haarcascade_frontalface_default.xml'))

        recognizer = recognizer = face_recognizer() 

    while True:
        print('='*120)
        print('Input image path for a single attendance checkup - enter(1)')
        print('Start webcam for live attendance                 - enter(2)')
        print('Go back to main Menu                             - enter(b)')
        print('Exit the program - enter(e)')
        inp = input('Action: ')

        if inp == '1':
            path = input('Enter image path (with extension): ')
            img = cv.imread(path, 1)
            if img == None:
                output('Invalid path!')
                continue
            img = img[...,::-1]
            check(img, recognizer, db, threshold=0.7)
        elif inp == '2':
                print('start webcam')
        elif inp == 'b':
            main_menu()
        elif inp == 'e':
            sys.exit('Program exited')
        else:
            output('Invalid input!')
    
            
def main_menu():    
    while True:
        print('='*120)
        print('Database         - enter(1)')
        print('Take attendance  - enter(2)')
        print('Exit the program - enter(e)')
        inp = input('Action: ')

        if inp == '1':
            db()
        elif inp == '2':
            attendance()
        elif inp == 'e':
            sys.exit('Program exited')
        else:
            output('Invalid input!')
        
        
def load_database():
    global db
    filename = 'database'
    try:
        with open(str('database' / filename), 'rb') as infile:
            db = pickle.load(infile)
    except:
        output('No database found!')
        
if __name__ == '__main__':
    load_database()
    main_menu()