import sys
from inception_blocks_v2 import *
from fr_utils import *
from pathlib import Path
import cv2 as cv
from mtcnn.mtcnn import MTCNN
import numpy as np
from tensorflow.keras import backend as K
import pickle
import itertools


# from concurrent.futures import ProcessPoolExecutor


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


def show_register():
    print('\n--------------------------------------------REGISTER--------------------------------------------')
    [print(key, '-' * (30 - len(key)), value) for key, value in register.items()]


def face_detector():
    if CNN_DETECTOR == True:
        K.set_image_data_format('channels_last')
        model = MTCNN()
        return model

    model = cv.CascadeClassifier(str(Path(cv.data.haarcascades) / 'haarcascade_frontalface_default.xml'))
    model = MTCNN()
    return model


def face_recognizer():
    K.set_image_data_format('channels_first')
    model = faceRecoModel(input_shape=(3, 96, 96))
    model.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
    load_weights_from_FaceNet(model)
    return model


def get_faces_mtcnn(image, f=6):
    shape = image.shape
    face_points = detector.detect_faces(image)
    faces = np.empty((len(face_points), 96, 96, 3), dtype='uint8')
    for i, pt in enumerate(face_points):
        x, y, w, h = pt['box']
        ZX = int(w / f)
        ZY = int(h / f)
        xa = max(x - ZX, 0)
        xb = min(x + w + ZX, shape[1])
        ya = max(y - ZY, 0)
        yb = min(y + h + ZY, shape[0])
        face = image[ya: yb, xa: xb]
        face = cv.resize(face, (96, 96), interpolation=cv.INTER_AREA)
        faces[i] = face
    return faces


def get_faces(image, f=6):
    shape = image.shape
    face_points = detector.detectMultiScale(image)
    faces = np.empty((len(face_points), 96, 96, 3), dtype='uint8')
    for i, pt in enumerate(face_points):
        x, y, w, h = pt
        ZX = int(w / f)
        ZY = int(h / f)
        xa = max(x - ZX, 0)
        xb = min(x + w + ZX, shape[1])
        ya = max(y - ZY, 0)
        yb = min(y + h + ZY, shape[0])
        face = image[ya: yb, xa: xb]
        face = cv.resize(face, (96, 96), interpolation=cv.INTER_AREA)
        faces[i] = face
    return faces


def get_encoding(image, get_face=False):
    if CNN_DETECTOR == True:
        faces = get_faces_mtcnn(image, f=6)
    else:
        faces = get_faces(image, f=6)
    encoding = img_to_encoding(faces[0], recognizer)[0]
    if get_face == True:
        return encoding, faces[0]
    return encoding


def check(image, threshold=0.7):
    global register
    encodings = []
    candidates = {}
    min_dist = threshold * 4
    image_fliped = cv.flip(image, 1)
    encodings.append(get_encoding(image))
    encodings.append(get_encoding(image_fliped))
    for name, against in db.items():
        def compute(enc1, enc2):
            return np.linalg.norm(np.subtract(enc1, enc2))

        #             dist = 1 - np.dot(encoding1.reshape(-1), encoding2.reshape(-1))

        # Todo: Add ProcessPoolExecutor
        dists = np.empty(4)
        pairs = itertools.product(encodings, against)
        for i, (enc1, enc2) in enumerate(pairs):
            dists[i] = compute(enc1, enc2)
        print(name, dists)
        if np.sum(dists < threshold) >= 2:
            candidates[name] = np.sum(dists)

    if len(candidates) == 0:
        output('No such person exist!')
    else:
        candidate = min(candidates, key=candidates.get)
        output(f'Hi {candidate}')
        register[candidate] = 1


def output(text):
    print(f'\n\n~~~~ {text}')


def load_models():
    global detector
    global recognizer

    try:
        detector
    except:
        print('Loading ...')
        detector = face_detector()
        print('Loading ...')
        recognizer = face_recognizer()


def save_database():
    import pickle
    try:
        if 'database' not in os.listdir():
            os.mkdir('database')
        filename = 'database.pickle'
        with open('database/' + filename, 'wb') as handle:
            pickle.dump(db, handle)
        print('Done!')
    except Exception as e:
        print(e)


def get_image_from_cam():
    print('Starting Camera. Press "c" to capture')
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot Open Camera')
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            output("Can't receive frame. Existing")
            break
        # Display the frame
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('c'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    return frame[..., ::-1]


def get_image_from_path():
    path = input('Enter image path (with extension): ')
    image = cv.imread(path, 1)
    if type(image) != type(None):
        image = image[..., ::-1]
    return image


def add_to_db(name, image):
    global register, db
    try:
        encodings = []
        image_fliped = cv.flip(image, 1)
        encoding, face = get_encoding(image, get_face=True)
        cv.imshow(name, face[:, :, ::-1])
        cv.waitKey(0)
        cv.destroyAllWindows()
        encodings.append(encoding)

        encoding = get_encoding(image_fliped)
        encodings.append(encoding)

        db[name] = encodings
        register.update({name: 0})
        output('Done!')
    except Exception as e:
        print(f'Invalid path! {e}')


def attendance():
    load_models()
    while True:
        print('=' * 120)
        print('Input image path for a single attendance checkup - enter(1)')
        print('Start webcam for live attendance checkup         - enter(2)')
        print('Go back to main Menu                             - enter(b)')
        print('Exit the program                                 - enter(e)')
        inp = input('Action: ')

        if inp == '1':
            image = get_image_from_path()
            if type(image) == type(None):
                output('Invalid path!')
                continue
            check(image, threshold=0.7)
        elif inp == '2':
            image = get_image_from_cam()[..., ::-1]
            check(image, threshold=0.7)
        elif inp == 'b':
            main_menu()
        elif inp == 'e':
            sys.exit('Program exited')
        else:
            output('Invalid input!')


def db_util():
    global db
    global register

    while True:
        print('=' * 120)
        output('Current database: ')
        print(list(db.keys()))
        print()
        print('Add                  - enter(1)')
        print('Delete               - enter(2)')
        print('Go back to main menu - enter(b)')
        print('Exit the program     - enter(e)')
        inp = input('Action: ')

        if inp == '1':
            load_models()

            print('=' * 120)
            print('Open webcam          - enter(1)')
            print('Provide image path   - enter(2)')
            print('Go back to main menu - enter(b)')
            inp = input('Action: ')
            if inp == '1':
                name = input('Enter name: ')
                image = get_image_from_cam()
                add_to_db(name, image)
            elif inp == '2':
                name = input('Enter name: ')
                image = get_image_from_path()
                if type(image) == type(None):
                    output('Invalid path!')
                    continue
                add_to_db(name, image)
            elif inp == 'b':
                main_menu()
            else:
                output('Invalid input!')

        elif inp == '2':
            inp = input('Enter name to delete: ')
            try:
                del (db[inp])
                del register[inp]
                output(f'Removed {inp}')
            except:
                output('No such name exist!')
        elif inp == 'b':
            main_menu()
        elif inp == 'e':
            sys.exit('Program exited')
        else:
            output('Invalid input!')


def main_menu():
    while True:
        print('=' * 120)
        print('Database                  - enter(1)')
        print('Take attendance           - enter(2)')
        print('show register             - enter(r)')
        print('Save the database to disk - enter(s)')
        print('Exit the program          - enter(e)')
        inp = input('Action: ')

        if inp == '1':
            db_util()
        elif inp == '2':
            attendance()
        elif inp == 'e':
            sys.exit('Program exited')
        elif inp == 'r':
            print(show_register())
        elif inp == 's':
            save_database()
        else:
            output('Invalid input!')


def load_database():
    global db
    global register
    filename = 'database.pickle'
    try:
        with open('database/' + filename, 'rb') as handle:
            db = pickle.load(handle)
        register = {k: 0 for k in db.keys()}
    except:
        output('No database found!')
        db = {}
        register = {}


if __name__ == '__main__':
    while True:
        try:
            CNN_DETECTOR = True
            load_database()
            main_menu()
        except Exception as e:
            print(e)
            continue
