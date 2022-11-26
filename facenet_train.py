from os import listdir
from os.path import isdir
from PIL import Image
from numpy import asarray
# from matplotlib import pyplot
import yolo_detect as yolo
import cv2
from numpy import savez_compressed
from keras.models import load_model
def extract_face(filename, yolo_mdel,required_size=(160, 160)):
    face=yolo.detect_face_image(filename,yolo_mdel)
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

def load_faces(directory,yolo_model):
	faces = list()
	# enumerate files
	for filename in listdir(directory):
		# path
		path = directory + filename
		# get face
		face = extract_face(path,yolo_model)
		# store
		faces.append(face)
	return faces

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory,yolo_model):
	X, y = list(), list()
	# enumerate folders, on per class
	for subdir in listdir(directory):
		# path
		path = directory + subdir + '/'
		# skip any files that might be in the dir
		if not isdir(path):
			continue
		# load all faces in the subdirectory
		faces = load_faces(path,yolo_model)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)
if(__name__=="__main__"):
	yolo_model = load_model('yolo_model/model.h5',compile=False)
  	# load train dataset
	trainX, trainy = load_dataset('data/train/',yolo_model)
	print(trainX.shape, trainy.shape)
    # load test dataset
	testX, testy = load_dataset('data/val/',yolo_model)
    # save arrays to one file in compressed format
	savez_compressed('data.npz', trainX, trainy, testX, testy)
	#luu tru cac vecto dac trung cua tap anh