# based on https://github.com/experiencor/keras-yolo3
import numpy as np
from numpy import expand_dims,load,asarray
from keras.models import load_model
from keras.utils.image_utils import img_to_array
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import Face_Embeddings as fc_emb
import yolo_detect as yolo

def extract_face_frame(frame, required_size=(160, 160)):
    image=cv2.resize(frame,required_size)
    image=img_to_array(image)
    return image
# draw all results
if(__name__=='__main__'):
    # load yolov3 model
    model_yolo = load_model('yolo_model/model.h5')
    facenet_model=load_model('facenet_model/facenet.h5')
    # define the expected input shape for the model
    input_w, input_h = 416, 416
    # define the anchors
    anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
    # define the probability threshold for detected objects
    class_threshold = 0.6
    #load faces
    data = load('data.npz')
    testX_faces = data['arr_2']
    # load face embeddings
    data = load('data-faces-embeddings.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)
    # fit model
    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainy)
    # define our new photo
    cap=cv2.VideoCapture(0)
    while True:
        ret,frame=cap.read()
        image_w=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        image_h=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #detect face
        face,y1,y2,x1,x2=yolo.detect_face_frame(model_yolo,frame,image_w,image_h,input_w,input_h,anchors)
        #get embedded face
        face=extract_face_frame(face,(160,160))
        face_arr=fc_emb.get_embedding(facenet_model,face)
        face_arr=asarray(face_arr)
        samples = expand_dims(face_arr, axis=0)
        yhat_class = model.predict(samples)
        yhat_prob = model.predict_proba(samples)
        # get name
        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index] * 100
        print(class_probability)
        predict_names = out_encoder.inverse_transform(yhat_class)
        if predict_names[0] == 'unknown':
            cv2.rectangle(frame, (x1,y1),(x2,y2), (0, 0, 255), 2)
            cv2.putText(frame, predict_names[0], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else:
            cv2.rectangle(frame, (x1,y1),(x2,y2), (0, 255, 0), 2)
            cv2.putText(frame, predict_names[0], (x1,y1),cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 200), 2)
        #end fix########################
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()