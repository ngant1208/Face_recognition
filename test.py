from keras.models import load_model
if(__name__=="__main__"):
    model=load_model('yolo_model/model.h5')
    print("done")