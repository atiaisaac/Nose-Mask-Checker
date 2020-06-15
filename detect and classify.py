import cv2
import numpy as np
try:
    import tensorflow.lite as tflite
except Exception:
    import tflite_runtime.interpreter as tflite
import time
import sys
import threading

root_dir = "./face-detection-with-OpenCV-and-DNN/"


class Inference_Engine:
    """
    Class for running an image classification inference ad the edge. Two class are expected-
        'mask' and 'no_mask'   
        """
    def __init__(self, frame, model_path, image_size):
        """
        Method for initializing Inference Engine object
        
        Args:
            frame (object)
            model_path (str)
            image_size (int)
            
        Attibutes:
            frame (object) - Video capture object
            model_path (str) - absolute path of tflite model file
            image_size (int) - desired frame shape in tuple form
            interpreter (Tensorflow object) - Tensorflow lite runtime object
            input_details (array) - input characteristics of tflite model
            output_details (array) - output characteristics of tflite model
            """
        
        self.image_size = image_size
        self.model_path = model_path
        self.frame = frame
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]["index"]
        self.output_details = self.interpreter.get_output_details()[0]["index"]
             
    def preprocess(self):
        """
        Method for prerpocessing input frame
        
        Attributes:
            face (object) - processed opencv object
            
        Return:
            object - the preprocessed image frame
            """
        
        face = cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB)
        face = cv2.resize(face,self.image_size)
        face = face/255.0
        face = np.expand_dims(face,axis=0)
        face = face.astype(np.float32) 
        return face
    
    def run(self):
        """
        Method to run the actual classification inference
        
        Attributes:
            predictions (numpy array) - array of probability scores with a shape same as the number of classes
            top_result (int) - class with the hights score
            results_indices (array) - array containing the sorted index of probability scores
            labels (list) - list of classes
            
        Return:
            Tuple of score and class name
            """
        
        face = self.preprocess()
        self.interpreter.set_tensor(self.input_details,face)
        self.interpreter.invoke()
        predictions = self.interpreter.get_tensor(self.output_details)[0]
        top_result = 1
        labels = ["No_mask","Mask"]
        result_indices = np.argsort(predictions)[::-1][:top_result]
        for index in range(top_result):
            score = predictions[result_indices[index]]
            class_name = labels[result_indices[index]]
        return (score,class_name)


class FPS_Enhancer:
    """
    Class to enhance the frame rate of video stream
    """     
    def __init__(self,src=0):
        """Method to read the first frame of the video stream

     Attributes:
        self.stream - video capture 
        self.grabbed (boolean) - confirmation of stream being grabbed
        self.frame - actual stream
        self.stopped (boolean) - variable to clear stream

    Return:
        None
     """
        self.stream = cv2.VideoCapture(src)
        if self.stream.isOpened == False:
            sys.exit
        (self.grabbed, self.frame) = self.stream.read()

        self.stopped = False

    def start(self):
        """Method to thread the frames"""

        threading.Thread(target=self.update,args=()).start()
        return self

    def update(self):
        """Method to loop over the frames"""

        while True:
            if self.stopped:
                return

            (self.grabbed,self.frame) = self.stream.read()

    def read(self):
        """Method to read in the frames

        Return:
            self.frame - siingle frame from stream
        """

        return self.frame

    def stop(self):
        """Method to set varibale for stopping the frame"""
        self.stopped = True


def detect_face(frame, face_net):
    """
    Main function for face detection using caffe model
    
    Arguments:
        frame - the current input frame
        face_net - instance object of caffe face detection model
    
    Attributes:
        (h,w) - tuple holding the height and width of the current frame
        blob - preprocessed frame for face detection
        detections - extracted features
        confidence - probability score of face
        boxes - bounding box coordinates
        startX - left
        startY - top
        endX - right
        endY - bottom
    
    Return:
        None
        """
    
    (h, w) = frame.shape[: 2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence >= 0.7:
            boxes = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = boxes.astype("int")
            (startX, startY) = (max(0, startX), max(0,startY))
            (endX, endY) = (min(w-1, endX),min(h-1, endY))
            
            face = frame[startY:endY,startX:endX]
            IE = Inference_Engine(face, model_path, IMAGE_SHAPE)
            (score, class_name) = IE.run()
            color = (0, 255, 0) if class_name == "Mask" else (0, 0, 255)
            text = "{0:.2f}%:{1}".format(score*100, class_name)

            cv2.rectangle(frame, (startX, startY), (endX, endY),color, 2)
            cv2.putText(frame, text, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    # cv2.imshow("Frame", frame)
    # This will stream to the ffserver running on the systems
    sys.stdout.buffer.write(frame)
    sys.stdout.flush()

IMAGE_H = 224
IMAGE_W = 224
IMAGE_SHAPE = (IMAGE_H, IMAGE_W)
model_path = "./nose_mask.tflite"

cap = FPS_Enhancer(src=0).start()
net = cv2.dnn.readNetFromCaffe(root_dir+"deploy.prototxt.txt", 
                               root_dir+"res10_300x300_ssd_iter_140000.caffemodel")
time.sleep(2)
while True:
    stream = cap.read()

    detect_face(stream, net)
    key = cv2.waitKey(25) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
cap.stop()