import cv2 as cv
import numpy as np
from tensorflow.keras import models


DEFAULT_IMG_SIZE = (224, 224)


def run_app():
    model = models.load_model("models/20220405_203250-20220405_203250.h5")

    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Our operations on the frame come here
        # Get image
        image = cv.resize(
                    frame, 
                    DEFAULT_IMG_SIZE,
                    interpolation= cv.INTER_LINEAR)

        # Get prediction
        prediction = model.predict(np.array([image]))
        prediction = np.argmax(prediction, axis=-1)

        # Put prediction on image
        if prediction:
            image = cv.putText(frame, "Mask detected", (50, 100), cv.FONT_HERSHEY_COMPLEX, .5, (0,255,0))
        else:
            image = cv.putText(frame, "No mask", (50, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255))

        # Display the resulting frame
        cv.namedWindow("window", cv.WINDOW_NORMAL)
        cv.imshow('window', frame)

        # Wait for 'q' key to shutdown the program
        if cv.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    run_app()