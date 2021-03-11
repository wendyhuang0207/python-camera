import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# videoWriter = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'MJPG'), 15, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # frame = cv2.flip(frame,0)
        # videoWriter.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
# videoWriter.release()
cv2.destroyAllWindows()