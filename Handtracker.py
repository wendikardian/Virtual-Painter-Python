# Step 1, initialization the file (import, webcam, etc)

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

# Step 2. 
# Define the tracker for hands that we are going to use for tracking a hand from mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands()

# Step 7
mpDraw = mp.solutions.drawing_utils

# Step 9
# Declare variable to calculate the FPS
# Ptime means previous time and cTime means current time
pTime = 0
cTime = 0

while True:
    res, frame = cap.read()

    # Step 3
    # Because the frame is BGR color, we need to convert it to RGB color because hands on MP only can work for RGB color
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Step 4
    # After we convert it into RGB color, we need to process it using the hands variable that already created before
    results = hands.process(frameRGB)

    # Step 5
    # Try to print the results, to find out what is inside the results
    # The results print, it print class mediapipe solitionOutput
    # print(results)
    # To detect if there some hands on the camera using this one
    # print(results.multi_hand_landmarks)

    # Step 6
    # If there is a hand detected on the camera
    if results.multi_hand_landmarks:
        # We will loop it, because maybe there is much more than 1 hand on the camera
        for handLms in results.multi_hand_landmarks:

            # Step 12
            # After to find landmark on the frame, next we need to figure out the id of the landmark
            for id, lm in enumerate(handLms.landmark):
                # Try to print it, this is the example of the what is the value of id and lm
                # 20 x: 0.5787419676780701 -> 20 is id
                # y: 0.9594467282295227
                # z: -0.01320577971637249
                # print(id, lm)
                h, w, c = frame.shape #try to get height, width, channel
                cx, cy = int(lm.x*w), int(lm.y*h)
                # Try to print it
                # If we print it, it will the display the id of every landmark to every coordinates on the frame 
                print(id, cx, cy)

                # id == 0 is one of the node landmark on the hand
                if id == 0:
                    cv2.circle(frame, (cx,cy), 20, (255,0,0), cv2.FILLED)

            # Step 6
            # After we define the mpDraw to draw the solutions, use it to draw it
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
            # Step 8. Try to run this and then see the frame of the camera if there is red node on the hand, try it for multiple hand
            # HAND_CONNECTIONS means it will connect every node so it has a connection to each other (similar with a graph)

    # Step 10
    # Calculate the time for display the FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    # Step 11
    # After find the FPS value we need to display it on the frame
    cv2.putText(frame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Frame", frame)

   


    key= cv2.waitKey(1)
    if key == ord('q'):
        break;

cap.release()
cv2.destroyAllWindows()