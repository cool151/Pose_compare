from flask import Flask , render_template  , request, url_for, Response, redirect
from PIL import Image
from flask_cors import CORS, cross_origin
from uuid import uuid4
from numpy.linalg import norm
import base64
import io
import PIL
import mediapipe as mp
import cv2
import numpy as np
import time

app = Flask(__name__)
camera = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
#----------Not to use ---------------------------------------------------------------------
def generate_frames(lts):
    cap = cv2.VideoCapture(0)
    prevTime = 0
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        while True:
            success, image = cap.read()
            if success:
                # print("Ignoring empty camera frame.")
                # # If loading a video, use 'break' instead of 'continue'.
                # continue

                # Convert the BGR image to RGB.
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                results = pose.process(image)

                #get coordinate
                lmList = []
                if results.pose_landmarks:
                    for id, lm in enumerate(results.pose_landmarks.landmark):
                        h, w, c = image.shape
                        # print(id, lm)
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([cx, cy])
                del lmList[1:11]

                # Draw the pose annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                currTime = time.time()
                fps = 1 / (currTime - prevTime)
                prevTime = currTime
                frame=cv2.flip(image,1)
                cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
                try:
                    cv2.putText(frame, 'similar:'+ calculate_lst(lmList,lts)[0], (300,70),cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
                except IndexError:
                    pass
                # cv2.imshow('BlazePose', image)
                frame = cv2.imencode('.jpg', frame )[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                key = cv2.waitKey(20)
                if key == 27:
                    break
#------------------------------------------------------------------------------------

EDGES = {
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (4, 6),
    (3, 5),
    (2, 14),
    (1, 13),
    (1, 2),
    (13, 14),
    (14, 16),
    (13, 15),
    (16, 18),
    (15, 17)
}
def calculate_lst(lsta, lstb):
        A=np.array(lsta)
        B=np.array(lstb)
        list1=[]
        list2=[]
        try:

            for edge in EDGES:
                p1, p2 = edge
                x1, x2 = A[p1][0] - A[p2][0], B[p1][0] - B[p2][0]
                y1, y2 = A[p1][1] - A[p2][1], B[p1][1] - B[p2][1]
                list1.append([x1, y1])
                list2.append([x2, y2])
        except IndexError:
            pass
        list1_1=np.array(list1)
        list2_1=np.array(list2)
        cosine = np.sum(list1_1*list2_1, axis=1)/(norm(list1_1, axis=1)*norm(list2_1, axis=1))
        csn = str(round(100*(sum(cosine)/len(cosine)),2)) + "%"
        if(round(100*(sum(cosine)/len(cosine)),2) > 70):
            color_set='1'
        else:
            color_set='0'
        return csn, color_set


def lst_cor(img, draw=True):
        mpDraw = mp.solutions.drawing_utils
        mpPose = mp.solutions.pose
        pose = mpPose.Pose(min_detection_confidence=0.85, min_tracking_confidence=0.85)
        canvas = np.zeros_like(img, np.uint8)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        if results.pose_landmarks:
            if draw:
                mpDraw.draw_landmarks(canvas, results.pose_landmarks,
                                        mpPose.POSE_CONNECTIONS,landmark_drawing_spec=mpDraw.DrawingSpec(color=(255,255,255),
                                                                               thickness=3, circle_radius=3),
                                  connection_drawing_spec=mpDraw.DrawingSpec(color=(0, 128, 0),
                                                                               thickness=5, circle_radius=2))
        lmList = []
        if results.pose_landmarks:
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([cx, cy])
        del lmList[1:11]

        return canvas, lmList

# @app.route('/video', methods = ['GET', 'POST'])
# def camera():
#     return render_template("flask_4_video.html")

@app.route('/', methods = ['GET', 'POST'])
@cross_origin(origins="*")
def upload_file():
    input_image_1=""
    input_image_2=""
    input_image_pic1=""
    input_image_pic2=""
    csn=""
    color_set=""
    if (request.files):
        #take the pic
        file = Image.open(request.files['image'])
        #convert img to base64 string
        buffer = io.BytesIO()
        file.save(buffer, "PNG")
        input_image_1=base64.b64encode(buffer.getvalue()).decode()

        # print(file)
        file_2 = Image.open(request.files['image_2'])
        #convert img to base64 string
        buffer_2 = io.BytesIO()
        file_2.save(buffer_2, "PNG")
        input_image_2=base64.b64encode(buffer_2.getvalue()).decode()

        open_cv_image = cv2.cvtColor(np.array(file), cv2.COLOR_RGB2BGR)
        open_cv_image_2 = cv2.cvtColor(np.array(file_2), cv2.COLOR_RGB2BGR)
        #xu ly anh mediapipe
        pic_1, A = lst_cor(open_cv_image, True)
        pic_2, B = lst_cor(open_cv_image_2, True)
        pic_1 = Image.fromarray(np.uint8(pic_1)).convert('RGB')
        pic_2 = Image.fromarray(np.uint8(pic_2)).convert('RGB')

        buffer_3 = io.BytesIO()
        pic_1.save(buffer_3, "PNG")
        input_image_pic1 = base64.b64encode(buffer_3.getvalue()).decode()
        buffer_4 = io.BytesIO()
        pic_2.save(buffer_4, "PNG")
        input_image_pic2 = base64.b64encode(buffer_4.getvalue()).decode()
        
        csn, color_set = calculate_lst(A,B)

    return render_template('flask_4.html', k1_img=input_image_1, k2_img=input_image_2, k3_img= input_image_pic1, k4_img=input_image_pic2, cosine_smr=csn, color_set=color_set)   
if __name__ == "__main__":
    app.run(debug = True)
