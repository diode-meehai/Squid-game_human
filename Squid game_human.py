import random

import cv2
import numpy as np
import imutils
import time
from threading import Thread
import threading

import multiprocessing
from playsound import playsound
import os

class SquidGame_Human:
    def __init__(self):
        # ------------------------------------------------#
        self.width_cam = 1200
        self.height_cam = 720
        self.cap = cv2.VideoCapture(1)
        self.out_frame = None

        # ------------------------------------------------#
        # Write down conf, nms thresholds,inp width/height
        self.confThreshold = 0.25
        self.nmsThreshold = 0.40
        # Load names of classes and turn that into a list
        classesFile = "./yolo_model/coco.names"
        modelConf = './yolo_model/yolov3-tiny.cfg'
        modelWeights = './yolo_model/yolov3-tiny.weights'
        # Set up the net
        self.net = cv2.dnn.readNetFromDarknet(modelConf, modelWeights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        with open(classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        # =============== Variable Mouse ==================#
        self.drawing = False
        self.DownPoint1 = ()
        self.DownPoint2 = ()
        self.DownPoint3 = ()
        self.DownPoint4 = ()
        self.DownPoint = []
        self.Click = 0
        self.Mouse_count = False
        # =============== Variable Mouse ==================#
        self.RandomTime()

        self.audio_AEIOU = os.path.dirname(__file__) + '/Sound_SquidGame/AEIOU_Robot2.mp3'
        self.img_move = cv2.imread('./img_SquidGame/Squid-Game-1.jpg')
        self.img_move = cv2.resize(self.img_move, (self.width_cam, self.height_cam))
        self.time_count_moving = self.RandomTimeMove
        self.time_check_moving = True

        self.Sound_GUN = None
        self.audio_gun = os.path.dirname(__file__) + '/Sound_SquidGame/GUN4.mp3'
        self.img_scan = cv2.imread('./img_SquidGame/Squid-Game-2.jpg')
        self.img_scan = cv2.resize(self.img_scan, (self.width_cam, self.height_cam))
        self.time_count_scan = self.RandomTimeScan
        self.time_check_scan = False

        self.FrameROI_Moving = None
        self.FirstFrame = None
        self.image_save_move = 0

        # --------------- StartGame ------------- #
        self.TimeStart = 10
        self.img_startGif = cv2.VideoCapture('./img_SquidGame/7B99.gif')
        self.TimeStartCheck = True
        # --------------- StartGame ------------- #

    def StartGame(self):
        lenFrame = int(self.img_startGif.get(cv2.CAP_PROP_FRAME_COUNT))
        print(lenFrame)

        while lenFrame > 0:
            print(lenFrame)
            success, frame_startGif = self.img_startGif.read()
            frame_startGif = cv2.resize(frame_startGif, (self.width_cam, self.height_cam))
            if success:
                time.sleep(1)
                cv2.imshow('TimeStart', frame_startGif)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            lenFrame -= 1

        self.img_startGif.release()
        cv2.destroyAllWindows()
        self.TimeStartCheck = False
        # self.LoopRunCam()

    def RandomTime(self):
        self.RandomTimeMove = random.randint(2, 10)
        self.RandomTimeScan = random.randint(2, 10)

    def countdown_moving(self):
        if self.Sound_GUN is not None:
            self.Sound_GUN.terminate()

        self.Sound_AEIOU = multiprocessing.Process(target=playsound, args=(self.audio_AEIOU,))
        self.Sound_AEIOU.start()
        if self.time_check_moving:
            while self.time_count_moving >= 0:
                time.sleep(1)
                self.time_count_moving -= 1
                print('Moving: ', self.time_count_moving)

            self.RandomTime()
            self.time_check_scan = True
            self.time_count_scan = self.RandomTimeScan
            self.time_check_moving = False

    def countdown_scan(self):
        self.Sound_AEIOU.terminate()
        while self.time_count_scan >= 0:
            time.sleep(1)
            self.time_count_scan -= 1
            print('Scan: ', self.time_count_scan)

        self.RandomTime()
        self.time_check_moving = True
        self.time_count_moving = self.RandomTimeMove
        self.time_check_scan = False

    def ObjMoving(self, frame):
        # firstFrame = None
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if self.FirstFrame is None:
                self.FirstFrame = np.copy(gray)

            if self.FirstFrame is not None:
                frameDelta = cv2.absdiff(self.FirstFrame, gray)
                thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                for c in cnts:
                    if cv2.contourArea(c) > 500:
                        (x, y, w, h) = cv2.boundingRect(c)
                        if w > 20 and h > 20:
                            # self.FrameROI_Moving = frame[y:y + h, x:x + w]  # [y1:y2, x1:x2]
                            # ================ Dectec People ================ #
                            pts = np.array([
                                [x, y], # X1,Y1
                                [x + w, y],
                                [x + w, y + h],
                                [x, y + h]
                            ])
                            canvas = np.zeros_like(frame)  # ---black in RGB---#
                            cv2.drawContours(canvas, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

                            out_frame = np.zeros_like(frame)  # Extract out the object and place into output image
                            out_frame[canvas == 255] = frame[canvas == 255]
                            # cv2.imshow('out_frame Moving: ', out_frame)

                            blob = cv2.dnn.blobFromImage(out_frame, 1 / 255, (416, 416), [0, 0, 0], 1,
                                                         crop=False)
                            self.net.setInput(blob)
                            outs = self.net.forward(self.getOutputsNames())
                            selected_boxes = self.postprocess(out_frame, outs)

                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(self.out_frame, str(selected_boxes), (10, 100), font, 1, (255, 200, 55), 2,
                                        cv2.LINE_AA)
                            # ================ Dectec People ================ #
                            cv2.rectangle(self.out_frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        return frame

    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        classIDs = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > self.confThreshold:
                    centerX = int(detection[0] * frameWidth)
                    centerY = int(detection[1] * frameHeight)

                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)

                    left = int(centerX - width / 2)
                    top = int(centerY - height / 2)

                    classIDs.append(classID)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        # print('ClassIDs' +str(classes)) # 2 = car, 7= 'truck'
        # ------------ Count --------------#
        boxes = np.array(boxes)
        classIDs = np.array(classIDs)
        confidences = np.array(confidences) # %
        selected_boxes = boxes[indices]

        # --- person --#
        person_deteced_indices = np.where(classIDs[indices] == 0)
        # person_ALL = len(person_deteced_indices)
        # print(classIDs[indices])
        selected_boxes = selected_boxes[person_deteced_indices]
        # confidences = confidences[car_deteced_indices[0]]
        # classIDs = classIDs[car_deteced_indices[0]]
        # ------------ Count --------------#

        subtract = False
        for i, box in enumerate(selected_boxes):
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            # ----------------- Center Box Human ---------------- #
            right = left + width
            bottom = top + height
            # center = (int((left + right) / 2), int((top + bottom) / 2))
            # centerCount_list.append(center)
            # ----------------- Center Box Human ---------------- #

            if self.classes[classIDs[i]] == 'person':
                self.drawPred(classIDs[i], confidences[i], left, top, right, bottom, subtract)

                self.Sound_GUN = multiprocessing.Process(target=playsound, args=(self.audio_gun,))
                self.Sound_GUN.start()

                img_save_preson_move = frame[top:bottom, left:right]  # [y1:y2, x1:x2]
                # cv2.imshow('image_save_move', img_save_preson_move)
                # cv2.waitKey(0)
                cv2.imwrite('./img_personMove/img_' + str(self.image_save_move) + '.jpg', img_save_preson_move)
                self.image_save_move += 1

        return len(selected_boxes)

    def drawPred(self, classId, conf, left, top, right, bottom, subtract):
        Percent = '%.2f' % conf
        assert (classId < len(self.classes))
        label = '%s:%s' % (self.classes[classId], Percent)
        cv2.putText(self.out_frame, str(label), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv2.rectangle(self.out_frame, (left, top), (right, bottom), (255,0,0), 2)

    def getOutputsNames(self):
        # Get the names of all the layers in the network
        layersNames = self.net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    # ================================================#
    def mouse_drawing(self, event, x, y, flags, params):
        # ----------Mouse 1------- #
        if not self.Mouse_count:
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.drawing is False:
                    if self.Click == 0:
                        self.DownPoint1 = (x, y)
                        print("P1:", self.DownPoint1)
                        cv2.circle(self.frame_First, (x, y), 5, (25, 255, 255), -1)
                        cv2.imshow("Detecion ROI", self.frame_First)
                        self.Click = 1

                    elif self.Click == 1:
                        self.DownPoint2 = (x, y)
                        print("P2:", self.DownPoint2)
                        cv2.circle(self.frame_First, (x, y), 5, (25, 255, 255), -1)
                        cv2.imshow("Detecion ROI", self.frame_First)
                        self.Click = 2

                    elif self.Click == 2:
                        self.DownPoint3 = (x, y)
                        print("P3:", self.DownPoint3)
                        cv2.circle(self.frame_First, (x, y), 5, (25, 255, 255), -1)
                        cv2.imshow("Detecion ROI", self.frame_First)
                        self.Click = 3

                    elif self.Click == 3:
                        self.DownPoint4 = (x, y)
                        print("P4:", self.DownPoint4)
                        cv2.circle(self.frame_First, (x, y), 5, (25, 255, 255), -1)
                        cv2.imshow("Detecion ROI", self.frame_First)
                        self.Click = 0
                        self.Mouse_count = True

    def ROI_FirstFrame(self, frame):
        # --------------------Roi Mouse--------------------#
        pts = np.array([
            [self.DownPoint1[0], self.DownPoint1[1]],
            [self.DownPoint2[0], self.DownPoint2[1]],
            [self.DownPoint3[0], self.DownPoint3[1]],
            [self.DownPoint4[0], self.DownPoint4[1]]])

        canvas = np.zeros_like(frame)  # ---black in RGB---#
        cv2.drawContours(canvas, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

        out_frame = np.zeros_like(frame)  # Extract out the object and place into output image
        out_frame[canvas == 255] = frame[canvas == 255]

        return out_frame

    def LoopRunCam(self):
        # ================================================#
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 20)
        _, self.frame_First = self.cap.read()
        self.frame_First = cv2.resize(self.frame_First, (self.width_cam, self.height_cam))
        self.frame_First = cv2.flip(self.frame_First, 1)

        cv2.namedWindow("Detecion ROI")
        cv2.setMouseCallback("Detecion ROI", self.mouse_drawing)
        cv2.imshow("Detecion ROI", self.frame_First)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        while cv2.waitKey(1) < 0:

            if self.TimeStartCheck:
                self.StartGame()

            # get frame from video
            hasFrame, frame = self.cap.read()
            frame = cv2.resize(frame, (self.width_cam, self.height_cam))
            frame = cv2.flip(frame, 1)

            if self.Mouse_count:  # True
                if hasFrame:
                    # ========= Check Moving Frame ========== #
                    if self.time_check_moving:
                        self.out_frame = np.copy(self.img_move)
                        cv2.putText(self.out_frame, 'Music..' + str(self.time_count_moving), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        # self.countdown_moving()
                        if self.time_count_moving == self.RandomTimeMove:
                            print('A E I O U')
                            time_move = threading.Timer(1.0, self.countdown_moving)
                            time_move.start()
                            self.time_count_moving -= 1

                        elif self.time_count_moving <= 0:
                            time_move.cancel()
                            # time_move.join()

                    else:
                        FrameCheckMoving = self.ROI_FirstFrame(frame=frame)
                        self.out_frame = np.copy(FrameCheckMoving)
                        cv2.putText(self.out_frame, 'Scan Moving ! ' + str(self.time_count_scan), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (100, 0, 255), 2)
                        Frame_Moving = self.ObjMoving(frame=FrameCheckMoving)

                        self.out_frame = cv2.addWeighted(self.out_frame, 0.9, self.img_scan, 0.1, 0.5)

                        if self.time_count_scan == self.RandomTimeScan:
                            self.FirstFrame = None
                            time_scan = Thread(target=self.countdown_scan)
                            time_scan.start()
                            self.time_count_scan -= 1
                        elif self.time_count_scan <= 0:
                            time_scan.join()

                cv2.imshow('Frame_Moving_prople', self.out_frame)

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    RunClasYOLO_Humans = SquidGame_Human()
    RunClasYOLO_Humans.LoopRunCam()
