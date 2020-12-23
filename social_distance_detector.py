from pyimagesearch import social_distancing_config as config 
from pyimagesearch.detection import detect_people
from scipy.spatial import distance as dist 
import numpy as np 
import argparse
import imutils
import cv2 
import os 

#Tạo phân tích cú pháp đối số và phân tích cú pháp đối số
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="", help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="", help="path to (optional) input video file")
ap.add_argument("-d", "--display", type=int, default="1", help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

#Nhãn lớp COCO mô hình YOLO được đào tạo 
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

#Lấy ra đường dẫn đến trọng số YOLO và cấu hình mô hình
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

print("[INFO] Loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

if config.USE_GPU:
    print("[INFO] Setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

print("[INFO] Accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None

#Vòng qua các khung từ luồng video
while True:
    #Đọc khung từ tệp
    (grabbed, frame) = vs.read()
    if not grabbed:
        break  
    #Thay đổi kích thước khung và sau đó phát hiện mọi người (và chỉ những người) trong đó 
    frame = imutils.resize(frame, width=700)
    results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
    #Khởi tạo bộ chỉ mục vi phạm xã hội tối thiểu khoảng cách
    violate = set()
    #Đảm bảo có * ít nhất * phát hiện 2 người (bắt buộc trong để tính toán bản đồ khoảng cách theo cặp)
    if len(results) >= 2:
        #Trích xuất tất cả centroid từ kết quả và tính toán khoảng cách Euclide giữa tất cả các cặp centroid
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")
        #Vòng lặp qua tam giác ma trận khoảng cách
        for i in range(0, D.shape[0]):
            for j in range(i+1, D.shape[1]):
                #Kiểm tra xem liệu khoảng cách giữa hai cặp centroid nhỏ hơn số đã định cấu hình trong pixel
                if D[i, j] < config.MIN_DISTANCE:
                    #Cập nhật tập hợp phạm vi với các chỉ mục của cặp centroid
                    violate.add(i)
                    violate.add(j)
        #Vòng lặp qua các kết quả
        for (i, (prob, bbox, centroids)) in enumerate(results):
            #Trích xuất hộp giới hạn và tọa độ trọng tâm, sau đó khởi tạo màu của chú thích
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroids
            color = (0, 255, 0)
            #Nếu cặp chỉ mục tồn tại trong tập hợp phạm vi, thì cập nhật màu
            if i in violate:
                color = (0, 0, 255)
            #Vẽ (1) hộp giới hạn xung quanh người và (2) toạ độ centroid của người
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)
        #Vẽ tổng số vi phạm gây mất trật tự xã hội trên khung đầu ra
        text = "Social Distancing Violations: {}".format(len(violate))
        cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
        if args["display"] > 0:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        if args["output"] != "" and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 25, (frame.shape[1], frame.shape[0]), True)
        if writer is not None:
            writer.writer(frame)
