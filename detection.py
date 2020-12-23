from .social_distancing_config import NMS_THRESH
from .social_distancing_config import MIN_CONF
import numpy as np 
import cv2

def detect_people(frame, net, ln, personIdx=0):
    #Lấy kích thước của khung và khởi tạo danh sách các kết quả 
    (H, W) = frame.shape[:2]
    result = []
    #Tạo một đốm màu từ khung nhập liệu và sau đó chuyển tiếp vượt qua của trình phát hiện đối tượng YOLO, cung cấp các giới hạn và các
    #xác suất liên quan
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    #Khơi tạo danh sách các giới hạn được phát hiện, các trọng tâm tương ứng
    boxes = []
    centroids = []
    confidences = []

    #Vòng lặp qua mỗi đầu ra của lớp
    for output in layerOutputs:
        #Vòng lặp qua từng phát hiện
        for detection in output:
            #Trích xuất ID lớp và độ tin cậy (xác suất)
            # phát hiện đối tượng hiện tại 
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            #Lọc các phát hiện bằng cách (1) đảm bảo rằng đối tượng được phát hiện là một người và (2) đó là người tối thiểu được đáp ứng
            if classID == personIdx and confidence > MIN_CONF:
                #Chia tỷ lệ toạ độ hộp giới hạn trở lại so với kích thước của hình ảnh, YOLO trả về toạ độ trung tâm (x,y) của
                #hộp giới hạn theo sau là chiều rộng của hộp và chiều cao
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                #Sữ dụng toạ độ tâm (x,y) để tính đỉnh và góc trái của hộp giới hạn
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))
                #Cập nhật danh sách toạ độ hộp giới hạn
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
    #Áp dụng chế độ triệt tiêu không cực đại để ngăn chặn sự chồng chéo, yếu hộp giới hạn
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)
    #Đảm bảo tồn tại ít nhất một giới hạn
    if len(idxs) > 0:
        #Lặp lại các chỉ mục đang lưu giữ
        for i in idxs.flatten():
            #Trích xuất toạ độ hộp giới hạn
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            #Cập nhật danh sách kết quả bao gồm 1 người, xác suất dự đoán, toạ độ hộp giới hạn
            r = (confidences[i], (x, y, x+w, y+h), centroids[i])
            result.append(r)
    return result