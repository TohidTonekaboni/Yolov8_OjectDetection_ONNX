import cv2
from yolov8 import YOLOv8

model_path = 'models/yolov8m.onnx'

# Initialize yolov7 object detector
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

# Capture
ip = 'rtsp://fusion:Fu$ion@192.168.0.33:554/mode=real&idc=4&ids=1'
cap = cv2.VideoCapture(ip)

while True:
    ret, frame = cap.read()

    # Update object localizer
    boxes, scores, class_ids = yolov8_detector(frame)

    combined_img = yolov8_detector.draw_detections(frame)
    cv2.imshow("Detected Objects", combined_img)

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
