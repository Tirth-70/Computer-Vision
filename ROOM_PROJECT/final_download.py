from ultralytics import YOLO

# Load model
model1 = YOLO("/Users/tirthpatel/Desktop/Computer Vision/Image Classification/Clean-Messy/yolo logs/train2/weights/best.pt")
model2 = YOLO('/Users/tirthpatel/Desktop/Computer Vision/Image Detection/stuff detection/yolo logs/train2/weights/best.pt')

vid = "ROOM_PROJECT/data/6612061-uhd_4096_2160_25fps.mp4"


# results = model2(vid, save=True, project='/Users/tirthpatel/Desktop/Computer Vision/ROOM_PROJECT/output')

vid = "/Users/tirthpatel/Desktop/Computer Vision/ROOM_PROJECT/output/predict/6612061-uhd_4096_2160_25fps.mp4"

results = model1(vid, save=True, project='/Users/tirthpatel/Desktop/Computer Vision/ROOM_PROJECT/output', name='final_output', device='mps')