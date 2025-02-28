import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

# Load Mask R-CNN model
class InferenceConfig(Config):
    NAME = "pedestrian_detection"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2  # Background + Pedestrian

config = InferenceConfig()
model = MaskRCNN(mode="inference", model_dir="logs", config=config)
model.load_weights("/content/mask_rcnn_pedestrian_trained.h5", by_name=True)

# Define pedestrian class ID
PEDESTRIAN_CLASS_ID = 1

# Cyclist lane boundaries
CYCLIST_LANE_X1, CYCLIST_LANE_X2 = 200, 400

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Convert frame to RGB for Mask R-CNN
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect pedestrians
        results = model.detect([rgb_frame], verbose=0)
        r = results[0]

        # Process detections
        for i, class_id in enumerate(r['class_ids']):
            if class_id == PEDESTRIAN_CLASS_ID:
                y1, x1, y2, x2 = r['rois'][i]
                box_center_x = (x1 + x2) // 2

                # Determine bounding box color
                if CYCLIST_LANE_X1 <= box_center_x <= CYCLIST_LANE_X2:
                    color = (0, 0, 255)  # Red - In cyclist lane (DANGER)
                    alert_text = "⚠️ ALERT! Pedestrian in lane!"
                elif CYCLIST_LANE_X1 - 100 <= box_center_x <= CYCLIST_LANE_X2 + 100:
                    color = (0, 165, 255)  # Amber - Near lane (WARNING)
                    alert_text = "⚠️ Pedestrian near lane!"
                else:
                    color = (0, 255, 0)  # Green - Safe
                    alert_text = ""

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"Pedestrian {r['scores'][i]:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Show alert message
                if alert_text:
                    cv2.putText(img, alert_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return img

# Streamlit UI
st.title("🚴 Pedestrian Detection for Cyclists")
st.write("Detect pedestrians in real-time using your phone's camera.")

webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
