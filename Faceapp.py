import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.set_page_config(page_title="Face Mask Detection", layout="centered")

st.markdown("""
<h1 style='text-align:center;
color:white;
padding:20px;
border-radius:15px;
background:linear-gradient(to right,#0f2027,#203a43,#2c5364);'>
😷 Face Mask Detection System
</h1>
""", unsafe_allow_html=True)

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("Newface_mask_model.h5", compile=False)

model = load_my_model()

net = cv2.dnn.readNet(
    "res10_300x300_ssd_iter_140000.caffemodel",
    "deploy.prototxt"
)

class VideoProcessor(VideoProcessorBase):

    previous_label = "No Mask"

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        h, w = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        net.setInput(blob)

        detections = net.forward()

        for i in range(detections.shape[2]):

            confidence_face = detections[0, 0, i, 2]

            if confidence_face > 0.5:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

                (x1, y1, x2, y2) = box.astype("int")

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                face = image[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                img = cv2.resize(face_rgb, (224, 224))
                img = img.astype("float32") / 255.0
                img = np.expand_dims(img, axis=0)

                pred = model.predict(img, verbose=0)[0][0]

                if pred > 0.70:
                    label = "No Mask"
                    color = (0, 0, 255)

                elif pred < 0.30:
                    label = "Mask"
                    color = (0, 255, 0)

                else:
                    label = self.previous_label

                    if label == "No Mask":
                        color = (0, 0, 255)
                    else:
                        color = (0, 255, 0)

                self.previous_label = label

                if label == "No Mask":
                    confidence = pred * 100
                    text = f"No Mask ({confidence:.2f}%)"
                else:
                    confidence = (1 - pred) * 100
                    text = f"Wearing Mask ({confidence:.2f}%)"

                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

                cv2.putText(
                    image,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

        return av.VideoFrame.from_ndarray(image, format="bgr24")

webrtc_streamer(
    key="mask-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)