import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np
import av  # Belangrijk: zorgt voor de juiste videoframe-afhandeling

# Pagina-instellingen voor mobiel gebruik
st.set_page_config(page_title="FoodeQ Slagmeter", layout="centered")

st.title("🎯 FoodeQ Slagmeter Pro")
st.write("Richt het blauwe vizier op de bovenste zwarte stip van de sticker.")

# --- SIDEBAR CONFIGURATIE ---
st.sidebar.header("⚙️ Instellingen")
stip_maat = st.sidebar.number_input(
    "Werkelijke diameter van de STIP (mm):", 
    value=3.0, 
    step=0.5,
    help="Meet de breedte van de stip als de machine stilstaat. Meestal is dit 3.0 of 4.0 mm."
)

# --- MODERN VIDEO PROCESSOR CLASS ---
class FoodeqStipProcessor:
    def __init__(self):
        self.referentie_stip_mm = 3.0
        self.nauwkeurige_slag = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Zet het binnenkomende 'av' frame om naar een OpenCV numpy array
        img = frame.to_ndarray(format="bgr24")
        h_img, w_img, _ = img.shape
        center_x, center_y = int(w_img / 2), int(h_img / 2)
        
        # Region of Interest (ROI): Een compact scan-vak in het centrum
        box_radius = 50
        ymin, ymax = center_y - box_radius, center_y + box_radius
        xmin, xmax = center_x - box_radius, center_x + box_radius
        
        # Teken het blauwe scan-kader op het scherm
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(img, "PLAATS HIER 1 STIP", (xmin, ymin - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Snijd de ROI uit voor de computervisie-analyse
        roi = img[ymin:ymax, xmin:xmax]
        
        # Beeldbewerking naar zwart/wit
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        grootste_contour = None
        max_area = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area and area > 150:
                max_area = area
                grootste_contour = cnt

        if grootste_contour is not None:
            # Bereken de meedraaiende box (voor schuine trillingen)
            rect = cv2.minAreaRect(grootste_contour)
            (cx, cy), (w, h), angle = rect
            
            if w > h:
                w, h = h, w
            
            # Kalibratie
            pixels_per_mm = w / self.referentie_stip_mm
            
            if pixels_per_mm > 0:
                self.nauwkeurige_slag = (h - w) / pixels_per_mm
                
                # Teken de meedraaiende groene box
                box = cv2.boxPoints(rect)
                box[:, 0] += xmin
                box[:, 1] += ymin
                box = np.int0(box)
                cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
                
                # Toon de live meting op het camerabeeld
                cv2.putText(img, f"SLAG: {round(self.nauwkeurige_slag, 1)} mm", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)
        else:
            cv2.putText(img, "Zoeken naar stip...", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        # Geef het bewerkte frame weer netjes terug in het juiste 'av' formaat
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- UPDATED STREAMLIT WEB-RTC CAMERASTREAM ---
ctx = webrtc_streamer(
    key="foodeq-stip-meter", 
    # Gebruik de moderne 'video_processor_factory' in plaats van transformer
    video_processor_factory=FoodeqStipProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False}
)

# Geef de sidebar-instelling live door aan de video_processor (moderne manier)
if ctx.video_processor:
    ctx.video_processor.referentie_stip_mm = stip_maat
