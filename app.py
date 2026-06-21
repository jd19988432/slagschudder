import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np

st.title("🎯 FoodeQ Slagmeter (Stip-Isolatie)")
st.write("Richt het vizier op de bovenste stip. De cijfers en driehoeken worden automatisch weggefilterd.")

class FoodeqStipCalculator(VideoTransformerBase):
    def __init__(self):
        # We gaan uit van de standaard diameter van de fysieke referentiestip (vaak 3.0 of 4.0 mm)
        self.referentie_stip_mm = 3.0 
        self.nauwkeurige_slag = 0.0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h_img, w_img, _ = img.shape
        center_x, center_y = int(w_img / 2), int(h_img / 2)
        
        # We maken een compact "Scan-Vak" (ROI) in het midden van het scherm
        # Dit vak is groot genoeg voor de trillende stip, maar sluit de cijfers eronder uit
        box_radius = 50
        ymin, ymax = center_y - box_radius, center_y + box_radius
        xmin, xmax = center_x - box_radius, center_x + box_radius
        
        # Teken het blauwe scan-kader op het scherm voor de technieker
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(img, "PLAATS HIER 1 STIP", (xmin, ymin - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Snijd alleen dit kleine vakje uit voor de computervisie (bespaart ook CPU!)
        roi = img[ymin:ymax, xmin:xmax]
        
        # Beeldbewerking op de ROI
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        grootste_contour = None
        max_area = 0
        
        # Zoek de meest duidelijke vorm (de trillende stip) binnen het scan-vak
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area and area > 150:
                max_area = area
                grootste_contour = cnt

        if grootste_contour is not None:
            # Bereken de meedraaiende box voor de schuine slag van de FoodeQ gút
            rect = cv2.minAreaRect(grootste_contour)
            (cx, cy), (w, h), angle = rect
            
            if w > h:
                w, h = h, w
            
            # Kalibratie op basis van de stilstaande breedte van de stip
            pixels_per_mm = w / self.referentie_stip_mm
            
            if pixels_per_mm > 0:
                self.nauwkeurige_slag = (h - w) / pixels_per_mm
                
                # Teken de groene box (omgerekend naar de coördinaten van het hele scherm)
                box = cv2.boxPoints(rect)
                box[:, 0] += xmin
                box[:, 1] += ymin
                box = np.int0(box)
                cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
                
                # Toon de meting live op het scherm
                cv2.putText(img, f"SLAG: {round(self.nauwkeurige_slag, 1)} mm", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        else:
            cv2.putText(img, "Zoeken naar stip...", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        return img

# Interface configuratie
st.sidebar.header("⚙️ Instellingen")
stip_maat = st.sidebar.number_input("Werkelijke diameter van de STIP (mm):", value=3.0, step=0.5)

ctx = webrtc_streamer(
    key="foodeq-stip-meter", 
    video_transformer_factory=FoodeqStipCalculator,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

if ctx.video_transformer:
    ctx.video_transformer.referentie_stip_mm = stip_maat
