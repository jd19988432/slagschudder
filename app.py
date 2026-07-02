import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
from streamlit_cropper import st_cropper

# Pagina-instellingen voor mobiel gebruik
st.set_page_config(page_title="FoodeQ Handmatige Slagmeter", layout="centered")

st.title("📹 FoodeQ Slagmeter (Met Frameselectie)")
st.write("Upload een korte video, kies een duidelijk frame en selecteer de cirkel.")

# --- SIDEBAR CONFIGURATIE ---
st.sidebar.header("⚙️ Instellingen")
stip_maat = st.sidebar.number_input(
    "Werkelijke diameter van de CIRKEL (mm):", 
    value=3.0, 
    step=0.5,
    help="De buitenste diameter van de cirkel in stilstand."
)

# --- VIDEO INPUT ---
geüploade_video = st.file_uploader("Upload of maak een video-opname", type=["mp4", "mov", "avi"])

if geüploade_video is not None:
    # Sla de video tijdelijk op
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(geüploade_video.read())
    
    # Open de video om de eigenschappen te lezen
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        total_frames = 90  # Fallback
        
    st.subheader("🎞️ Stap 1: Kies een duidelijk frame")
    
    # Interactieve schuifbalk om het beste frame te kiezen
    gekozen_frame_idx = st.slider(
        "Sleep de balk om door de video te bladeren naar een scherp beeld:", 
        min_value=1, 
        max_value=total_frames, 
        value=min(10, total_frames)  # Start standaard op frame 10 (vaak stabieler dan frame 1)
    )
    
    # Zet de videotijd op het gekozen frame en lees het uit
    cap.set(cv2.CAP_PROP_POS_FRAMES, gekozen_frame_idx - 1)
    ret, geselecteerd_frame = cap.read()
    cap.release()
    
    if ret:
        # Omzetten naar PIL Image voor de cropper
        frame_rgb = cv2.cvtColor(geselecteerd_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        st.subheader("📍 Stap 2: Sleep het vakje over de trillende cirkel")
        st.write("Zorg dat de volledige trillingsbaan (de hele ovaal) ruim binnen het groene vak valt.")
        
        # De cropper tool op het door de gebruiker gekozen frame
        cropped_pil_img = st_cropper(pil_img, realtime_update=True, box_color='#00FF00', aspect_ratio=None)
        
        if cropped_pil_img:
            # Vind de exacte locatie van de uitsnede in het originele frame via Template Matching
            cropped_cv_img = cv2.cvtColor(np.array(cropped_pil_img), cv2.COLOR_RGB2BGR)
            res = cv2.matchTemplate(geselecteerd_frame, cropped_cv_img, cv2.TM_SQDIFF)
            _, _, min_loc, _ = cv2.minMaxLoc(res)
            
            xmin, ymin = min_loc
            h_crop, w_crop, _ = cropped_cv_img.shape
            xmax, ymax = xmin + w_crop, ymin + h_crop
            
            st.subheader("🚀 Stap 3: Start de meting")
            if st.button("Analyseer Volledige Video"):
                # Open de video opnieuw voor de analyse van álle frames
                cap = cv2.VideoCapture(tfile.name)
                slagen_lijst = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.info("De video wordt nu geanalyseerd binnen jouw gekozen vak...")
                
                frame_count = 0
                voorbeeld_frame = None

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    frame_count += 1
                    
                    # Snijd de geselecteerde regio uit het huidige frame
                    roi = frame[ymin:ymax, xmin:xmax]
                    
                    # Computervisie bewerkingen op de ROI
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (5, 5), 0)
                    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    grootste_contour = None
                    max_area = 0
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area > max_area and area > 15:
                            max_area = area
                            grootste_contour = cnt
                    
                    if grootste_contour is not None:
                        rect = cv2.minAreaRect(grootste_contour)
                        (cx, cy), (w, h), angle = rect
                        
                        if w > h:
                            w, h = h, w
                        
                        if w > 0:
                            pixels_per_mm = w / stip_maat
                            slag_mm = (h - w) / pixels_per_mm
                            slagen_lijst.append(slag_mm)
                            
                            # Sla een visueel voorbeeld op (bijvoorbeeld van het frame dat de gebruiker zelf koos)
                            if frame_count == gekozen_frame_idx or voorbeeld_frame is None:
                                box = cv2.boxPoints(rect)
                                box = np.intp(box)
                                cv2.drawContours(roi, [box], 0, (0, 255, 0), 3)
                                frame[ymin:ymax, xmin:xmax] = roi
                                voorbeeld_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    progress = min(frame_count / total_frames, 1.0)
                    progress_bar.progress(progress)

                cap.release()
                progress_bar.progress(1.0)
                status_text.empty()

                # --- RESULTAAT WEERGEVEN ---
                if slagen_lijst:
                    gefilterde_slagen = [s for s in slagen_lijst if 0.5 < s < 20.0]
                    
                    if gefilterde_slagen:
                        gemiddelde_slag = np.mean(gefilterde_slagen)
                        
                        st.success(f"📋 Analyse succesvol afgerond!")
                        st.metric(label="Berekende Gemiddelde Slag", value=f"{round(gemiddelde_slag, 1)} mm")
                        
                        if voorbeeld_frame is not None:
                            st.image(voorbeeld_frame, caption=f"Meting gecontroleerd op jouw gekozen frame (nr. {gekozen_frame_idx})", use_container_width=True)
                    else:
                        st.warning("Geen stabiele metingen kunnen doen. Was de video stabiel gefilmd?")
                else:
                    st.error("⚠️ Kon de cirkel niet isoleren. Sleep het kader iets ruimer om de cirkel heen.")
    else:
        st.error("Kon de video of het gekozen frame niet laden.")
