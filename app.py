import streamlit as st
import cv2
import numpy as np
import tempfile
from streamlit_image_crop import streamlit_image_crop

# Pagina-instellingen voor mobiel gebruik
st.set_page_config(page_title="FoodeQ Handmatige Slagmeter", layout="centered")

st.title("📹 FoodeQ Slagmeter (Met Stip-Selectie)")
st.write("Upload je video en geef daarna handmatig aan waar de stip zit.")

# --- SIDEBAR CONFIGURATIE ---
st.sidebar.header("⚙️ Instellingen")
stip_maat = st.sidebar.number_input(
    "Werkelijke diameter van de STIP (mm):", 
    value=3.0, 
    step=0.5,
    help="De breedte van de zwarte stip in stilstand. Meestal is dit 3.0 of 4.0 mm."
)

# --- VIDEO INPUT ---
geüploade_video = st.file_uploader("Upload of maak een video-opname", type=["mp4", "mov", "avi"])

if geüploade_video is not None:
    # Sla de video tijdelijk op om deze uit te kunnen lezen
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(geüploade_video.read())
    
    # Open de video om het allereerste frame te pakken voor de selectie
    cap = cv2.VideoCapture(tfile.name)
    ret, eerste_frame = cap.read()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release() # Sluit hem direct weer af voor hergebruik zo meteen
    
    if ret:
        # Zet het eerste frame om naar RGB zodat Streamlit het begrijpt
        eerste_frame_rgb = cv2.cvtColor(eerste_frame, cv2.COLOR_BGR2RGB)
        
        st.subheader("📍 Stap 1: Sleep het vakje over de trillende stip")
        st.write("Gebruik de tool hieronder om in te zoomen op alléén de stip die je wilt meten.")
        
        # De interactieve cropper. Dit geeft de coördinaten (ROI) terug van wat je selecteert
        cropped_box = streamlit_image_crop(
            eerste_frame_rgb,
            box_color='#00FF00',
            aspect_ratio=None # Vrij slepen, geen vaste vierkant-verhouding verplicht
        )
        
        # Als de gebruiker een selectie heeft gemaakt, kunnen we gaan rekenen
        if cropped_box:
            # Haal de exacte pixel-coördinaten op uit de selectie tool
            # De tool geeft percentages of direct pixels terug, we rekenen het om naar exacte matrix-indices
            h_orig, w_orig, _ = eerste_frame.shape
            
            xmin = int(cropped_box['left'] * w_orig)
            ymin = int(cropped_box['top'] * h_orig)
            xmax = int((cropped_box['left'] + cropped_box['width']) * w_orig)
            ymax = int((cropped_box['top'] + cropped_box['height']) * h_orig)
            
            # Zorg dat de waarden binnen de video-grenzen blijven
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(w_orig, xmax), min(h_orig, ymax)
            
            if st.button("🚀 Start Analyse op Geselecteerde Stip"):
                # Open de video opnieuw voor de volledige frame-voor-frame analyse
                cap = cv2.VideoCapture(tfile.name)
                slagen_lijst = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.info("De geselecteerde stip wordt nu frame voor frame geanalyseerd...")
                
                frame_count = 0
                voorbeeld_frame = None

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    frame_count += 1
                    
                    # KRUCIALESTAP: We snijden ELK frame exact zo uit als jij hebt aangegeven
                    roi = frame[ymin:ymax, xmin:xmax]
                    h_roi, w_roi, _ = roi.shape
                    
                    # Beeldbewerking ALLEEN op het kleine uitgesneden vakje
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (5, 5), 0)
                    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Omdat het vakje nu heel klein is, is de grootste contour gegarandeerd jouw stip!
                    grootste_contour = None
                    max_area = 0
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area > max_area and area > 20: # Zelfs kleine selecties werken nu
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
                            
                            # Sla een visueel voorbeeld op van het eerste frame
                            if voorbeeld_frame is None:
                                box = cv2.boxPoints(rect)
                                box = np.intp(box)
                                # Teken de groene box in de ROI
                                cv2.drawContours(roi, [box], 0, (0, 255, 0), 3)
                                # Plak de ROI terug in het originele frame voor een mooi screenshot
                                frame[ymin:ymax, xmin:xmax] = roi
                                voorbeeld_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Update voortgangsbalk
                    if total_frames > 0:
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
                            st.image(voorbeeld_frame, caption="Visuele controle (Groene box staat nu exact in jouw selectievak)", use_container_width=True)
                    else:
                        st.warning("Geen realistische metingen kunnen doen binnen het geselecteerde vak.")
                else:
                    st.error("⚠️ Het lukte niet om de stip binnen jouw geselecteerde vak te isoleren. Maak het vak iets ruimer om de trillende stip heen.")
    else:
        st.error("Kon de video niet openen om het eerste frame te laden.")
