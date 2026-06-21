import streamlit as st
import cv2
import numpy as np
import tempfile

# Pagina-instellingen voor mobiel gebruik
st.set_page_config(page_title="FoodeQ Video Slagmeter", layout="centered")

st.title("📹 FoodeQ Slagmeter via Video")
st.write("Maak een korte video (2-3 sec) van de trillende sticker en upload deze hierboven.")

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
    # Sla de geüploade video tijdelijk op
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(geüploade_video.read())
    
    # Open de video met OpenCV
    cap = cv2.VideoCapture(tfile.name)
    
    slagen_lijst = []
    
    # Gecorrigeerde OpenCV eigenschap om frames te tellen
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Robuuste fallback: als de smartphone/browser de framecount niet doorgeeft,
    # schatten we deze op basis van een video van ~3 seconden (ongeveer 90 frames)
    if total_frames <= 0:
        total_frames = 90
        
    # Voortgangsbalk en statusbericht tonen in Streamlit
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.info("De video wordt frame voor frame geanalyseerd...")
    
    frame_count = 0
    voorbeeld_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        h_img, w_img, _ = frame.shape
        
        # Beeldbewerking naar zwart/wit voor hoog contrast
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        frame_metingen = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 150 < area < (h_img * w_img * 0.1):
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (w, h), angle = rect
                
                if w > h:
                    w, h = h, w
                
                if w > 0 and (h / w) > 1.1:
                    pixels_per_mm = w / stip_maat
                    slag_mm = (h - w) / pixels_per_mm
                    frame_metingen.append((slag_mm, rect))
        
        # Neem de meest duidelijke meting uit dit frame
        if frame_metingen:
            beste_slag, beste_rect = max(frame_metingen, key=lambda x: x[0])
            slagen_lijst.append(beste_slag)
            
            # Sla het allereerste frame op als visueel voorbeeld
            if voorbeeld_frame is None:
                # GECORRIGEERD: Veilig omzetten van de box coördinaten naar 32-bit integers
                box = cv2.boxPoints(beste_rect)
                box = np.intp(box)  # np.intp vervangt het verouderde np.int0
                
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 4)
                voorbeeld_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
        # Update de progressbar
        progress = min(frame_count / total_frames, 1.0)
        progress_bar.progress(progress)

    cap.release()
    progress_bar.progress(1.0)
    status_text.empty() # Haal het laadbericht weg

    # --- RESULTAAT TONEN ---
    if slagen_lijst:
        # Filter extreme uitschieters (bijvoorbeeld door een schok van de hand)
        gefilterde_slagen = [s for s in slagen_lijst if 0.5 < s < 20.0]
        
        if gefilterde_slagen:
            gemiddelde_slag = np.mean(gefilterde_slagen)
            max_gemeten = np.max(gefilterde_slagen)
            
            st.success(f"📋 Analyse voltooid op basis van {len(gefilterde_slagen)} video-frames!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Gemiddelde Slag (Millimeters)", value=f"{round(gemiddelde_slag, 1)} mm")
            with col2:
                st.metric(label="Maximale uitslag in video", value=f"{round(max_gemeten, 1)} mm")
            
            if voorbeeld_frame is not None:
                st.image(voorbeeld_frame, caption="Visuele controle (Groene box op de stip)", use_container_width=True)
        else:
            st.warning("De video bevat frames, maar de waarden waren niet realistisch.")
    else:
        st.error("⚠️ Het is niet gelukt om de trillende stip in de video te isoleren.")
        st.write("Zorg ervoor dat de video scherp is, de sticker goed verlicht is en dat je dichtbij genoeg filmt.")
