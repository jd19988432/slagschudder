import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
from streamlit_cropper import st_cropper  # De juiste, up-to-date bibliotheek

# Pagina-instellingen voor mobiel gebruik
st.set_page_config(page_title="FoodeQ Handmatige Slagmeter", layout="centered")

st.title("📹 FoodeQ Slagmeter (Met Stip-Selectie)")
st.write("Upload een korte video en sleep het kader over de trillende stip.")

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
    cap.release()
    
    if ret:
        # Omzetten van OpenCV (BGR) naar PIL Image voor de cropper
        eerste_frame_rgb = cv2.cvtColor(eerste_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(eerste_frame_rgb)
        
        st.subheader("📍 Stap 1: Sleep het vakje over de trillende stip")
        st.write("Snijd de afbeelding zo bij dat ALLEEN de trillende stip (de ovaal) in het vak valt.")
        
        # De interactieve cropper tool (geeft direct de uitgesneden PIL Image terug)
        # We zetten aspect_ratio op None zodat je vrij kunt slepen
        cropped_pil_img = st_cropper(pil_img, realtime_update=True, box_color='#00FF00', aspect_ratio=None)
        
        if cropped_pil_img:
            # Bereken de coördinaten en de schaal van het uitgesneden stuk ten opzichte van het origineel
            # st_cropper geeft de uitgesneden afbeelding, we zoeken de locatie in het origineel
            cropped_cv_img = cv2.cvtColor(np.array(cropped_pil_img), cv2.COLOR_RGB2BGR)
            
            # Vind waar het uitgesneden stuk zich bevindt via sjabloon-matching (Template Matching)
            # Dit is de meest foutloze manier om de exacte ROI-coördinaten terug te krijgen
            res = cv2.matchTemplate(eerste_frame, cropped_cv_img, cv2.TM_SQDIFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            # Dit zijn de exacte pixelcoördinaten van jouw getrokken vakje
            xmin, ymin = min_loc
            h_crop, w_crop, _ = cropped_cv_img.shape
            xmax, ymax = xmin + w_crop, ymin + h_crop
            
            st.subheader("📍 Stap 2: Start de berekening")
            if st.button("🚀 Analyseer Video binnen dit vak"):
                # Open de video opnieuw voor de volledige frame-voor-frame analyse
                cap = cv2.VideoCapture(tfile.name)
                slagen_lijst = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.info("De video wordt nu specifiek binnen jouw selectievak geanalyseerd...")
                
                frame_count = 0
                voorbeeld_frame = None

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    frame_count += 1
                    
                    # Snijd exact jouw vakje uit elk frame van de video
                    roi = frame[ymin:ymax, xmin:xmax]
                    h_roi, w_roi, _ = roi.shape
                    
                    # Beeldbewerking ALLEEN op jouw geselecteerde stip
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
                            
                            # Sla een visueel voorbeeld op van de meting
                            if voorbeeld_frame is None:
                                box = cv2.boxPoints(rect)
                                box = np.intp(box)
                                cv2.drawContours(roi, [box], 0, (0, 255, 0), 3)
                                frame[ymin:ymax, xmin:xmax] = roi
                                voorbeeld_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
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
                            st.image(voorbeeld_frame, caption="Groene box staat nu exact op de stip in jouw selectievak", use_container_width=True)
                    else:
                        st.warning("Geen stabiele metingen kunnen doen. Was de video stabiel gefilmd?")
                else:
                    st.error("⚠️ Kon de stip niet isoleren. Sleep het kader iets ruimer om de stip heen.")
    else:
        st.error("Kon de video niet inladen.")
