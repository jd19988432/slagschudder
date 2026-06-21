import streamlit as st
import cv2
import numpy as np

# Pagina-instellingen voor mobiel gebruik
st.set_page_config(page_title="FoodeQ Slagmeter Pro", layout="centered")

st.title("🎯 FoodeQ Slagmeter Pro")
st.write("Maak een foto van de trillende sticker en upload deze hieronder om de slag te berekenen.")

# --- INSTELLINGEN IN DE SIDEBAR ---
st.sidebar.header("⚙️ Instellingen")
stip_maat = st.sidebar.number_input(
    "Werkelijke diameter van de STIP (mm):", 
    value=3.0, 
    step=0.5,
    help="De breedte van de zwarte stip in stilstand. Meestal is dit 3.0 of 4.0 mm."
)

# --- FOTO INPUT (Activeert de telefoon-camera) ---
# 'camera' dwingt de telefoon om de camera-app te openen. 
# Je kunt ook een bestaande foto uit je galerij kiezen.
geüploade_foto = st.camera_input("Maak een foto van de sticker")

if geüploade_foto is not None:
    # 1. Converteer de geüploade foto naar een OpenCV afbeelding
    file_bytes = np.asarray(bytearray(geüploade_foto.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h_img, w_img, _ = img.shape
    
    st.info("Afbeelding ontvangen. Analyse wordt uitgevoerd...")
    
    # 2. Beeldbewerking naar zwart/wit voor hoog contrast
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. Zoek naar de contouren (de uitgerekte stippen door de trilling)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    geldige_metingen = []
    
    for cnt in contours:
        # Filter op grootte zodat we geen letters of kleine stofjes meten
        area = cv2.contourArea(cnt)
        if 200 < area < (h_img * w_img * 0.05): # Mag niet de hele foto beslaan
            
            # Bereken de meedraaiende box (voor de schuine slag)
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect
            
            # Zorg dat 'w' de kortste zijde is (de onveranderde breedte van de stip)
            if w > h:
                w, h = h, w
            
            # Controleer of de vorm inderdaad is uitgerekt (een ovaal is geworden door trilling)
            # Een perfecte cirkel heeft een verhouding van 1.0. Een trillende stip is langer.
            if w > 0 and (h / w) > 1.1:
                pixels_per_mm = w / stip_maat
                slag_mm = (h - w) / pixels_per_mm
                
                # Sla de geldige meting op
                geldige_metingen.append({
                    "slag": slag_mm,
                    "rect": rect,
                    "cx": cx,
                    "cy": cy
                })

    # 4. Toon de resultaten aan de technieker
    if geldige_metingen:
        # Sorteer zodat we de meest duidelijke meting (meest centrale of grootste) bovenaan hebben
        best_match = max(geldige_metingen, key=lambda x: x['slag'])
        
        # Teken de groene meedraaiende box op de originele foto
        box = cv2.boxPoints(best_match["rect"])
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (0, 255, 0), 4) # Dikkere lijn voor betere zichtbaarheid op mobiel
        
        # Zet de foto om naar RGB voor Streamlit weergave
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Grote duidelijke weergave van het resultaat
        st.success(f"📋 Berekende Slag: **{round(best_match['slag'], 1)} mm**")
        
        # Toon de geanalyseerde foto met de groene box eromheen
        st.image(img_rgb, caption="Geanalyseerde sticker (Groene box = gedetecteerde slag)", use_container_width=True)
    else:
        st.warning("⚠️ Geen duidelijke trillende stip kunnen detecteren.")
        st.write("Toon de sticker dichterbij en zorg dat de foto goed scherp is (niet bewogen door je handen).")
