#!/usr/bin/env python3
"""
Generate an ArUco marker (ID=6) sized to HALF of an A4 page (i.e., A5).
Outputs:
  - aruco_id6_A5.png  (300 DPI raster)
  - aruco_id6_A5.pdf  (vector PDF with exact A5 page size)
Print the PDF at 100% (no scaling) for correct physical dimensions.
"""

import cv2
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A5  # A5 = half of A4
from reportlab.lib.utils import ImageReader

# ------------------ CONFIG ------------------
ARUCO_ID = 6
# Choose the dictionary you use in your detector
DICT_NAME = "DICT_6X6_50"

PAGE_SIZE = A5  # (width, height) in points (1 pt = 1/72 inch)
DPI = 300      # Print resolution for PNG export
BORDER_MM = 10.0  # White border (in mm) around the marker
# --------------------------------------------

def get_aruco_dict(dict_name: str):
    """Map string to cv2.aruco dictionary."""
    d = cv2.aruco
    table = {
        "DICT_4X4_50":       d.DICT_4X4_50,
        "DICT_4X4_100":      d.DICT_4X4_100,
        "DICT_5X5_50":       d.DICT_5X5_50,
        "DICT_5X5_100":      d.DICT_5X5_100,
        "DICT_6X6_50":       d.DICT_6X6_50,
        "DICT_6X6_100":      d.DICT_6X6_100,
        "DICT_6X6_250":      d.DICT_6X6_250,
        "DICT_7X7_50":       d.DICT_7X7_50,
        "DICT_7X7_100":      d.DICT_7X7_100,
        "DICT_7X7_250":      d.DICT_7X7_250,
        "APRILTAG_36h11":    d.DICT_APRILTAG_36h11,
    }
    if dict_name not in table:
        raise ValueError(f"Unknown dictionary name: {dict_name}")
    return d.getPredefinedDictionary(table[dict_name])

def mm_to_inches(mm): 
    return mm / 25.4

def main():
    # --- Page dimensions in pixels (for PNG) ---
    page_w_pt, page_h_pt = PAGE_SIZE
    page_w_in = page_w_pt / 72.0
    page_h_in = page_h_pt / 72.0
    page_w_px = int(round(page_w_in * DPI))
    page_h_px = int(round(page_h_in * DPI))

    # Marker size in pixels (leave margins)
    border_in = mm_to_inches(BORDER_MM)
    border_px = int(round(border_in * DPI))
    marker_side_px = min(page_w_px, page_h_px) - 2 * border_px
    marker_side_px = max(marker_side_px, 100)  # safety lower bound

    # Create blank white page
    page = np.full((page_h_px, page_w_px), 255, dtype=np.uint8)

    # Generate the marker
    aruco_dict = get_aruco_dict(DICT_NAME)
    marker = cv2.aruco.drawMarker(aruco_dict, ARUCO_ID, marker_side_px)

    # Place marker in the center of the page
    y0 = (page_h_px - marker_side_px) // 2
    x0 = (page_w_px - marker_side_px) // 2
    page[y0:y0 + marker_side_px, x0:x0 + marker_side_px] = marker

    # Save PNG (300 DPI)
    png_path = f"aruco_id{ARUCO_ID}_A5.png"
    cv2.imwrite(png_path, page)

    # --- Create A5 PDF ---
    # Convert grayscale numpy array to RGB PIL image
    page_rgb = cv2.cvtColor(page, cv2.COLOR_GRAY2RGB)
    pil_img = Image.fromarray(page_rgb)

    # Wrap PIL image for ReportLab
    img_reader = ImageReader(pil_img)

    pdf_path = f"aruco_id{ARUCO_ID}_A5.pdf"
    c = canvas.Canvas(pdf_path, pagesize=PAGE_SIZE)
    c.drawImage(img_reader, 0, 0, width=page_w_pt, height=page_h_pt,
                preserveAspectRatio=False, mask='auto')
    c.showPage()
    c.save()

    print(f"Saved: {png_path} ({page_w_px}x{page_h_px} px @ {DPI} DPI)")
    print(f"Saved: {pdf_path} (A5 true size)")

if __name__ == "__main__":
    main()
