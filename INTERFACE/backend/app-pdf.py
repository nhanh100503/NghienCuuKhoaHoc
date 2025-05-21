from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import fitz  # PyMuPDF
import io
import base64
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app, resources={r"/extract-images": {"origins": "http://localhost:5173"}})

@app.route('/extract-images', methods=['POST'])
def extract_images_from_pdf():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF file provided'}), 400

    pdf_file = request.files['pdf']
    
    try:
        images_data = []
        original_name = os.path.splitext(pdf_file.filename)[0]
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")

        global_img_index = 1
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pdf_page_number = page.number + 1
            image_list = page.get_images(full=True)

            # Find page number at the bottom
            page_dict = page.get_text("dict")
            blocks = page_dict["blocks"]
            page_number_found = None

            for block in blocks:
                if block["type"] == 0:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text.isdigit():
                                x0, y0, x1, y1 = span["bbox"]
                                if y0 > 750 and (x0 > 400 or (250 < x0 < 350)):
                                    page_number_found = text

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                width = base_image["width"]
                height = base_image["height"]

                try:
                    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    img_np = np.array(pil_img)
                    
                    # Check image size
                    too_small = width < 50 or height < 50
                    
                    # Check for monochrome
                    std_color = np.std(img_np)
                    is_monochrome = std_color < 10
                    
                    # Validate image
                    is_valid = not (too_small or is_monochrome)
                    
                    if not is_valid:
                        print(f"Image invalid: width={width}, height={height}, std_color={std_color}, monochrome={is_monochrome}, too_small={too_small}")
                    
                except Exception as e:
                    print(f"Error processing image: {e}")
                    is_valid = False

                # Find caption
                caption = ""
                for block in blocks:
                    if block["type"] == 1:  # Image block
                        image_bbox = block["bbox"]
                        ix0, iy0, ix1, iy1 = image_bbox

                        min_distance = float('inf')
                        for tblock in blocks:
                            if tblock["type"] == 0:  # Text block
                                for line in tblock["lines"]:
                                    line_text = ""
                                    line_y = None
                                    for span in line["spans"]:
                                        sx0, sy0, sx1, sy1 = span["bbox"]
                                        if line_y is None:
                                            line_y = sy0
                                        line_text += span["text"].strip() + " "

                                    if line_y and line_y > iy1:
                                        distance = line_y - iy1
                                        if distance <= 50:
                                            caption = line_text.strip()
                                            break
                                if caption:
                                    break

                if caption:
                    print(f"Found caption: {caption}")
                else:
                    print("No caption found.")

                b64_image = base64.b64encode(image_bytes).decode('utf-8')
                image_filename = f"img_{global_img_index}_page_{pdf_page_number}_.png"

                images_data.append({
                    'index': global_img_index,
                    'name': image_filename,
                    'base64': b64_image,
                    'caption': caption,
                    'page_number_position': page_number_found,
                    'is_valid': is_valid
                })

                global_img_index += 1

        return jsonify({'images': images_data}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5002)