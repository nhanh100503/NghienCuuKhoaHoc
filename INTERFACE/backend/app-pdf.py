import io
import os
import re
import base64
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  

CORS(app)  #

def extract_doi(text):
    doi_match = re.search(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', text, re.IGNORECASE)
    return doi_match.group(0) if doi_match else None

def extract_title_by_fontsize(page):
    max_fontsize = 0
    title_lines = []
    for obj in page.extract_words(use_text_flow=True, extra_attrs=["size"]):
        if obj["size"] > max_fontsize:
            max_fontsize = obj["size"]
    for obj in page.extract_words(use_text_flow=True, extra_attrs=["size"]):
        if abs(obj["size"] - max_fontsize) < 0.5:
            title_lines.append(obj["text"])
    return " ".join(title_lines).strip() if title_lines else "Không rõ tiêu đề"


def extract_authors_and_date(text):
    lines = text.split('\n')

    # 1. Tìm ngày: chỉ lấy phần ngày tháng, ví dụ 'Accepted 6 Oct 2024'
    approved_date = "Không rõ ngày"
    for line in lines:
        m = re.search(r"(Duyệt đăng|Ngày chấp nhận|Accepted)\s*[:\-]?\s*(\d{1,2}\s*\w+\s*\d{4})", line, re.IGNORECASE)
        if m:
            approved_date = f"{m.group(1).capitalize()} {m.group(2)}"
            break

    # 2. Tìm tác giả: lấy tối đa 2 dòng sau tiêu đề có dấu *, hoặc dấu phẩy, hoặc từ 'and'
    author_lines = []
    for i in range(1, min(len(lines), 6)):
        line = lines[i].strip()
        # Dòng có dấu * hoặc dấu , hoặc từ and là tác giả
        if '*' in line or ',' in line or re.search(r'\band\b', line, re.IGNORECASE):
            author_lines.append(line)
        else:
            if author_lines:
                break

    if author_lines:
        authors_raw = " ".join(author_lines)
        # Loại bỏ số mũ, dấu *, email, và phần địa chỉ thường bắt đầu bằng Faculty, University, Department ...
        authors_raw = re.sub(r'[\d¹²³⁴⁵⁶⁷⁸⁹⁰*]+', '', authors_raw)
        authors_raw = re.sub(r'\S+@\S+', '', authors_raw)  # loại email
        # Cắt bỏ phần sau từ khóa địa chỉ phổ biến
        authors_raw = re.split(r'\b(Faculty|University|Department|Institute|School|Center)\b', authors_raw)[0]
        authors_clean = re.sub(r'\s+', ' ', authors_raw).strip()
    else:
        authors_clean = "Không rõ tác giả"

    return authors_clean, approved_date



@app.route('/extract-images', methods=['POST'])
def extract_images_from_pdf():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF file provided'}), 400

    pdf_file = request.files['pdf']

    try:
        images_data = []
        original_name = os.path.splitext(pdf_file.filename)[0]

        # Đọc pdf_file thành bytes để mở bằng 2 thư viện
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pdf_file_stream = io.BytesIO(pdf_bytes)
        pdf_plumber_doc = pdfplumber.open(pdf_file_stream)

        real_pages = {}
        captions_by_page = {}
        global_img_index = 1

        title = "Không rõ tiêu đề"
        authors = "Không rõ tác giả"
        doi = None
        approved_date = "Không rõ ngày"

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pdf_page_number = page.number + 1
            image_list = page.get_images(full=True)

            pdfplumber_page = pdf_plumber_doc.pages[page_num]
            text = pdfplumber_page.extract_text()

            if text and page_num == 0:
                title = extract_title_by_fontsize(pdfplumber_page)
                authors, approved_date = extract_authors_and_date(text)
                doi = extract_doi(text)

            # Dùng pdfplumber_page để lấy từ nằm dưới đáy trang (bottom_texts)
            words = pdfplumber_page.extract_words()
            bottom_texts = [
                obj for obj in words if obj["top"] > (pdfplumber_page.height * 0.85)
            ]

            center_x = pdfplumber_page.width / 2
            closest_to_center = None
            min_distance = float("inf")

            for word in bottom_texts:
                word_center = (word["x0"] + word["x1"]) / 2
                if re.fullmatch(r"\d{1,4}", word["text"]):
                    distance = abs(center_x - word_center)
                    if distance < min_distance:
                        min_distance = distance
                        closest_to_center = word["text"]

            real_page = int(closest_to_center) if closest_to_center else page_num + 1
            real_pages[page_num + 1] = real_page

            captions = [line.strip() for line in text.split('\n') if re.search(r'(hình|figure)\s*\d+', line.lower())]
            captions_by_page[page_num + 1] = captions

            page_dict = pdfplumber_page.to_dict()
            blocks = page_dict.get("blocks", [])

            page_number_found = None

            for block in blocks:
                if block["type"] == 0:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            txt = span.get("text", "").strip()
                            if txt.isdigit():
                                x0, y0, x1, y1 = span.get("bbox", (0,0,0,0))
                                if y0 > 750 and (x0 > 400 or (250 < x0 < 350)):
                                    page_number_found = txt

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                width = base_image["width"]
                height = base_image["height"]

                try:
                    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    img_np = np.array(pil_img)

                    # Kiểm tra kích thước ảnh nhỏ
                    too_small = width < 50 or height < 50

                    # Kiểm tra ảnh đơn sắc (monochrome) dựa trên độ lệch chuẩn màu
                    std_color = np.std(img_np)
                    is_monochrome = std_color < 10

                    # Ảnh hợp lệ khi không quá nhỏ và không đơn sắc
                    is_valid = not (too_small or is_monochrome)

                except Exception as e:
                    print(f"Error processing image: {e}")
                    is_valid = False

                # Tìm caption gần ảnh (cách dưới ảnh < 50)
                caption = ""
                for block in blocks:
                    if block["type"] == 1:  # Image block
                        image_bbox = block["bbox"]
                        ix0, iy0, ix1, iy1 = image_bbox

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
                        break

                b64_image = base64.b64encode(image_bytes).decode('utf-8')
                image_filename = f"img_{global_img_index}_page_{pdf_page_number}.png"

                images_data.append({
                    'index': global_img_index,
                    'name': image_filename,
                    'base64': b64_image,
                    'caption': caption,
                    'page_number_position': page_number_found,
                    'is_valid': is_valid,
                    'title': title,
                    'doi': doi,
                    'authors': authors,
                    'approved_date': approved_date
                })
                global_img_index += 1

        return jsonify({'images': images_data}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5002)
