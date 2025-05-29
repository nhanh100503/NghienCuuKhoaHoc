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
from collections import defaultdict
import dateparser

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  

CORS(app)  #

def extract_doi(text):
    doi_match = re.search(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', text, re.IGNORECASE)
    return doi_match.group(0) if doi_match else None


def extract_title_by_fontsize(page):
    words = page.extract_words(use_text_flow=True, extra_attrs=["size", "top"])

    # Tìm max font size trong page
    max_fontsize = max(w["size"] for w in words)

    # Nhóm từ theo dòng (theo top), chỉ lấy những dòng có font size gần max
    line_groups = defaultdict(list)
    tolerance = 2  # khoảng cách y nhỏ để nhóm cùng dòng

    for w in words:
        if abs(w["size"] - max_fontsize) < 0.5:
            # Tìm dòng đã tồn tại gần bằng top này chưa
            line_key = None
            for k in line_groups:
                if abs(k - w["top"]) < tolerance:
                    line_key = k
                    break
            if line_key is None:
                line_key = w["top"]
            line_groups[line_key].append(w)

    if not line_groups:
        return "Không rõ tiêu đề", None

    # Sắp xếp các dòng theo top (từ trên xuống)
    sorted_tops = sorted(line_groups.keys())

    # Gom tất cả từ trong tất cả các dòng này thành tiêu đề đầy đủ
    title_text = " ".join(
        " ".join(w["text"] for w in line_groups[top])
        for top in sorted_tops
    ).strip()

    # Lấy vị trí dòng cuối cùng
    last_line_top = sorted_tops[-1]

    return title_text, last_line_top


def extract_authors_and_date(page, title_line_top):
    words = page.extract_words(use_text_flow=True, extra_attrs=["size", "top", "fontname", "x0", "x1"])

    tolerance = 2
    line_groups = defaultdict(list)

    for w in words:
        line_key = None
        for k in line_groups:
            if abs(k - w["top"]) < tolerance:
                line_key = k
                break
        if line_key is None:
            line_key = w["top"]
        line_groups[line_key].append(w)

    sorted_line_tops = sorted(line_groups.keys())

    # Tìm dòng bắt đầu lấy tác giả (dòng đầu tiên có top > title_line_top)
    author_start_idx = None
    for i, top in enumerate(sorted_line_tops):
        if top > title_line_top:
            author_start_idx = i
            break

    if author_start_idx is None:
        return "Không rõ tác giả", "Date unknown"

    def is_italic_line(words_line):
        for w in words_line:
            fontname = w.get("fontname", "").lower()
            if "italic" in fontname or "oblique" in fontname:
                return True
        return False

    def line_text_from_words(words_line, max_gap=2):
        # Sắp xếp từ theo vị trí x0 (trái sang phải)
        words_sorted = sorted(words_line, key=lambda w: w["x0"])
        text = words_sorted[0]["text"]
        for i in range(1, len(words_sorted)):
            gap = words_sorted[i]["x0"] - words_sorted[i-1]["x1"]
            if gap < max_gap:
                # Nối liền không thêm dấu cách
                text += words_sorted[i]["text"]
            else:
                # Thêm dấu cách
                text += " " + words_sorted[i]["text"]
        return text

    author_lines = []
    stop_keywords = ["TÓM TẮT", "ABSTRACT", "SUMMARY", "KEYWORDS"]

    for idx in range(author_start_idx, len(sorted_line_tops)):
        line_words = line_groups[sorted_line_tops[idx]]
        if is_italic_line(line_words):
            # Bỏ qua dòng in nghiêng, không thêm
            continue
        line_text = line_text_from_words(line_words)
        # Nếu dòng rỗng hoặc chứa từ khóa dừng, thì dừng lấy
        if not line_text.strip():
            break
        if any(k.lower() in line_text.lower() for k in stop_keywords):
            break
        author_lines.append(line_text)

    authors_raw = " ".join(author_lines)
    authors_raw = re.sub(r'[\d¹²³⁴⁵⁶⁷⁸⁹⁰*]+', '', authors_raw)
    authors_raw = re.sub(r'\S+@\S+', '', authors_raw)
    authors_raw = re.split(r'\b(Faculty|University|Department|Institute|School|Center|Khoa|Trường)\b', authors_raw)[0]
    authors_clean = re.sub(r'\s+', ' ', authors_raw).strip()
    if not authors_clean:
        authors_clean = "Không rõ tác giả"

    full_text = page.extract_text()
    approved_date = "Date unknown"

    approval_keywords = r"(Duyệt đăng|Ngày chấp nhận|Ngày duyệt đăng|Accepted|Approval date|Accepted date|Date accepted|Duyệt đăng \(Accepted\))"

    # Các định dạng ngày hỗ trợ
    date_patterns = [
        r"\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}",           
        r"\d{1,2}\s+\w+\s+\d{4}",                           
        r"\w+\s+\d{1,2},?\s+\d{4}",                        
    ]

    found = False
    for line in full_text.split("\n"):
        for date_pattern in date_patterns:
            pattern = rf"{approval_keywords}[\s:–\-]*({date_pattern})"
            m = re.search(pattern, line, re.IGNORECASE)
            if m:
                approved_date = m.group(2).strip()  # Chỉ lấy phần ngày
                found = True
                break
        if found:
            break

    return authors_clean, approved_date



def is_image_too_small(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return image.size[0] < 100 or image.size[1] < 100
    except Exception:
        return True

def is_black_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.sum(gray < 30) / gray.size > 0.9

def is_white_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.sum(gray > 245) / gray.size > 0.99

def check_and_repair_image(image_bytes):
    try:
        np_img = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        # if image is None or image.size == 0:
        #     return None, "decode_failed"
        # if is_image_too_small(image_bytes):
        #     return None, "too_small"
        if is_black_image(image):
            return None, "too_black"
        # if is_white_image(image):
        #     return None, "too_white"
        repaired_img_bytes = cv2.imencode('.png', image)[1]
        return repaired_img_bytes.tobytes(), "viewable"
    except Exception:
        return None, "error"

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
            pdfplumber_page = pdf_plumber_doc.pages[page_num]  # pdfplumber

            text = pdfplumber_page.extract_text()

            if text and page_num == 0: 
                title, title_line_top  = extract_title_by_fontsize(pdfplumber_page)
                authors, approved_date = extract_authors_and_date(pdfplumber_page, title_line_top)
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
                
                if page_num == 0 and img_index == 0:
                    continue  # bỏ ảnh đầu tiên

                repaired_image, status = check_and_repair_image(image_bytes)
                if status != "viewable":
                    continue

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
