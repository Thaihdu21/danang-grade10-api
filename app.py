# app.py
import pickle
import numpy as np
import json
import os
import unicodedata
import re
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ========== LOAD MODEL & DATA ==========
with open('diem_thpt_model.pkl', 'rb') as f:
    model = pickle.load(f)

scaler     = model['scaler']
reg_model  = model['regression_model']
clf_model  = model['classification_model']
class_names = model['class_names']

with open('diem-thi-lop10-danang-2026.json', 'r', encoding='utf-8') as f:
    raw = json.load(f)

students_list = raw if isinstance(raw, list) else raw.get('du_lieu', [])

# Index theo SBD
students_by_sbd = {str(s['so_bao_danh']).strip(): s for s in students_list}

# ========== HELPER ==========
def normalize(text: str) -> str:
    """Bỏ dấu, lowercase, xóa khoảng trắng thừa"""
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def build_features(toan, ngu_van, ngoai_ngu, mon_chuyen=0):
    tong_3 = toan + ngu_van + ngoai_ngu
    return np.array([[
        toan, ngu_van, ngoai_ngu, mon_chuyen,
        tong_3,
        tong_3 / 3,
        max(toan, ngu_van, ngoai_ngu),
        min(toan, ngu_van, ngoai_ngu),
        max(toan, ngu_van, ngoai_ngu) - min(toan, ngu_van, ngoai_ngu),
        1 if mon_chuyen > 0 else 0
    ]])

def student_summary(s: dict) -> dict:
    """Trả về thông tin tóm tắt của 1 thí sinh (dùng cho list)"""
    return {
        'so_bao_danh': s.get('so_bao_danh'),
        'ho_ten':      s.get('ho_ten'),
        'toan':        s.get('toan'),
        'ngu_van':     s.get('ngu_van'),
        'ngoai_ngu':   s.get('ngoai_ngu'),
        'mon_chuyen':  s.get('mon_chuyen'),
        'tong_dai_tra': s.get('tong_dai_tra'),
    }

# ========== ROUTES ==========

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'students': len(students_list)})


@app.route('/api/predict', methods=['POST'])
def predict():
    """Dự đoán điểm từ input thủ công"""
    data       = request.json or {}
    toan       = float(data.get('toan', 0))
    ngu_van    = float(data.get('ngu_van', 0))
    ngoai_ngu  = float(data.get('ngoai_ngu', 0))
    mon_chuyen = float(data.get('mon_chuyen', 0))

    features        = build_features(toan, ngu_van, ngoai_ngu, mon_chuyen)
    features_scaled = scaler.transform(features)

    pred_score = reg_model.predict(features_scaled)[0]
    pred_class = clf_model.predict(features_scaled)[0]

    return jsonify({
        'predicted_score': round(float(pred_score), 2),
        'xep_loai':        class_names[pred_class],
    })


@app.route('/api/search-name', methods=['GET'])
def search_name():
    """
    Tìm kiếm thí sinh theo tên (không dấu, không phân biệt hoa thường).
    Trả về:
      - results: danh sách tóm tắt (nếu nhiều kết quả)
      - total: số kết quả
    """
    query = request.args.get('q', '').strip()
    if len(query) < 2:
        return jsonify({'error': 'Nhập ít nhất 2 ký tự', 'results': [], 'total': 0}), 400

    q_norm = normalize(query)

    matched = [
        s for s in students_list
        if q_norm in normalize(s.get('ho_ten') or '')
    ]

    # Sắp xếp: tên khớp chính xác lên đầu, sau đó theo tổng điểm giảm dần
    matched.sort(key=lambda s: (
        normalize(s.get('ho_ten', '')) != q_norm,   # exact match lên đầu
        -(s.get('tong_dai_tra') or 0)
    ))

    return jsonify({
        'total':   len(matched),
        'results': [student_summary(s) for s in matched[:999999999999]],  # tối đa 50
    })


@app.route('/api/student/<sbd>', methods=['GET'])
def get_student(sbd: str):
    """Lấy toàn bộ thông tin 1 thí sinh theo SBD"""
    s = students_by_sbd.get(sbd) or students_by_sbd.get(sbd.zfill(5))
    if not s:
        return jsonify({'error': 'Không tìm thấy SBD'}), 404

    # Gọi ML predict nếu có đủ điểm
    ml = {}
    if all(s.get(k) is not None for k in ['toan', 'ngu_van', 'ngoai_ngu']):
        features        = build_features(
            float(s['toan']),
            float(s['ngu_van']),
            float(s['ngoai_ngu']),
            float(s.get('mon_chuyen') or 0)
        )
        features_scaled = scaler.transform(features)
        ml = {
            'ml_predicted_score': round(float(reg_model.predict(features_scaled)[0]), 2),
            'ml_xep_loai':        class_names[int(clf_model.predict(features_scaled)[0])],
        }

    return jsonify({**s, **ml})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
