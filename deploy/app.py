"""Demo dự đoán mức hài lòng của hành khách (E7).

Chạy từ thư mục gốc repo:  streamlit run deploy/app.py
Cần deploy/model_lgb.txt và deploy/meta.json (tạo bằng python deploy/train_model.py).
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
META = json.loads((HERE / "meta.json").read_text(encoding="utf-8"))
FEATURES, CAT_COLS, SCORE_COLS, NUM_COLS = META["features"], META["cat_cols"], META["score_cols"], META["num_cols"]

VI = {  # tên hiển thị
    "Gender": "Giới tính", "Customer Type": "Loại khách", "Type of Travel": "Mục đích chuyến đi", "Class": "Hạng ghế",
    "Inflight wifi service": "Wifi trên máy bay", "Departure/Arrival time convenient": "Giờ đi / đến thuận tiện",
    "Ease of Online booking": "Đặt vé online", "Gate location": "Vị trí cổng", "Food and drink": "Đồ ăn, thức uống",
    "Online boarding": "Làm thủ tục online", "Seat comfort": "Ghế ngồi", "Inflight entertainment": "Giải trí",
    "On-board service": "Phục vụ trên máy bay", "Leg room service": "Chỗ để chân", "Baggage handling": "Hành lý",
    "Checkin service": "Check-in", "Cleanliness": "Vệ sinh", "Age": "Tuổi", "Flight Distance": "Quãng đường bay",
    "Departure Delay in Minutes": "Trễ khởi hành (phút)", "Arrival Delay in Minutes": "Trễ hạ cánh (phút)",
}


@st.cache_resource
def load_model():
    return lgb.Booster(model_file=str(HERE / "model_lgb.txt"))


def to_frame(df):
    X = df[FEATURES].copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=META["categories"][c])
    for c in SCORE_COLS + NUM_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def vn(x, spec):
    """Định dạng số kiểu Việt Nam: dấu phẩy thập phân, dấu chấm phân cách hàng nghìn."""
    return format(x, spec).translate(str.maketrans(",.", ".,"))


def predict(df):
    return load_model().predict(to_frame(df))


def contributions(row_df):
    """Đóng góp của từng feature vào logit (SHAP của LightGBM); cột cuối là giá trị nền."""
    c = load_model().predict(to_frame(row_df), pred_contrib=True)[0]
    return pd.Series(c[:-1], index=FEATURES), c[-1]


st.set_page_config(page_title="Dự đoán hài lòng hành khách", layout="wide")
st.title("Dự đoán mức hài lòng của hành khách hàng không")
st.caption(f"Model: LightGBM trên 21 cột gốc, {META['n_trees']} cây, train trên {vn(META['train_rows'], ',')} dòng. "
           f"CV AUC 5 fold: {vn(META['cv_auc'], '.4f')}. Đồ án CS116, cuộc thi Kaggle Playground S6E10.")

tab1, tab2, tab3 = st.tabs(["Một hành khách", "Dự đoán từ file CSV", "Về model"])

with tab1:
    with st.form("passenger"):
        c1, c2, c3 = st.columns(3)
        row = {}
        with c1:
            st.subheader("Thông tin chuyến bay")
            for c in CAT_COLS:
                row[c] = st.selectbox(VI[c], META["categories"][c])
            for c in NUM_COLS:
                lo, hi, med = META["num_ranges"][c]
                row[c] = st.number_input(VI[c], min_value=0.0, max_value=float(hi), value=float(med), step=1.0)
        half = len(SCORE_COLS) // 2 + 1
        for col, cols in ((c2, SCORE_COLS[:half]), (c3, SCORE_COLS[half:])):
            with col:
                st.subheader("Điểm dịch vụ (0 = không áp dụng)" if col is c2 else " ")
                for c in cols:
                    row[c] = st.slider(VI[c], 0, 5, 3)
        submitted = st.form_submit_button("Dự đoán")
    if submitted:
        df = pd.DataFrame([row])
        p = float(predict(df)[0])
        m1, m2 = st.columns([1, 2])
        with m1:
            st.metric("Xác suất hài lòng", vn(p, ".1%"))
            st.progress(p)
            st.caption(f"Tỉ lệ hài lòng chung trong dữ liệu train: {vn(META['positive_rate'], '.1%')}.")
        with m2:
            contrib, base = contributions(df)
            top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:8]
            show = pd.DataFrame({"feature": [VI[f] for f in top.index], "giá trị": [str(row[f]) for f in top.index],
                                 "đóng góp (logit)": top.values.round(3)})
            st.write("Các feature ảnh hưởng nhiều nhất tới dự đoán này (dương: tăng khả năng hài lòng):")
            st.bar_chart(show.set_index("feature")["đóng góp (logit)"], horizontal=True)
            st.dataframe(show, hide_index=True, width="stretch")

with tab2:
    st.write("File CSV cần có đủ 21 cột như `test.csv` của cuộc thi (cột `id` nếu có sẽ được giữ lại).")
    up = st.file_uploader("Chọn file CSV", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Thiếu cột: {', '.join(missing)}")
        else:
            out = pd.DataFrame({"id": df["id"]}) if "id" in df.columns else pd.DataFrame(index=df.index)
            out["satisfaction"] = predict(df)
            st.success(f"Đã dự đoán {vn(len(out), ',')} dòng; xác suất hài lòng trung bình "
                       f"{vn(out['satisfaction'].mean(), '.1%')}.")
            st.dataframe(out.head(20), width="stretch")
            st.download_button("Tải kết quả (CSV)", out.to_csv(index=False).encode("utf-8"), "predictions.csv",
                               "text/csv")

with tab3:
    st.markdown(f"""
- **Model:** LightGBM, cấu hình tốt nhất khi tinh chỉnh ở E4 (Random Search, seed 1); learning rate
  {vn(META['learning_rate'], 'g')}, {META['n_trees']} cây.
- **Đầu vào:** 21 cột gốc, không cần tiền xử lý; LightGBM tự xử lý giá trị thiếu ở `Arrival Delay`.
- **Đánh giá:** CV AUC {vn(META['cv_auc'], '.4f')} trên 5 fold `data/folds.csv` (trung bình fold 1–4, chưa dùng
  khi dò tham số: {vn(META['cv_auc_folds1_4'], '.4f')}).
- **Vì sao không triển khai bài nộp cuối:** bài nộp (private 0,96239) là stacking trên khoảng 250 OOF công khai cùng
  nhiều model khác; muốn dự đoán dòng mới phải chạy lại tất cả các model đó. Model này đơn giản hơn nhiều, AUC thấp
  hơn khoảng 0,0033, nhưng chạy được ngay trên máy cá nhân.
""")
    st.json(META["params"])
