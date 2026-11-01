# CS116 – Kaggle Playground Series S6E10: Predicting Airline Satisfaction

Đồ án môn CS116 (Lập trình Python cho máy học), nhóm **Everhope**.

- Cuộc thi: [playground-series-s6e10](https://www.kaggle.com/competitions/playground-series-s6e10): phân loại nhị phân, dự đoán xác suất hành khách hài lòng.
- Metric: **ROC AUC**.
- Kết quả: **hạng 66 / ~3748 đội trên private LB** (AUC 0,96239), top ~1,8%; public LB 0,96190.

## Cấu trúc

```
├── data/              # không commit (trừ folds.csv), xem data/README.md
├── src/               # code dùng chung
│   ├── config.py      # đường dẫn, tên cột, N_FOLDS, SEED
│   ├── data.py        # load_data(): train/test + cột fold
│   ├── cv.py          # run_cv(), compare(), log_result(): CV 5 fold + bảng kết quả
│   ├── utils.py       # save_preds() / load_preds(): oof_<tên>.npy, test_<tên>.npy
│   ├── blend.py       # blend / stack LR trên OOF
│   ├── submit.py      # tạo và kiểm tra file nộp
│   └── make_folds.py  # đã chạy một lần để tạo data/folds.csv
├── exp/               # các thí nghiệm E0–E7, xem exp/README.md
├── results/           # bảng kết quả (csv) do script sinh ra
└── reports/figures/   # hình cho báo cáo
```

## Cài đặt

```bash
conda create -n cs116 python=3.11
conda activate cs116
pip install -r requirements.txt
```

Tải dữ liệu theo `data/README.md`. Các thí nghiệm nặng (GPU, model lớn) chạy trên Kaggle Notebook; đặt `DATA_DIR`, `OOF_DIR` trỏ tới dữ liệu.

## Nguyên tắc thực nghiệm

1. Một bộ fold duy nhất `data/folds.csv` (StratifiedKFold 5 fold, seed 42) cho mọi thí nghiệm.
2. OOF sạch: mọi bước học từ dữ liệu fit trong phần train của fold; early stopping không dùng fold đang chấm.
3. Một thay đổi chỉ được giữ khi tăng CV ổn định (phần lớn fold, nhiều seed); tin CV hơn public LB.

## Nguồn tham khảo

Lớp stack cuối dùng bộ OOF công khai của cộng đồng và công thức stack công khai; ghi nguồn chi tiết trong báo cáo và trong `exp/E5_*`.
