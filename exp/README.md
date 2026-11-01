# Thí nghiệm

Phần thực nghiệm đi theo **ML pipeline** của Bài 03 (dữ liệu → kiểm tra dữ liệu → tiền xử lý → huấn luyện → đánh giá →
tinh chỉnh → kết hợp mô hình), kể cả vòng lặp "chưa đạt thì quay lại". Quá trình thi thật của nhóm cũng là các vòng lặp
như vậy:

| Vòng | Thời gian | Nội dung | Kết quả | Đạt top 5%? |
| --- | --- | --- | --- | --- |
| 1 | 12–14/10 | EDA, tạo feature, LightGBM đơn | CV 0,96096, public 0,96061 | chưa (ngưỡng ≈ 0,9619) |
| 2 | 15–19/10 | mô hình khác loại (LR, KNN, TabM, FM, FT-Transformer, TabPFN, mô hình ngôn ngữ) | stack chỉ model của nhóm: 0,96121 (10 model, 19/10); đầy đủ 25 model gồm Q-FT, B, quan: 0,96162 | chưa |
| 3 | 15–25/10 | stacking với bộ OOF công khai, Q-FT tự viết | CV 0,962224 | gần |
| 4 | 25–31/10 | chọn 2 bài nộp, đối chiếu public / private | private 0,96239, hạng 66 / ~3748 | đạt |

## Cấu trúc mỗi mục

Mỗi mục E0–E6 có ba khối:

1. **Trong cuộc thi**: nhóm đã làm gì, ngày nào, kết quả thật (nguồn: `results/competition_log.csv`, code gốc trong
   `exp/competition/`).
2. **Thí nghiệm hệ thống**: chạy trên cùng `data/folds.csv`, theo nội dung bài học, để giải thích hoặc kiểm chứng lựa
   chọn ở khối 1.
3. **Quyết định**: kết luận rút ra và chuyển sang mục sau.

Bảng kết quả có cột **thời điểm**: ngày trong cuộc thi (ví dụ `14/10`) hoặc `kiểm chứng` (chạy sau cuộc thi).

| Mã | Bước pipeline | Bài học | Trong cuộc thi | Thí nghiệm hệ thống | Trạng thái |
| --- | --- | --- | --- | --- | --- |
| E0 | Thiết lập đánh giá | 05 | `folds.csv` 12/10, luật OOF sạch 14/10, ngưỡng giữ thay đổi 18–20/10 | độ đo AUC; 3 quy trình đánh giá; đo rò rỉ (TE, early stopping) | xong |
| E1 | Kiểm tra dữ liệu (EDA) | 03, 06 | EDA 12–13/10, so sánh dữ liệu gốc 15/10 | thêm IQR / z-score, giá trị thiếu, PCA, t-SNE, KMeans | xong |
| E2 | Tiền xử lý, tạo và chọn feature | 04 | bộ feature `A_lr_v1` (0,9597), `A_lgb_v1` (0,9610); ghép dòng gốc 18/10 | "bậc thang" từ LR baseline 0,9217: mỗi bậc thêm một bước, tới bộ feature thật | chưa |
| E3 | Huấn luyện, so sánh mô hình | 07, 08 | LGB, LR, KNN, TabM, FM, FT-T, TabPFN, Q-FT, mô hình ngôn ngữ | DT, RF, Naive Bayes, SVM, MLP; bias / variance; learning curve | chưa |
| E4 | Tinh chỉnh siêu tham số | 09 | dò stacker (v5), lưới hệ số họ (v16), số epoch Q-FT | Grid / Random / Bayesian (Optuna) cùng ngân sách trên LGB | chưa |
| E5 | Kết hợp mô hình | 10 | stack chỉ model của nhóm (0,96162, 25 model), stack với pool công khai, FWLS, Q-FT qua cổng | averaging, weighted, rank, blending, stacking trên member của nhóm | chưa |
| E6 | Đánh giá cuối, chọn bài nộp | 05, 11 | 36 bài nộp, chọn 2 slot, public và private | ROC, AUC theo nhóm khách, lỗi tự tin; nhiễu public LB | chưa |
| E7 | Triển khai (tuỳ chọn) | 11 | – | demo Streamlit | chưa |

## Chạy

Mỗi thí nghiệm viết dạng script có ô `# %%`, rồi chuyển thành notebook đã chạy sẵn output:

```bash
python exp/tools/py2nb.py exp/E1_eda.py      # -> exp/E1_eda.ipynb
```

Script thêm `src/` vào đường dẫn import (`sys.path.insert(0, ...)`); bảng ghi ra `results/`, hình ra `reports/figures/`.
Mọi bước học từ dữ liệu (điền thiếu, chuẩn hoá, target encoding, chọn feature, early stopping) chỉ fit trên phần train
của fold, qua `fit_predict` của `src/cv.py`.
