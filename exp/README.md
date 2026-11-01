# Thí nghiệm

Mỗi thí nghiệm là một script/notebook, ghi kết quả ra `results/<tên>.csv` (qua `src/cv.py`) và hình ra `reports/figures/`.
Script trong `exp/` thêm `src/` vào đường dẫn import:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
```

Quy ước chung: cùng `data/folds.csv`, metric AUC, báo cáo trung bình ± độ lệch chuẩn 5 fold, chênh so với baseline và số fold tăng.
Mọi bước học từ dữ liệu (imputer, scaler, target encoding, chọn feature, early stopping) chỉ fit trên phần train của fold.

| Mã | Nội dung | Bài học | Trạng thái |
| --- | --- | --- | --- |
| E0 | Thiết lập đánh giá: K-fold, AUC, OOF sạch và OOF bẩn | Tuần 5 | chưa làm |
| E1 | EDA: đơn biến, hai biến, tương quan, thiếu/trùng/ngoại lệ (IQR), adversarial validation, gốc và tổng hợp, PCA/t-SNE | Tuần 3, 6 | chưa làm |
| E2 | Tiền xử lý (ablation): giá trị thiếu, ngoại lệ, scale, mã hoá, tạo feature, chọn feature | Tuần 4 | chưa làm |
| E3 | So sánh model: LR, KNN, NB, DT, RF, SVM, MLP, XGB, LGB, CatBoost, model mạng nơ-ron cho bảng; bias/variance, learning curve | Tuần 7, 8 | chưa làm |
| E4 | Tinh chỉnh tham số: Grid, Random, Bayesian (Optuna) cùng ngân sách | Tuần 9 | chưa làm |
| E5 | Ensemble: averaging, weighted, rank, blending, stacking (nested CV), bagging; stack với pool OOF công khai | Tuần 10 | chưa làm |
| E6 | Đánh giá cuối: ROC, confusion matrix, AUC theo nhóm, lỗi tự tin, CV và public/private LB | Tuần 5, 11 | chưa làm |
| E7 | Triển khai: demo Streamlit (+ API) | Tuần 11 | chưa làm |
