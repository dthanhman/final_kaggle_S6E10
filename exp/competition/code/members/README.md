# Code model của các thành viên khác (lưu tra cứu)

Chép nguyên văn, kể cả output đã chạy, từ git repo làm việc cũ của nhóm. Không chạy lại trong repo này. Các model này
được dùng qua file OOF / test trên cùng `data/folds.csv` (bảng ở E3, mục 1).

| File | Nguồn (repo cũ) | Model / file OOF | CV khi thi | OOF sạch? |
| --- | --- | --- | --- | --- |
| `power_v4_xgb_catboost.ipynb` | `cs116.ipynb` (tên gốc), commit `f5fc727` (17/10, "Power v4") | 4 model: XGBoost loại dòng gốc trùng + teacher XGB (`quan_xgb_safe_teacher_v4`), XGBoost + 2 teacher XGB/CatBoost chỉ train trên dữ liệu gốc (`quan_xgb_dual_teacher_v4`), CatBoost GPU + 2 teacher (`quan_cat_dual_teacher_v4`), XGBoost ghép dòng gốc trọng số 0,35 (`quan_xgb_original_aug_v4`) | 0,96086 / 0,96092 / 0,96094 / 0,95853 | có: early stopping trên 10% tách từ phần train của fold; target encoding (5 fold bên trong) chỉ fit trên phần train; fit lại trên cả phần train với số cây đã chọn |
| `B_realmlp_pytabkit.ipynb` | `notebooks/B_realmlp_pytabkit.ipynb`, commit `bc85e46` (18/10, nhánh `feat/lukimluan-models`) | RealMLP-TD (pytabkit), 16 epoch, batch 2048, 21 cột + 6 feature tổng hợp (`B_realmlp_v1`) | 0,95770 | có: chọn epoch trên 10% tách từ phần train của fold |
| `B_personal_specialist_gbdt.ipynb` | `notebooks/B_personal_specialist_gbdt.ipynb`, commit `bc85e46` | Dual-Specialist LightGBM: 2 model riêng cho Personal / Business Travel (`B_personal_spec_v1`) | 0,95864 | **không**: `eval_set` của early stopping là chính các dòng validation của fold (`va_pers`, `va_biz`) → không đưa vào stack ở E3 |

`A_qft2_v2avg` (một member của stack) không có notebook riêng: là trung bình logit của Q-FT seed 42 và seed 7, tính lại và kiểm chứng ở `exp/scripts/E3_models.py` mục 1.

Đường dẫn trong output (máy cá nhân, `/kaggle/input/...`) giữ nguyên như khi chạy.
