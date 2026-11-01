# Code gốc khi thi

Thư mục `code/` chứa **nguyên văn** các script đã tạo ra những con số "trong cuộc thi" của báo cáo (chép từ repo làm việc
của nhóm, 12–31/10/2026). Chúng được **lưu để tra cứu, không chạy lại** trong đồ án: đường dẫn bên trong theo cấu trúc
repo cũ (`notebooks/`, `oof/`, `data/public_*`), và phần lớn chạy trên Kaggle với các dataset OOF công khai hoặc GPU.
Kết quả từng script ghi trong `results/competition_log.csv` (cột `notes` nêu tên script và kernel).

Phần **tái lập thật** nằm ở:

| Tái lập | Ở đâu | Kết quả |
| --- | --- | --- |
| File nộp slot 1 (private 0,96239) từ các phần đã lưu | `reproduce_final.py` | trùng từng byte với file đã nộp (cùng md5) |
| Model `A_lr` (CV khi thi 0,959737) | E2, bậc L6 của LR (`exp/E2_preprocess.py`) | 0,959727 (chênh 1e-5) |
| Model `A_lgb` (CV khi thi 0,960893) | E2, bậc "4t, lr 0,02" (`exp/kaggle/E2_lgb.py`) | 0,960889 (chênh 4e-6) |

Chênh lệch khi tái lập đến từ số luồng (LightGBM không cho kết quả y hệt khi đổi số luồng) và phiên bản scikit-learn,
nhỏ hơn nhiễu giữa hai seed (Q-FT: 2,4e-5). *Ghi chú:* số "khi thi" là bản `_v2` (18/10), sau khi sửa một lỗi đảo cột
Flight Distance / Age ở bản đầu; `src/features.py` là bản đã sửa.

## Danh sách script

**Model tự train** (mỗi script lưu `oof_<tên>.npy`, `test_<tên>.npy` trên `folds.csv`):

| Script | Model | CV khi thi | Chạy ở |
| --- | --- | --- | --- |
| `A_lgb_fe.py` | LightGBM, bậc thang feature (nay là `src/features.py`) | `A_lgb` 0,960893 | máy cá nhân |
| `A_lr.py` | Logistic Regression trên one-hot thưa + TE trong fold + teacher gốc | `A_lr` 0,959737 | máy cá nhân |
| `A_knn.py` | kNN trên không gian đặc trưng tự chọn thang | `A_knn_v2` 0,950270 | máy cá nhân |
| `A_tabm.py` | TabM | `A_tabm_v1` 0,959541 | Kaggle GPU |
| `A_fm.py` | Factorization Machine | `A_fm_v1` 0,957154 | máy cá nhân |
| `A_ftt.py` | FT-Transformer với token tra cứu giá trị | `A_ftt_v1` 0,960050 | Kaggle GPU |
| `A_glr.py` | LR + token GPT-2 của chuỗi số | `A_glr_v1` 0,959867 | máy cá nhân |
| `A_llr_gen.py`, `A_llr_eval.py` | LLR: distilgpt2 sinh dòng theo nhãn, train trên dữ liệu gốc | `A_llr_v1` 0,951541 | Kaggle GPU |
| `A_gptc.py` | GPT-2 phân loại dòng dạng văn bản | `A_gptc_v1` 0,956221 | Kaggle GPU |
| `A_qft2.py` | **Q-FT**: transformer tự viết, tiền huấn luyện RTD rồi fine-tune | `A_qft2_v2` 0,961091 | Kaggle GPU (T4 ×2) |
| `A_kozy_clean.py` | chạy lại sạch một member công khai (notebook gốc đã bị tác giả xoá) | `A_kozy_v1` 0,958661 | máy cá nhân |
| `A_lgb_origrow.py` | LightGBM ghép thêm dòng dữ liệu gốc vào phần train của fold | 0,959729 (đối chứng 0,959630) | máy cá nhân |
| `A_orig_teacher.py` | teacher chỉ train trên dữ liệu gốc, các cách xử lý delay | – | máy cá nhân |
| `A_tfm_probe.py` | thử TabPFN với context lớn | `A_tabpfn_probe` 0,959990 (fold 0) | Kaggle GPU |

Model của các thành viên khác (XGBoost / CatBoost có teacher từ dữ liệu gốc, RealMLP, model chuyên khách đi việc riêng)
được nhận dưới dạng file OOF / test trên cùng `folds.csv`; code nằm trong git repo làm việc cũ của nhóm: notebook `cs116.ipynb` (4 model XGBoost / CatBoost dùng dữ liệu gốc, 17/10), `B_realmlp_pytabkit.ipynb` và `B_personal_specialist_gbdt.ipynb` (18/10). Đã kiểm tra: XGBoost / CatBoost và RealMLP sạch (early stopping trên 10% tách từ phần train của fold); Dual-Specialist LightGBM dừng sớm trên fold đang chấm nên không đưa vào stack ở E3.

**Stacking và chọn bài:**

| Script | Nội dung |
| --- | --- |
| `A_stack_v4.py` | stack LR theo họ + LGB trên 251 OOF công khai (`stack_fw_p247`, CV 0,962218) |
| `A_stack_v17.py`, `A_stack_v17_exp.py` | stack S14 + Q-FT (`stack_v17b_qft15`, CV 0,962224), kernel `cs116-stack-v17` |
| `A_team_tfm_stack.py`, `A_team_into_stack.py` | stack chỉ model của nhóm; thử đưa model nhóm vào stack công khai |
| `A_final_pair.py`, `A_slot1_v17b.py` | dựng file slot 1 (0,8 stack + 0,2 file 0,96206 của Kirill) |
| `A_slot2_mix.py`, `A_slot2_hedge.py`, `A_slot2_step.py`, `A_slot2_centroid.py` | các phiên bản slot 2 (dựa trên file công khai) |
| `A_shakeup.py`, `A_slot2_noise.py` | ước lượng nhiễu public / private khi chọn bài |

## Nguồn công khai đã dùng

Pool OOF công khai và công thức stack (ghi nguồn trong báo cáo): bộ nạp của Dariush Afshar, LR theo họ của ewtherka,
quy tắc chia họ của goodpjw2008, LightGBM + tỉ lệ 0,2 từ "Trust the CV" (Prabhav Semwal), dataset OOF của Chris Deotte,
goodpjw2008 và các tác giả khác liệt kê trong `dataset_sources` của kernel `cs116-stack-v17`; file 0,96206 của Kirill.
