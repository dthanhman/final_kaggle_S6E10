# Thí nghiệm

Phần thực nghiệm đi theo ML pipeline của Bài 03: dữ liệu → kiểm tra dữ liệu → tiền xử lý → huấn luyện → đánh giá →
tinh chỉnh → kết hợp model, có vòng lặp quay lại khi kết quả chưa đạt. Quá trình thi của nhóm cũng gồm bốn vòng như
vậy:

| Vòng | Thời gian | Nội dung | Kết quả | Đạt top 5%? |
| --- | --- | --- | --- | --- |
| 1 | 12–14/10 | EDA, tạo feature, LightGBM đơn | CV 0,9609, public 0,96061 | chưa (ngưỡng khoảng 0,9619) |
| 2 | 15–19/10 | model khác loại (LR, kNN, TabM, FM, FT-Transformer, TabPFN, GPT-2) | stacking chỉ gồm model nhóm tự train: 0,96121 (10 model, 19/10); về sau đủ 25 OOF sạch, có Q-FT: 0,96162 | chưa |
| 3 | 15–25/10 | stacking với pool OOF công khai, Q-FT | CV 0,962224 | gần |
| 4 | 25–31/10 | chọn 2 bài nộp, đối chiếu public và private | private 0,96239, hạng 66 / khoảng 3.748 đội | đạt |

## Cấu trúc mỗi mục

Mỗi mục E0–E6 có ba phần:

1. **Trong cuộc thi**: nhóm đã làm gì, vào ngày nào, kết quả ra sao (nguồn: `results/competition_log.csv`, code gốc
   trong `exp/competition/`).
2. **Thí nghiệm kiểm chứng**: chạy lại trên cùng `data/folds.csv` theo nội dung bài học tương ứng, để giải thích hoặc
   kiểm tra lựa chọn ở phần 1.
3. **Quyết định**: kết luận rút ra và cách áp dụng ở mục sau.

Các bảng kết quả có cột thời điểm: ngày trong cuộc thi (ví dụ `14/10`) hoặc `kiểm chứng` nếu chạy sau cuộc thi.

Từ E2 trở đi, mỗi mục có bảng "Đã thử nhưng không giữ" ghi các ý tưởng không thành công: thời điểm, mức thay đổi so
với nền, số fold tăng và lý do không giữ (nhỏ hơn nhiễu, rò rỉ dữ liệu, pool đã có thông tin đó, overfit public LB…).
E6 tổng hợp lại những kết quả này.

| Mã | Bước pipeline | Bài | Trong cuộc thi | Thí nghiệm kiểm chứng | Trạng thái |
| --- | --- | --- | --- | --- | --- |
| E0 | Thiết lập đánh giá | 05 | `folds.csv` 12/10, quy tắc OOF sạch 14/10, ngưỡng giữ thay đổi 18–20/10 | độ đo AUC; 3 quy trình đánh giá; đo rò rỉ (target encoding, early stopping) | xong |
| E1 | Kiểm tra dữ liệu (EDA) | 03, 06 | EDA 12–13/10, so sánh dữ liệu gốc 15/10 | IQR / z-score, giá trị thiếu, PCA, t-SNE, KMeans | xong |
| E2 | Tiền xử lý, tạo và chọn feature | 04 | bộ feature của `A_lr` (0,95974) và `A_lgb` (0,96089); ghép dòng gốc 18/10 | "bậc thang" từ LR baseline 0,9217, mỗi bậc thêm một bước | xong |
| E3 | Huấn luyện, so sánh model | 07, 08 | LightGBM, LR, kNN, TabM, FM, FT-Transformer, TabPFN, Q-FT, GPT-2 | Decision Tree, Random Forest, Naive Bayes, SVM, MLP; bias / variance; learning curve | xong |
| E4 | Tinh chỉnh siêu tham số | 09 | dò stacker (v5), lưới hệ số theo họ (v16), số epoch Q-FT | Grid / Random / Optuna cùng số lần thử trên LightGBM | xong |
| E5 | Kết hợp model | 10 | stacking chỉ model của nhóm (0,96162), stacking với pool công khai, FWLS, Q-FT | averaging, weighted, rank, blending, stacking trên OOF của nhóm | xong |
| E6 | Đánh giá cuối, chọn bài nộp | 05, 11 | 36 bài nộp, chọn 2 bài cuối, public và private | ROC, AUC theo nhóm khách, lỗi tự tin; nhiễu public LB | xong |
| E7 | Triển khai | 11 | – | app Streamlit dự đoán một khách hoặc cả file CSV, giải thích bằng SHAP (`deploy/`) | xong |

## Thuật ngữ

| Thuật ngữ | Nghĩa trong đồ án |
| --- | --- |
| fold, `folds.csv` | 5 phần chia cố định của tập train (StratifiedKFold, seed 42); mọi model dùng chung |
| OOF (out-of-fold) | dự đoán cho mỗi dòng train từ model không được train trên fold chứa dòng đó |
| OOF sạch | OOF mà nhãn của fold đang chấm không được dùng ở bất kỳ bước nào (target encoding, early stopping, chọn tham số) |
| stacking, stacker | kết hợp nhiều model bằng một meta model (stacker) học trên OOF của chúng; thường là LR trên logit |
| member | một model đầu vào của stacking |
| meta fold, meta seed | cách chia fold dùng để đánh giá stacker (nested CV); đổi meta seed để kiểm tra độ ổn định |
| pool công khai | bộ OOF của các thí sinh khác chia sẻ trên Kaggle, cùng cách chia fold seed 42 |
| cổng | ngưỡng giữ một thay đổi khi stacking: tăng ít nhất 5×10⁻⁶ ở cả 3 meta seed và ít nhất 10/15 fold (E0) |
| dữ liệu gốc | bộ dữ liệu thật 129.880 dòng mà dữ liệu cuộc thi được sinh ra từ đó |
| teacher | model chỉ train trên dữ liệu gốc; dự đoán của nó được dùng làm feature cho model train trên dữ liệu cuộc thi |
| target encoding (TE) | thay giá trị category bằng tỉ lệ hài lòng trung bình của giá trị đó, tính trong fold |
| Q-FT | transformer cho dữ liệu bảng do nhóm tự viết: tiền huấn luyện bằng RTD rồi fine-tune để phân lớp |
| RTD (replaced token detection) | nhiệm vụ tiền huấn luyện tự giám sát: đoán ô nào trong dòng đã bị thay bằng giá trị khác (theo ELECTRA) |
| FWLS | feature-weighted linear stacking: hệ số của stacker thay đổi theo một số feature (ở đây là nhóm khách) |
| foundation model cho dữ liệu bảng | model tiền huấn luyện sẵn, dự đoán bằng cách đọc dữ liệu train làm ngữ cảnh (TabPFN, TabICL…) |
| public / private LB | bảng xếp hạng chấm trên 20% / 80% tập test; private LB quyết định thứ hạng cuối |
| Δ | chênh lệch AUC so với cấu hình nền; chênh lệch nhỏ ghi dạng 3×10⁻⁶ |

## Chạy

Mỗi thí nghiệm là một script có các ô `# %%`, được chuyển thành notebook đã chạy sẵn output:

```bash
python exp/tools/py2nb.py exp/E1_eda.py      # -> exp/E1_eda.ipynb
```

Script thêm `src/` vào đường dẫn import (`sys.path.insert(0, ...)`), ghi bảng ra `results/` và hình ra
`reports/figures/`. Mọi bước học từ dữ liệu (điền thiếu, chuẩn hoá, target encoding, chọn feature, early stopping)
chỉ fit trên phần train của fold, thông qua `fit_predict` của `src/cv.py`.

## Tài liệu tham khảo

- Bergstra, J., Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. *JMLR* 13.
- Bergstra, J., Bardenet, R., Bengio, Y., Kégl, B. (2011). Algorithms for Hyper-Parameter Optimization (TPE). *NeurIPS*.
- Akiba, T. và cộng sự (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *KDD*.
- Breiman, L. (2001). Random Forests. *Machine Learning* 45.
- Geurts, P., Ernst, D., Wehenkel, L. (2006). Extremely Randomized Trees. *Machine Learning* 63.
- Chen, T., Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
- Ke, G. và cộng sự (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*.
- Prokhorenkova, L. và cộng sự (2018). CatBoost: Unbiased Boosting with Categorical Features. *NeurIPS*.
- Wolpert, D. (1992). Stacked Generalization. *Neural Networks* 5.
- Caruana, R., Niculescu-Mizil, A., Crew, G., Ksikes, A. (2004). Ensemble Selection from Libraries of Models. *ICML*.
- Sill, J., Takács, G., Mackey, L., Lin, D. (2009). Feature-Weighted Linear Stacking. arXiv:0911.0460.
- Williams, C., Seeger, M. (2001). Using the Nyström Method to Speed Up Kernel Machines. *NeurIPS*.
- Rendle, S. (2010). Factorization Machines. *ICDM*.
- Gorishniy, Y., Rubachev, I., Khrulkov, V., Babenko, A. (2021). Revisiting Deep Learning Models for Tabular Data (FT-Transformer). *NeurIPS*.
- Gorishniy, Y., Kotelnikov, A., Babenko, A. (2025). TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling. *ICLR*.
- Holzmüller, D., Grinsztajn, L., Steinwart, I. (2024). Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data (RealMLP). *NeurIPS*.
- Hollmann, N. và cộng sự (2025). Accurate Predictions on Small Data with a Tabular Foundation Model (TabPFN). *Nature* 637.
- Clark, K., Luong, M.-T., Le, Q., Manning, C. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators. *ICLR*.
- Radford, A. và cộng sự (2019). Language Models are Unsupervised Multitask Learners (GPT-2). OpenAI.

Nguồn OOF công khai và công thức stacking của cộng đồng Kaggle được ghi trong `exp/competition/README.md`.
