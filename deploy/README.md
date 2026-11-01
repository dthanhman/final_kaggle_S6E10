# E7. Triển khai: demo dự đoán mức hài lòng

App Streamlit dự đoán xác suất một hành khách hài lòng, từ 21 thông tin của chuyến bay (Bài 11).

```bash
python deploy/train_model.py      # train lại model (khoảng 10 giây), ghi model_lgb.txt và meta.json
streamlit run deploy/app.py       # mở app ở http://localhost:8501
```

Repo đã có sẵn `model_lgb.txt` và `meta.json`, nên có thể chạy app ngay mà không cần train lại. Cần cài thêm
`streamlit` (có trong `requirements.txt`).

## App gồm

| Tab | Chức năng |
| --- | --- |
| Một hành khách | nhập thông tin chuyến bay và 13 điểm dịch vụ, xem xác suất hài lòng và 8 feature ảnh hưởng nhiều nhất tới dự đoán đó (đóng góp SHAP của LightGBM, `pred_contrib`) |
| Dự đoán từ file CSV | tải lên file có 21 cột như `test.csv`, xem trước và tải về file kết quả `id,satisfaction` |
| Về model | cấu hình, số liệu đánh giá và giới hạn |

## Chọn model

Model triển khai là **LightGBM trên 21 cột gốc**, cấu hình tốt nhất khi tinh chỉnh ở E4 (Random Search, seed 1:
`num_leaves` 125, `min_child_samples` 15, `colsample_bytree` 0,69, `subsample` 0,99, `reg_lambda` 20,2; learning rate
0,1), train trên toàn bộ 699.635 dòng với 194 cây (số cây trung bình khi early stopping ở E4, cộng 10% vì có thêm
dữ liệu).

| | Model triển khai | Bài nộp cuối (slot 1) |
| --- | --- | --- |
| Thành phần | 1 LightGBM, 194 cây | stacking trên khoảng 250 OOF công khai và model của nhóm, trộn 20% một file công khai |
| AUC | CV 0,95893 (E4) | CV 0,96222, private 0,96239 |
| Dự đoán dòng mới | được: 1,5 ms/dòng; cả 299.844 dòng test trong 0,7 giây | không: phải chạy lại mọi model nguồn, nhiều model chỉ có OOF |
| Kích thước | 2,6 MB | – |

Chênh lệch AUC khoảng 0,0033 là cái giá để có một model chạy được độc lập. Bộ feature của E2 (CV 0,96089) cũng
không được dùng vì cần dữ liệu gốc và target encoding lúc dự đoán.

## Kiểm tra

- Model nạp lại từ file cho AUC 0,964 trên 20 nghìn dòng train (dữ liệu đã học, nên chỉ để kiểm tra nạp đúng; số đánh
  giá là CV ở E4). Xác suất trung bình trên tập test 0,444, khớp tỉ lệ hài lòng của train (44,4%).
- App được kiểm tra bằng `streamlit.testing` (cả 3 tab không lỗi). Với cùng các giá trị mặc định (điểm dịch vụ đều
  bằng 3), khách đi công tác cho xác suất 7,1%, khách đi việc riêng 2,6%; với khách đi công tác, đổi điểm làm thủ tục
  online từ 1 lên 5 làm xác suất tăng từ 27% lên 74%.

## Giới hạn

- Xác suất chưa được hiệu chỉnh: model tối ưu thứ hạng (AUC), nên con số 70% không nhất thiết nghĩa là 70% khách như
  vậy hài lòng.
- Model học từ dữ liệu tổng hợp của cuộc thi; với dữ liệu thật (ví dụ tỉ lệ trễ chuyến khác, xem E1 mục 8) cần đánh giá
  lại.
- Phiên bản: LightGBM 4.6.0, Streamlit 1.66. Nếu môi trường dùng NumPy 1.x thì cần `pyarrow` < 20.
