# Dữ liệu

Không commit dữ liệu lên repo. Tải về thư mục này:

```bash
kaggle competitions download -c playground-series-s6e10 -p data
unzip data/playground-series-s6e10.zip -d data
```

Sau khi tải, thư mục có:

| File | Nội dung |
| --- | --- |
| `train.csv` | 699.635 dòng, 21 feature + `satisfaction` (bool, True 44,36%) |
| `test.csv` | 299.844 dòng |
| `sample_submission.csv` | định dạng nộp `id,satisfaction` (xác suất) |
| `folds.csv` | **có sẵn trong repo**: 5 fold StratifiedKFold, seed 42 (`src/make_folds.py`) |

Dữ liệu gốc (dùng ở thí nghiệm so sánh gốc và tổng hợp): [binary-aviation-satisfaction-129k](https://www.kaggle.com/datasets/arseniyshutko/binary-aviation-satisfaction-129k), đặt ở `data/original/data.csv`:

```bash
kaggle datasets download -d arseniyshutko/binary-aviation-satisfaction-129k -p data/original --unzip
```
