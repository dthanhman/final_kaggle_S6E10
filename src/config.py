"""Cấu hình chung. Không đổi N_FOLDS / SEED: mọi kết quả đều dựa trên data/folds.csv."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Trên Kaggle/Colab: đặt biến môi trường DATA_DIR / OOF_DIR trỏ tới chỗ chứa data.
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
OOF_DIR = Path(os.environ.get("OOF_DIR", ROOT / "oof"))
SUB_DIR = ROOT / "submissions"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "reports" / "figures"

N_FOLDS = 5
SEED = 42

ID = "id"
TARGET = "satisfaction"

CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
SCORE_COLS = [  # điểm dịch vụ 0-5 (0 thường = không áp dụng)
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
]
NUM_COLS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
FEATURES = CAT_COLS + SCORE_COLS + NUM_COLS
