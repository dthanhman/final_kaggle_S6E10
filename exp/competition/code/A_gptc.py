"""CS116 | distilgpt2 fine-tune làm classifier trên dòng dạng text của CHÍNH dữ liệu cuộc thi, chạy Kaggle GPU.

Khác A_llr_gen.py (LM chỉ học dữ liệu gốc 129k, LLR không dùng nhãn cuộc thi): ở đây model học trên phần train
của từng fold (folds.csv), nên học được cả dấu vết của bộ sinh S6E10 (cách viết số theo token GPT-2,
notebooks/A_llm_fingerprint.py). Ý từ writeup S6E9 (Will: +5e-5 nested; hạng 13: -1.1e-5).

Cách làm (mỗi fold k):
  - Mỗi dòng viết "cột is giá trị, ..., satisfaction is" (thứ tự cột ngẫu nhiên khi train), giá trị giữ đúng chuỗi CSV.
  - Điểm = h_cuối · (E[" True"] - E[" False"]): chỉ hai hàng của ma trận embedding, khởi tạo từ LM head của distilgpt2.
  - Loss BCE, 1 epoch cố định trên phần train của fold, KHÔNG dừng sớm theo fold đang chấm (OOF sạch);
    MONITOR dòng tách từ phần train chỉ để in AUC.
  - Chấm fold k và test bằng K thứ tự cột cố định (thứ tự CSV + ngẫu nhiên), lấy TB logit.
Output (/kaggle/working): fold_<k>.npz (chạy tiếp được nếu hết giờ: gắn output cũ làm input),
  khi đủ 5 fold: oof_A_gptc_v1.npy, test_A_gptc_v1.npy (logit), log.
Chạy thử trên máy: SMOKE=1 INPUT=data WORK=<thư mục> python notebooks/A_gptc.py  (GPT-2 tí hon ngẫu nhiên, vài trăm dòng)
"""
import os, glob, math, time, random
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config

SMOKE = os.environ.get("SMOKE") == "1"
INPUT = os.environ.get("INPUT", "/kaggle/input")
WORK = os.environ.get("WORK", "/kaggle/working")
NAME = os.environ.get("NAME", "A_gptc_v1")
FOLDS = [int(f) for f in os.environ.get("FOLDS", "0,1,2,3,4").split(",")]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(WORK, exist_ok=True)
EPOCHS, LR, BS, K, MAXLEN, MONITOR = (1, 1e-3, 16, 2, 256, 100) if SMOKE else (1, 5e-5, 64, 2, 256, 5000)
SCORE_BS = 32 if SMOKE else 256
TIME_LIMIT = 11.0 * 3600            # Kaggle cắt ở 12 giờ: không bắt đầu fold mới nếu không kịp
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)
torch.manual_seed(0); random.seed(0); np.random.seed(0)


def find(pred, pattern="*.csv"):
    for f in sorted(glob.glob(f"{INPUT}/**/{pattern}", recursive=True)):
        try:
            h = pd.read_csv(f, nrows=5, dtype=str)
        except Exception:
            continue
        if pred(f, h):
            return f
    raise FileNotFoundError(pattern)


TRAIN = find(lambda f, h: f.endswith("train.csv") and "id" in h and "satisfaction" in h)
TEST = find(lambda f, h: f.endswith("test.csv") and "id" in h and "satisfaction" not in h)
FOLDF = find(lambda f, h: f.endswith("folds.csv") and list(h.columns) == ["id", "fold"])
log("train", TRAIN, "| test", TEST, "| folds", FOLDF, "| device", DEV)
# đọc dạng chuỗi để giữ đúng cách viết trong CSV (vd Arrival Delay '18.0'); ô thiếu -> chuỗi rỗng
tr = pd.read_csv(TRAIN, dtype=str, keep_default_na=False)
te = pd.read_csv(TEST, dtype=str, keep_default_na=False)
fo = pd.read_csv(FOLDF)
assert (fo["id"].astype(str).values == tr["id"].values).all(), "folds.csv không khớp thứ tự id của train.csv"
fold = fo["fold"].to_numpy()
y = tr["satisfaction"].map({"True": 1, "False": 0, "1": 1, "0": 0}).to_numpy()
assert not np.isnan(y.astype(float)).any()
FEATS = [c for c in tr.columns if c not in ("id", "satisfaction")]
if SMOKE:
    keep = np.r_[np.where(fold == 0)[0][:150], np.where(fold != 0)[0][:450]]
    tr, y, fold, te = tr.iloc[keep].reset_index(drop=True), y[keep], fold[keep], te.head(200)
log(f"train {len(tr):,} | test {len(te):,} | {len(FEATS)} đặc trưng | tỉ lệ True {y.mean():.4f}")

if SMOKE and not os.environ.get("SMOKE_TOK"):
    import tiktoken
    _e = tiktoken.get_encoding("gpt2")

    class _Tok:  # chạy thử không có mạng: tokenizer GPT-2 qua tiktoken
        eos_token_id = 50256
        def __call__(self, texts):
            return {"input_ids": [_e.encode(t) for t in texts]}
    tok = _Tok()
else:
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("distilgpt2")
EOS = tok.eos_token_id
T_ID, F_ID = tok([" True"])["input_ids"][0], tok([" False"])["input_ids"][0]
assert len(T_ID) == 1 and len(F_ID) == 1, "' True' / ' False' phải là một token"
T_ID, F_ID = T_ID[0], F_ID[0]


def text(r, order):
    return ", ".join(f"{c} is {r[c]}" for c in order) + ", satisfaction is"


class Clf(torch.nn.Module):
    """distilgpt2 + đầu 2 lớp lấy từ hàng ' True' / ' False' của embedding (LM head buộc với wte)."""
    def __init__(self):
        super().__init__()
        if SMOKE:
            lm = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=50257, n_positions=512))
        else:
            lm = GPT2LMHeadModel.from_pretrained("distilgpt2")
        self.body: GPT2Model = lm.transformer
        w = lm.lm_head.weight.detach()
        self.w = torch.nn.Parameter((w[T_ID] - w[F_ID]).clone())
        self.b = torch.nn.Parameter(torch.zeros(()))

    def forward(self, inp, att):
        h = self.body(input_ids=inp, attention_mask=att).last_hidden_state
        last = att.sum(1) - 1
        return h[torch.arange(len(h), device=h.device), last].float() @ self.w.float() + self.b


def batch(texts):
    ids = [[EOS] + x[-(MAXLEN - 1):] for x in tok(texts)["input_ids"]]
    L = max(map(len, ids))
    inp = torch.full((len(ids), L), EOS); att = torch.zeros((len(ids), L), dtype=torch.long)
    for i, x in enumerate(ids):
        inp[i, :len(x)] = torch.tensor(x); att[i, :len(x)] = 1
    return inp.to(DEV), att.to(DEV)


@torch.no_grad()
def score(model, recs, order):
    model.eval(); out = np.zeros(len(recs))
    for i in range(0, len(recs), SCORE_BS):
        inp, att = batch([text(r, order) for r in recs[i:i + SCORE_BS]])
        with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
            out[i:i + len(inp)] = model(inp, att).float().cpu().numpy()
    return out


ORDERS = [FEATS[:]] + [random.Random(1000 + k).sample(FEATS, len(FEATS)) for k in range(1, K)]
REC_ALL, REC_TE = tr[FEATS].to_dict("records"), te[FEATS].to_dict("records")


def run_fold(k):
    t0 = time.time()
    tri = np.where(fold != k)[0]; vai = np.where(fold == k)[0]
    rng = np.random.default_rng(k)
    mon = rng.choice(tri, min(MONITOR, len(tri) // 10), replace=False)
    fit = np.setdiff1d(tri, mon)
    model = Clf().to(DEV)
    total = EPOCHS * math.ceil(len(fit) / BS)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, int(0.03 * total))) * max(0.0, 1 - s / total))
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    pyr = random.Random(k); step = 0
    model.train()
    for ep in range(EPOCHS):
        perm = rng.permutation(fit)
        for i in range(0, len(perm), BS):
            b = perm[i:i + BS]
            texts = [text(REC_ALL[j], pyr.sample(FEATS, len(FEATS))) for j in b]
            inp, att = batch(texts)
            yt = torch.tensor(y[b], dtype=torch.float32, device=DEV)
            with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
                z = model(inp, att)
            loss = lossf(z.float(), yt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            if step % 1000 == 0 or step == total:
                log(f"fold {k} step {step}/{total} loss {loss.item():.4f} | {step * BS / (time.time() - t0):.0f} dòng/s")
        mrec = [REC_ALL[j] for j in mon]
        log(f"fold {k} epoch {ep}: AUC monitor (tách từ phần train) {roc_auc_score(y[mon], score(model, mrec, ORDERS[0])):.6f}")
        model.train()
    vrec = [REC_ALL[j] for j in vai]
    va = np.column_stack([score(model, vrec, o) for o in ORDERS])
    ts = np.column_stack([score(model, REC_TE, o) for o in ORDERS])
    np.savez(f"{WORK}/fold_{k}.npz", idx=vai, va=va, test=ts)
    log(f"fold {k}: AUC {roc_auc_score(y[vai], va.mean(1)):.6f} | theo thứ tự "
        + " ".join(f"{roc_auc_score(y[vai], va[:, j]):.6f}" for j in range(K))
        + f" | tương quan 2 thứ tự {np.corrcoef(va.T)[0, 1]:.4f} | {(time.time() - t0) / 60:.1f} phút")
    del model, opt; torch.cuda.empty_cache() if DEV == "cuda" else None
    return time.time() - t0


done = {k: f for k in range(5) for f in sorted(glob.glob(f"{INPUT}/**/fold_{k}.npz", recursive=True))[:1]}
log("fold đã có từ lần chạy trước:", sorted(done))
spent = []
for k in FOLDS:
    if k in done:
        continue
    if spent and time.time() - T0 + 1.15 * max(spent) > TIME_LIMIT:
        log(f"dừng trước fold {k}: không kịp trong giới hạn giờ"); break
    spent.append(run_fold(k))

parts = {k: f"{WORK}/fold_{k}.npz" for k in range(5) if os.path.exists(f"{WORK}/fold_{k}.npz")}
parts = {**done, **parts}
if sorted(parts) == list(range(5)) or (SMOKE and 0 in parts):
    oof, test = np.full(len(tr), np.nan), []
    for k, f in sorted(parts.items()):
        d = np.load(f); oof[d["idx"]] = d["va"].mean(1); test.append(d["test"].mean(1))
    if SMOKE:
        oof = np.nan_to_num(oof)
    else:
        assert not np.isnan(oof).any()
    np.save(f"{WORK}/oof_{NAME}.npy", oof); np.save(f"{WORK}/test_{NAME}.npy", np.mean(test, 0))
    log(f"Đã lưu oof_{NAME}.npy, test_{NAME}.npy | CV AUC = {roc_auc_score(y, oof):.6f} (logit, TB {K} thứ tự cột)")
else:
    log(f"mới có fold {sorted(parts)}: gắn output kernel này làm input rồi chạy lại để làm nốt")
log("xong")
