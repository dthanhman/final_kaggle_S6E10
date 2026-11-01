"""CS116 | LLR surrogate của bộ sinh (ý từ writeup S6E9 hạng 2/3), chạy Kaggle GPU.

Bằng chứng S6E10 sinh bằng LM tokenizer GPT-2: notebooks/A_llm_fingerprint.py (Flight Distance 3 chữ số tách 1 token
xuất hiện ít hơn ~16 lần so với dữ liệu gốc; số token giải thích R² 0.69, đối chứng 0.002).

Cách làm:
  1. Fine-tune distilgpt2 CHỈ trên dữ liệu gốc (129,880 dòng), mỗi dòng viết "cột is giá trị, ..." theo thứ tự cột
     ngẫu nhiên (kiểu GReaT); nhãn đứng đầu với xác suất P_LABEL_FIRST, còn lại ở vị trí ngẫu nhiên.
  2. Với mỗi dòng train/test của cuộc thi và K thứ tự cột cố định:
       LLR = log p(đặc trưng | "satisfaction is True") - log p(đặc trưng | "satisfaction is False").
     Không dùng nhãn của cuộc thi -> feature "label-free", dùng được trong mọi fold mà OOF vẫn sạch.
Output (/kaggle/working): llr_{train,test}.npy (TB K thứ tự), llr_orders_{train,test}.npy (K cột),
  lp_{true,false}_{train,test}.npy, model/ (để tính lại), log.
Chạy thử trên máy: SMOKE=1 python notebooks/A_llr_gen.py  (GPT-2 tí hon khởi tạo ngẫu nhiên, vài trăm dòng)
"""
import os, glob, math, time, random
import numpy as np, pandas as pd, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config

SMOKE = os.environ.get("SMOKE") == "1"
INPUT = os.environ.get("INPUT", "/kaggle/input")
WORK = os.environ.get("WORK", "/kaggle/working")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(WORK, exist_ok=True)
EPOCHS, LR, BS, P_LABEL_FIRST, K, MAXLEN = (1, 1e-3, 8, 0.5, 2, 256) if SMOKE else (4, 1e-4, 32, 0.5, 2, 256)
SCORE_BS = 8 if SMOKE else 64   # 256 OOM trên T4 (logits 256 x L x 50257)
# model đã fine-tune (dataset thanhmn123/cs116-llr-distilgpt2 = output model/ của lần chạy 17/10) -> bỏ qua bước 1
PRETRAINED = next(iter(sorted(glob.glob(f"{INPUT}/**/model.safetensors", recursive=True))), None)
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)
torch.manual_seed(0); random.seed(0); np.random.seed(0)


def find_csv(pred):
    for f in sorted(glob.glob(f"{INPUT}/**/*.csv", recursive=True)):
        try:
            h = pd.read_csv(f, nrows=5, dtype=str)
        except Exception:
            continue
        if pred(f, h):
            return f
    raise FileNotFoundError("không tìm thấy csv")


TRAIN = find_csv(lambda f, h: f.endswith("train.csv") and "id" in h and "satisfaction" in h)
TEST = find_csv(lambda f, h: f.endswith("test.csv") and "id" in h and "satisfaction" not in h)
ORIG = find_csv(lambda f, h: "id" not in h and "satisfaction" in h and "Flight Distance" in h)
log("train", TRAIN, "| test", TEST, "| gốc", ORIG, "| device", DEV)
# đọc dạng chuỗi để giữ đúng cách viết trong CSV (vd Arrival Delay '18.0'); ô thiếu -> chuỗi rỗng
og = pd.read_csv(ORIG, dtype=str, keep_default_na=False)
tr = pd.read_csv(TRAIN, dtype=str, keep_default_na=False)
te = pd.read_csv(TEST, dtype=str, keep_default_na=False)
FEATS = [c for c in og.columns if c != "satisfaction"]
assert set(FEATS) <= set(tr.columns) and set(FEATS) <= set(te.columns), "tên cột gốc và cuộc thi khác nhau"
if SMOKE:
    og, tr, te = og.sample(400, random_state=0), tr.head(300), te.head(200)
log(f"gốc {len(og):,} dòng, {len(FEATS)} đặc trưng | nhãn gốc: {og.satisfaction.value_counts().to_dict()}")
LAB = {"1": "True", "0": "False", "True": "True", "False": "False", "satisfied": "True",
       "neutral or dissatisfied": "False"}
og["satisfaction"] = og["satisfaction"].map(LAB)
assert og["satisfaction"].notna().all()

tok = GPT2TokenizerFast.from_pretrained("distilgpt2") if not SMOKE else None
if SMOKE:
    from transformers import GPT2TokenizerFast as T
    tok = T.from_pretrained("gpt2") if os.environ.get("SMOKE_TOK") else None
if tok is None:
    import tiktoken
    _e = tiktoken.get_encoding("gpt2")

    class _Tok:  # chỉ dùng khi chạy thử không có mạng: tokenizer GPT-2 qua tiktoken
        eos_token_id = 50256
        def __call__(self, texts):
            return {"input_ids": [_e.encode(t) for t in texts]}
    tok = _Tok()
EOS = tok.eos_token_id


def row_text(r, order):
    return ", ".join(f"{c} is {r[c]}" for c in order)


def train_text(r):
    order = FEATS[:]; random.shuffle(order)
    lab = f"satisfaction is {r['satisfaction']}"
    parts = [f"{c} is {r[c]}" for c in order]
    pos = 0 if random.random() < P_LABEL_FIRST else random.randint(0, len(parts))
    parts.insert(pos, lab)
    return ", ".join(parts)


# ---------------- 1. fine-tune trên dữ liệu gốc ----------------
if PRETRAINED:
    model = GPT2LMHeadModel.from_pretrained(os.path.dirname(PRETRAINED))
    log("dùng model đã fine-tune:", os.path.dirname(PRETRAINED))
elif SMOKE:
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=50257, n_positions=512))
else:
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
model.to(DEV)
idx = np.random.default_rng(0).permutation(len(og)); nval = max(50, int(0.03 * len(og)))
og_tr, og_va = og.iloc[idx[nval:]], og.iloc[idx[:nval]]
rows_tr = og_tr.to_dict("records")
rows_va = og_va.to_dict("records")


def collate(texts):
    ids = [[EOS] + x[:MAXLEN - 2] + [EOS] for x in tok(texts)["input_ids"]]
    L = max(map(len, ids))
    inp = torch.full((len(ids), L), EOS); att = torch.zeros((len(ids), L), dtype=torch.long)
    for i, x in enumerate(ids):
        inp[i, :len(x)] = torch.tensor(x); att[i, :len(x)] = 1
    lab = inp.clone(); lab[att == 0] = -100
    return inp.to(DEV), att.to(DEV), lab.to(DEV)


steps_per_epoch = math.ceil(len(rows_tr) / BS)
total = EPOCHS * steps_per_epoch
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, int(0.03 * total))) * max(0.0, 1 - s / total))
scaler = torch.amp.GradScaler(enabled=DEV == "cuda")


@torch.no_grad()
def val_loss():
    model.eval(); tot, n = 0.0, 0
    rnd = random.getstate(); random.seed(123)
    for i in range(0, len(rows_va), BS):
        inp, att, lab = collate([train_text(r) for r in rows_va[i:i + BS]])
        with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
            out = model(input_ids=inp, attention_mask=att, labels=lab)
        tot += out.loss.item() * len(inp); n += len(inp)
    random.setstate(rnd); model.train()
    return tot / n


step = 0
model.train()
for ep in range(0 if PRETRAINED else EPOCHS):
    random.shuffle(rows_tr)
    for i in range(0, len(rows_tr), BS):
        inp, att, lab = collate([train_text(r) for r in rows_tr[i:i + BS]])
        with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
            loss = model(input_ids=inp, attention_mask=att, labels=lab).loss
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step(); step += 1
        if step % 1000 == 0:
            log(f"epoch {ep} step {step}/{total} loss {loss.item():.4f}")
    log(f"epoch {ep} xong | val loss {val_loss():.4f}")
if not SMOKE and not PRETRAINED:
    model.save_pretrained(f"{WORK}/model"); tok.save_pretrained(f"{WORK}/model")
model.eval()


# ---------------- 2. LLR cho dòng cuộc thi ----------------
# lúc train ", Age is .." được tách thành ',' + ' Age' ... -> prefix kết thúc ở ',' và thân bắt đầu bằng dấu cách
PRE = {lab: [EOS] + tok([f"satisfaction is {lab},"])["input_ids"][0] for lab in ("True", "False")}


@torch.no_grad()
def feat_logprob(texts, prefix):
    """Tổng log p(token đặc trưng | prefix) cho từng chuỗi."""
    body = tok([" " + t for t in texts])["input_ids"]
    seqs = [prefix + b + [EOS] for b in body]
    L = max(map(len, seqs)); P = len(prefix)
    inp = torch.full((len(seqs), L), EOS); att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        inp[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
    inp, att = inp.to(DEV), att.to(DEV)
    with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
        logits = model(input_ids=inp, attention_mask=att).logits[:, :-1]
    # log p = logit đích - logsumexp; không tạo bản log_softmax cỡ vocab (OOM ở bản trước)
    lp = logits.gather(-1, inp[:, 1:, None])[..., 0].float() - logits.float().logsumexp(-1)
    del logits
    mask = att[:, 1:].clone(); mask[:, :P - 1] = 0                    # chỉ tính token sau prefix
    return (lp * mask).sum(1).cpu().numpy()


ORDERS = [random.Random(1000 + k).sample(FEATS, len(FEATS)) for k in range(K)]
for name, df in [("train", tr), ("test", te)]:
    recs = df[FEATS].to_dict("records")
    lpt, lpf = np.zeros((len(df), K)), np.zeros((len(df), K))
    for k, order in enumerate(ORDERS):
        for i in range(0, len(recs), SCORE_BS):
            texts = [row_text(r, order) for r in recs[i:i + SCORE_BS]]
            lpt[i:i + len(texts), k] = feat_logprob(texts, PRE["True"])
            lpf[i:i + len(texts), k] = feat_logprob(texts, PRE["False"])
            if (i // SCORE_BS) % 500 == 0:
                log(f"{name} order {k}: {i:,}/{len(recs):,}")
    llr = lpt - lpf
    np.save(f"{WORK}/llr_{name}.npy", llr.mean(1)); np.save(f"{WORK}/llr_orders_{name}.npy", llr)
    np.save(f"{WORK}/lp_true_{name}.npy", lpt.mean(1)); np.save(f"{WORK}/lp_false_{name}.npy", lpf.mean(1))
    if name == "train":
        from sklearn.metrics import roc_auc_score
        yy = (df["satisfaction"].map(LAB) == "True").astype(int).values
        log(f"AUC của LLR trên train (chỉ để báo cáo, nhãn không dùng khi tính): TB {roc_auc_score(yy, llr.mean(1)):.6f} | "
            + " ".join(f"order{k} {roc_auc_score(yy, llr[:, k]):.6f}" for k in range(K))
            + f" | tương quan 2 thứ tự {np.corrcoef(llr.T)[0, 1]:.4f}")
log("xong")
