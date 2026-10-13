# %% [markdown]
# # Fine-tune lại Qwen2.5-7B-Instruct (LoRA) nhiều epoch để vẽ biểu đồ — Kaggle TPU v5e-8
#
# Giống notebook huấn luyện trước (cùng dữ liệu, cùng cách chia train/val, cùng tham số LoRA), khác ở chỗ:
# - Huấn luyện **6 epoch** (đổi `NUM_EPOCHS` nếu muốn), lưu LoRA adapter sau **mỗi** epoch.
# - Đánh giá validation **4 lần mỗi epoch** và cuối mỗi epoch, nên có đường validation loss / perplexity mịn để vẽ.
# - Ghi toàn bộ log (train loss, val loss, learning rate) ra `log_history.json` sau mỗi lần log,
#   để notebook `kaggle_gpu_metrics` đọc lại và vẽ biểu đồ cho bài báo.
# - Vẽ sẵn biểu đồ loss và perplexity ngay trong notebook này (`figures/`).
#
# Train nhiều epoch hơn sẽ thấy rõ điểm bắt đầu overfit (validation loss tăng lại); epoch dùng cho hệ thống vẫn
# là epoch có validation loss thấp nhất.
#
# **Cài đặt Kaggle**: Accelerator **TPU VM v5e-8**, Internet **On**, Input là dataset chứa
# `task3_hoi_thoai_5000_chatml.jsonl` và `task3_hoi_thoai_5000.json`. Chạy bằng **Save Version → Save & Run All**
# và tạo notebook MỚI (đừng lưu đè lên modelG101 để không mất adapter cũ).

# %%
# Kaggle TPU image có sẵn torch + torch_xla. Không nâng cấp torch/torch_xla để tránh lệch phiên bản libtpu.
!pip install -q -U "transformers>=4.45,<5" "peft>=0.13" "accelerate>=1.0" "datasets>=3.0"

# %%
import os, json, math, random, glob
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_USE_BF16", "0")

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

xr.use_spmd()
print("torch", torch.__version__, "| torch_xla", torch_xla.__version__, "| devices", xr.global_runtime_device_count())

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback, set_seed
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file

# %%
SEED = 42
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
NUM_EPOCHS = 6
EVALS_PER_EPOCH = 4
RUN_NAME = f"qwen25-7b-lora-{NUM_EPOCHS}ep"
OUTPUT_DIR = f"/kaggle/working/{RUN_NAME}"
FIG_DIR = "/kaggle/working/figures"
MAX_LEN_CAP = 2048
N_VAL = 200
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert hits, f"Không tìm thấy {name} trong /kaggle/input"
    return hits[0]

CHATML_PATH = find("task3_hoi_thoai_5000_chatml.jsonl")
META_PATH = find("task3_hoi_thoai_5000.json")
set_seed(SEED)

# %%
rows = [json.loads(l) for l in open(CHATML_PATH, encoding="utf-8") if l.strip()]
meta = json.load(open(META_PATH, encoding="utf-8"))
assert len(rows) == len(meta), "chatml và json phải cùng số dòng, cùng thứ tự"
groups = [m["metadata"]["scenario_type"] for m in meta]

# Chia phân tầng theo nhóm A/B/C — giống hệt lần huấn luyện trước
rng = random.Random(SEED)
by_group = {}
for i, g in enumerate(groups):
    by_group.setdefault(g, []).append(i)
val_idx = []
for g, idx in by_group.items():
    rng.shuffle(idx)
    val_idx += idx[: round(N_VAL * len(idx) / len(rows))]
val_set = set(val_idx)
train_rows = [rows[i] for i in range(len(rows)) if i not in val_set]
val_rows = [rows[i] for i in sorted(val_set)]
print("train", len(train_rows), "| val", len(val_rows))

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = "<|endoftext|>"

def encode(messages):
    """Token hóa theo chat template của Qwen; chỉ tính loss trên phần assistant."""
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    assert full[: len(prompt)] == prompt
    labels = [-100] * len(prompt) + full[len(prompt):]
    return full, labels

enc_train = [encode(r["messages"]) for r in train_rows]
enc_val = [encode(r["messages"]) for r in val_rows]
lens = [len(ids) for ids, _ in enc_train + enc_val]
print("token length: max", max(lens), "| p99", int(np.percentile(lens, 99)), "| median", int(np.median(lens)))
MAX_LEN = min(MAX_LEN_CAP, int(math.ceil(max(lens) / 128) * 128))
print("MAX_LEN =", MAX_LEN)

def to_dataset(encoded):
    data = {"input_ids": [], "attention_mask": [], "labels": []}
    for ids, labels in encoded:
        if len(ids) > MAX_LEN:
            continue
        pad = MAX_LEN - len(ids)
        data["input_ids"].append(ids + [tokenizer.pad_token_id] * pad)
        data["attention_mask"].append([1] * len(ids) + [0] * pad)
        data["labels"].append(labels + [-100] * pad)
    return Dataset.from_dict(data).with_format("torch")

train_ds = to_dataset(enc_train)
val_ds = to_dataset(enc_val)

BATCH = 8
steps_per_epoch = len(train_ds) // BATCH
EVAL_STEPS = max(1, steps_per_epoch // EVALS_PER_EPOCH)
print("steps/epoch", steps_per_epoch, "| eval mỗi", EVAL_STEPS, "bước | tổng", steps_per_epoch * NUM_EPOCHS, "bước")

# %%
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.config.use_cache = False
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# %%
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH,    # SPMD FSDPv2: đây là batch toàn cục, chia hết cho 8 chip
    per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=1,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    optim="adamw_torch",
    bf16=True,
    seed=SEED,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="no",                   # checkpoint của Trainer ghi cả mô hình 15 GB; tự lưu LoRA bằng callback
    dataloader_drop_last=True,
    remove_unused_columns=False,
    report_to="none",
    fsdp="full_shard",
    fsdp_config={
        "transformer_layer_cls_to_wrap": ["Qwen2DecoderLayer"],
        "xla": True,
        "xla_fsdp_v2": True,
        "xla_fsdp_grad_ckpt": True,
    },
)

peft_model = model

def save_lora(out_dir):
    """Gom các tham số LoRA (đang chia trên 8 chip) về CPU và lưu theo định dạng PEFT adapter."""
    xm.mark_step()
    state = {}
    for name, param in peft_model.named_parameters():
        if not param.requires_grad:
            continue
        for wrapper in ("_orig_module.", "_fsdp_wrapped_module.", "_checkpoint_wrapped_module."):
            name = name.replace(wrapper, "")
        name = name.replace(".default.", ".")
        state[name] = param.detach().to("cpu").contiguous()
    os.makedirs(out_dir, exist_ok=True)
    save_file(state, os.path.join(out_dir, "adapter_model.safetensors"))
    peft_model.peft_config["default"].save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"saved {len(state)} LoRA tensors -> {out_dir}")

LOG_PATH = os.path.join(OUTPUT_DIR, "log_history.json")

class EpochCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        save_lora(os.path.join(OUTPUT_DIR, f"epoch-{round(state.epoch)}"))
        if state.global_step % EVAL_STEPS != 0:
            control.should_evaluate = True    # luôn có một điểm validation đúng cuối mỗi epoch
        return control

    def on_log(self, args, state, control, **kwargs):
        # ghi log liên tục: phiên bị ngắt giữa chừng vẫn còn dữ liệu để vẽ
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(state.log_history, f, ensure_ascii=False, indent=1)

trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
                  callbacks=[EpochCallback()])

# %%
trainer.train()
with open(LOG_PATH, "w", encoding="utf-8") as f:
    json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=1)

# %%
# Validation loss / perplexity cuối mỗi epoch và epoch tốt nhất
log = trainer.state.log_history
epoch_end = {}
for h in log:
    if "eval_loss" in h and abs(h["epoch"] - round(h["epoch"])) < 1e-3:
        epoch_end[round(h["epoch"])] = h["eval_loss"]
for e, l in sorted(epoch_end.items()):
    print(f"epoch {e}: val_loss {l:.4f} | perplexity {math.exp(l):.3f}")
best_epoch = min(epoch_end, key=epoch_end.get)
print("epoch tốt nhất theo validation loss:", best_epoch)
json.dump({"best_epoch": best_epoch, "epoch_val_loss": epoch_end},
          open(os.path.join(OUTPUT_DIR, "summary.json"), "w"), indent=1)
print(sorted(os.listdir(OUTPUT_DIR)))

# %%
# Biểu đồ loss và perplexity cho bài báo (PNG 300 dpi + PDF vector)
import matplotlib.pyplot as plt

LANG = "vi"   # "en" nếu bài viết bằng tiếng Anh
TXT = {
    "vi": {"step": "Epoch", "loss": "Loss (cross-entropy, phần assistant)", "train": "Train loss",
           "val": "Validation loss", "best": "Epoch tốt nhất", "ppl": "Perplexity trên validation",
           "ppl_t": "Perplexity validation theo epoch", "loss_t": "Train loss và validation loss"},
    "en": {"step": "Epoch", "loss": "Loss (cross-entropy, assistant tokens)", "train": "Train loss",
           "val": "Validation loss", "best": "Best epoch", "ppl": "Validation perplexity",
           "ppl_t": "Validation perplexity by epoch", "loss_t": "Training and validation loss"},
}[LANG]
BLUE, ORANGE = "#2a78d6", "#eb6834"         # slot 1-2 của bảng màu categorical đã kiểm định
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white",
})

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"))
    plt.show()

tr = [(h["epoch"], h["loss"]) for h in log if "loss" in h and "eval_loss" not in h]
ev = [(h["epoch"], h["eval_loss"]) for h in log if "eval_loss" in h]
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(*zip(*tr), color=BLUE, lw=1.5, label=TXT["train"])
ax.plot(*zip(*ev), color=ORANGE, lw=2, marker="o", ms=4, label=TXT["val"])
ax.axvline(best_epoch, color=INK2, ls=":", lw=1)
ax.annotate(f"{TXT['best']}: {best_epoch}", (best_epoch, epoch_end[best_epoch]), xytext=(8, 18),
            textcoords="offset points", color=INK, fontsize=9)
ax.set_xlabel(TXT["step"]); ax.set_ylabel(TXT["loss"]); ax.set_title(TXT["loss_t"], loc="left", color=INK)
ax.set_xticks(range(0, NUM_EPOCHS + 1))
ax.legend(loc="upper right")
save(fig, "loss_curves")

fig, ax = plt.subplots(figsize=(6.4, 3.4))
xs = sorted(epoch_end)
ax.plot(xs, [math.exp(epoch_end[e]) for e in xs], color=BLUE, lw=2, marker="o", ms=6)
for e in xs:
    ax.annotate(f"{math.exp(epoch_end[e]):.3f}", (e, math.exp(epoch_end[e])), xytext=(0, 7),
                textcoords="offset points", ha="center", fontsize=8, color=INK2)
ax.set_xlabel(TXT["step"]); ax.set_ylabel(TXT["ppl"]); ax.set_title(TXT["ppl_t"], loc="left", color=INK)
ax.set_xticks(xs)
save(fig, "val_perplexity")

# %%
# Nén adapter + log + biểu đồ để tải về (Kaggle không có lệnh zip)
import shutil
print(shutil.make_archive(f"/kaggle/working/{RUN_NAME}", "zip", "/kaggle/working", RUN_NAME))
print(shutil.make_archive("/kaggle/working/figures", "zip", "/kaggle/working", "figures"))
