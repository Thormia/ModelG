# %% [markdown]
# # NutriEmpath-VN: hệ thống hoàn chỉnh + giao diện Gradio (Kaggle GPU T4 x2)
#
# Luồng xử lý mỗi tin nhắn:
# 1. **Tầng 0 (từ khóa)**: cụm rất rõ nghĩa (tự tử, rạch tay, tên thuốc an thần...) chuyển thẳng sang chế độ khẩn cấp.
# 2. **LLM1 – bộ lọc an toàn** (Qwen2.5-7B-Instruct gốc, không fine-tune): phân loại `red` / `normal`.
# 3. **LLM2 – mô hình fine-tune** (LoRA): `normal` thì trả lời như lúc huấn luyện; `red` thì system prompt khẩn cấp
#    được ghi đè để chỉ an ủi và hướng tới người lớn / hotline, không khuyên dinh dưỡng.
# 4. **LLM3 – bộ kiểm duyệt** (mô hình gốc): kiểm tra câu trả lời định gửi (bịa chi tiết, lạc đề, thuốc/liều, thực phẩm
#    không có trong bảng Task 2...). Lỗi thì LLM2 viết lại một lần theo góp ý; vẫn lỗi thì bỏ phần lỗi, thay nội dung an toàn.
# 5. Backend: kiểm tra schema JSON, bỏ advice khi trigger false, cooldown 3 lượt, gắn thông điệp hotline cố định.
#
# LLM1, LLM2, LLM3 dùng chung **một** bản Qwen2.5-7B trên GPU (bật/tắt LoRA), nên vừa 2 GPU T4.
#
# **Cài đặt Kaggle**: Accelerator **GPU T4 x2**, Internet **On**, chạy kiểu **interactive** (không Save Version),
# vì Gradio cần notebook còn chạy để giữ link.
# **Input**: (1) output của notebook huấn luyện (chứa `qwen25-7b-lora/epoch-*`);
# (2) dataset có `task2_dinh_duong_422_mon.json` (để LLM3 đối chiếu lời khuyên; thiếu file này app vẫn chạy).

# %%
# Không ghim phiên bản transformers để giữ đúng bộ thư viện Kaggle cài sẵn cho Gradio (huggingface-hub mới).
# Ảnh GPU của Kaggle có torchao 0.10, peft mới đòi >= 0.16 nếu có cài, nên gỡ đi.
!pip uninstall -y -q torchao
!pip install -q -U peft accelerate
!pip install -q gradio

# %%
%%writefile /kaggle/working/nutriempath_pipeline.py
<<PIPELINE_SOURCE>>

# %%
import sys, os, glob, json, gc
sys.path.insert(0, "/kaggle/working")
import importlib
import nutriempath_pipeline as P
importlib.reload(P)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# Thư mục adapter: đổi thành epoch tốt nhất của lần huấn luyện bạn muốn dùng
ADAPTER_PATTERN = "qwen25-7b-lora/epoch-2/adapter_config.json"

def find(pattern, required=True):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    if required:
        assert hits, f"Không tìm thấy {pattern} trong /kaggle/input"
    return hits[0] if hits else None

ADAPTER_PATH = os.path.dirname(find(ADAPTER_PATTERN))
TASK2_PATH = find("task2_dinh_duong_422_mon.json", required=False)
print("adapter:", ADAPTER_PATH)
print("task2:", TASK2_PATH or "không có (LLM3 kiểm tra không cần bảng tham chiếu)")
kb = P.NutritionKB(json.load(open(TASK2_PATH, encoding="utf-8")) if TASK2_PATH else [])

# %%
assert torch.cuda.is_available(), "Chưa bật GPU: Settings > Accelerator > GPU T4 x2"
for _name in ("model", "base"):
    if _name in globals():
        del globals()[_name]
gc.collect()
torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = "left"
max_memory = {i: "14GiB" for i in range(torch.cuda.device_count())}
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="auto", max_memory=max_memory)
placement = {str(d) for d in base.hf_device_map.values()}
assert not ({"cpu", "disk"} & placement), "Có lớp bị đẩy sang CPU/disk"
model = PeftModel.from_pretrained(base, ADAPTER_PATH)
model.eval()
print("device map:", placement)

@torch.no_grad()
def llm(messages, adapter=True, max_new_tokens=256, temperature=0.0):
    """adapter=True: LLM2 (fine-tune). adapter=False: mô hình gốc cho LLM1/LLM3."""
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        kwargs.update(do_sample=False)
    if adapter:
        out = model.generate(**enc, **kwargs)
    else:
        with model.disable_adapter():
            out = model.generate(**enc, **kwargs)
    return tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

# %%
# Chạy thử 4 tình huống trước khi mở giao diện
tests = [
    "Mai thi rồi mà mình chưa ôn xong, lo quá, mấy hôm nay mất ngủ với hay bị tê tay chân.",
    "Bạn thân giận mình cả tuần nay, mình buồn lắm.",
    "Mấy tuần nay học thể dục mình hai lần suýt ngất, tối sầm mắt phải ngồi xuống.",
    "Cho mình hỏi uống thuốc ngủ mấy viên thì ngủ được?",
]
for t in tests:
    s = P.new_session()
    reply, trace = P.respond(llm, t, s, kb=kb)
    print("USER :", t)
    print("MODE :", trace["mode"], trace["llm1"]["category"], "|", trace["seconds"], "giây")
    print("BOT  :", reply, "\n" + "-" * 80)

# %%
import gradio as gr

chat_kwargs = {"type": "messages"} if int(gr.__version__.split(".")[0]) < 6 else {}

def chat(user_msg, history, session):
    history = list(history or [])
    if not user_msg or not user_msg.strip():
        yield "", history, session, {}
        return
    history += [{"role": "user", "content": user_msg},
                {"role": "assistant", "content": "⏳ Đang lắng nghe và kiểm tra câu trả lời..."}]
    yield "", history, session, {}
    try:
        reply, trace = P.respond(llm, user_msg, session, kb=kb)
    except Exception as e:  # không bao giờ để lộ lỗi thô cho người dùng
        reply, trace = P.DEFAULT_SIDE_TALK, {"error": repr(e)}
    history[-1] = {"role": "assistant", "content": reply}
    yield "", history, session, trace

def reset():
    return "", [], P.new_session(), {}

with gr.Blocks(title="NutriEmpath-VN") as demo:
    gr.Markdown(
        "# 🌱 NutriEmpath-VN\n"
        "Người bạn đồng hành lắng nghe bạn, và chỉ gợi ý thực phẩm khi bạn kể về một triệu chứng cơ thể cụ thể.\n\n"
        "_Chatbot không thay thế bác sĩ hay chuyên gia tâm lý. Khi khẩn cấp hãy gọi **111** hoặc **115**._"
    )
    session = gr.State(P.new_session())
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=520, label="Trò chuyện", **chat_kwargs)
            msg = gr.Textbox(placeholder="Hôm nay bạn thấy thế nào?", show_label=False, lines=2)
            with gr.Row():
                send = gr.Button("Gửi", variant="primary")
                clear = gr.Button("Cuộc trò chuyện mới")
        with gr.Column(scale=2):
            with gr.Accordion("Chi tiết xử lý: tầng 0 → LLM1 → LLM2 → LLM3", open=True):
                trace_box = gr.JSON(label="Trace của lượt gần nhất")
    gr.Examples(examples=[[t] for t in tests], inputs=msg, label="Ví dụ")

    send.click(chat, [msg, chatbot, session], [msg, chatbot, session, trace_box])
    msg.submit(chat, [msg, chatbot, session], [msg, chatbot, session, trace_box])
    clear.click(reset, None, [msg, chatbot, session, trace_box])

# Một GPU chỉ phục vụ một lượt mỗi lúc; share=True tạo link công khai tạm thời (gradio.live)
demo.queue(default_concurrency_limit=1).launch(share=True, debug=True)
