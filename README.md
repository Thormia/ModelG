# NutriEmpath-VN: Symptom-Gated Nutrition Advice for Empathetic Teen Support

Chatbot hỗ trợ tinh thần cho thanh thiếu niên Việt Nam. Mô hình luôn trả lời thấu cảm, và **chỉ** đưa lời khuyên
dinh dưỡng khi người dùng kể một triệu chứng cơ thể cụ thể (gating). Các ca cần chuyên gia (Nhóm C) được chuyển hướng
tới người lớn / hotline (111, 115) thay vì khuyên ăn uống.

Mô hình: **Qwen2.5-7B-Instruct + LoRA** (r=16), fine-tune trên Kaggle TPU v5e-8, đầu ra là JSON:

```json
{"thought_process": {"detected_emotions": "...", "detected_symptoms": ["..."],
                     "trigger_nutrition": true, "reasoning": "..."},
 "side_talk": "...",
 "nutritional_advice": "..."}
```

## Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| `data/task1_tam_ly_5000.json` | Task 1: 4.893 mẫu tư vấn tâm lý (ESConv, EmpatheticDialogues, CACTUS, counsel-chat, Student-10K) |
| `data/task2_dinh_duong_422_mon.json` | Task 2: 422 cặp triệu chứng → dưỡng chất → thực phẩm, có trích dẫn (PubMed, EU Reg. 432/2012, VN FCT 2007, USDA) |
| `data/task3_hoi_thoai_5000.json` | Task 3: 4.428 hội thoại huấn luyện (Nhóm A 1.495 / B 2.485 / C 448), có metadata |
| `data/task3_hoi_thoai_5000_chatml.jsonl` | Task 3 dạng ChatML, khớp từng dòng với file trên |
| `expert/task3_expert_verify_40.jsonl` | 40 mẫu Task 3 gửi chuyên gia duyệt |
| `train/kaggle_tpu_finetune_v2.ipynb` | Fine-tune LoRA nhiều epoch trên TPU, lưu adapter mỗi epoch, vẽ loss/perplexity |
| `train/kaggle_gpu_metrics.ipynb` | Đánh giá từng epoch và baseline trên GPU T4 x2: F1 gating, EM, ROUGE, BERTScore, JSON hợp lệ, Nhóm C |
| `model-g.ipynb` | Notebook huấn luyện 6 epoch đã chạy (kèm output) |
| `model-g-metric.ipynb` | Notebook đánh giá đã chạy (kèm output): bảng kết quả và biểu đồ, mô hình đề xuất Epoch 3 |
| `diagrams/` | Flowchart mô hình đề xuất và luồng dữ liệu (`.drawio`, mở bằng draw.io) |
| `app/` | Pipeline demo (từ khóa → LLM guard → LLM fine-tune → LLM kiểm duyệt) và giao diện Gradio; không bắt buộc |
| `README.txt` | Mô tả schema chi tiết của các file dữ liệu |

File `.py` trong `train/` và `app/` là bản text của notebook cùng tên.

## Chia dữ liệu

Từ `task3_hoi_thoai_5000_chatml.jsonl`, phân tầng theo nhóm, seed 42: **4.228 train / 200 validation**
(validation: Nhóm A 68, B 112, C 20).

## Kết quả (validation, 200 mẫu)

Huấn luyện 6 epoch. Mô hình đề xuất là **epoch 3**: F1 gating ngang epoch 2 (0,991) nhưng không khuyên dinh dưỡng sai
lần nào (precision 1,000, FP = 0, FN = 2), EM triệu chứng và ROUGE-L side_talk cao nhất. Epoch 2 có validation loss
thấp nhất (0,682); validation loss bắt đầu tăng từ epoch 3.

| Mô hình | Val loss | PPL | JSON hợp lệ | F1 gating | EM triệu chứng | ROUGE-L side_talk | BERTScore F1 side_talk | Nhóm C chuyển hướng an toàn |
|---|---|---|---|---|---|---|---|---|
| Epoch 1 | 0,721 | 2,06 | 100% | 0,982 | 0,695 | 0,218 | 0,890 | 95% |
| Epoch 2 | 0,682 | 1,98 | 100% | 0,991 | 0,700 | 0,229 | 0,892 | 100% |
| **Epoch 3 (đề xuất)** | 0,705 | 2,02 | **100%** | **0,991** | **0,725** | **0,235** | **0,892** | **100%** |
| Epoch 4 | 0,778 | 2,18 | 99,5% | 0,986 | 0,715 | 0,218 | 0,889 | 95% |
| Epoch 5 | 0,899 | 2,46 | 97% | 0,982 | 0,670 | 0,216 | 0,887 | 100% |
| Epoch 6 | 0,941 | 2,56 | 97,5% | 0,982 | 0,675 | 0,215 | 0,887 | 100% |
| Base + prompt chi tiết | – | – | 100% | 0,086 | 0,305 | 0,115 | 0,876 | 25% |
| Base zero-shot | – | – | 0% | 0,000 | 0,000 | 0,101 | 0,856 | 0% |

BERTScore dùng `xlm-roberta-large`, không rescale baseline. ROUGE dùng tokenizer giữ dấu tiếng Việt.
Validation cũng là tập dùng để chọn epoch, nên các con số trên có phần lạc quan; chưa có tập test độc lập.
Chênh lệch giữa epoch 2 và 3 tương ứng 1–5 mẫu trên 200, cần kiểm chứng thêm trên tập test.

## Chạy lại trên Kaggle

1. Tạo dataset Kaggle chứa thư mục `data/`.
2. `train/kaggle_tpu_finetune_v2.ipynb`: Accelerator **TPU v5e-8**, thêm dataset làm input, Save & Run All
   (khoảng 1,5 giờ cho 6 epoch). Output: `qwen25-7b-lora-6ep/epoch-*`, `log_history.json`, `figures/`.
3. `train/kaggle_gpu_metrics.ipynb`: Accelerator **GPU T4 x2**, input là dataset và output của bước 2,
   Save & Run All. Output: `metrics/` (CSV, LaTeX, predictions, `table_paper.tex`) và `figures/`.
   Epoch đề xuất đặt ở `SELECTED_EPOCH` (mặc định 3).

## Lưu ý

Chatbot không thay thế bác sĩ hay chuyên gia tâm lý. Lời khuyên dinh dưỡng chỉ mang tính tham khảo từ thực phẩm
thông thường, không đề cập thuốc hay thực phẩm chức năng.
