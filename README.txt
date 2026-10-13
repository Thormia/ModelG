DỮ LIỆU CUỐI - PAPER G
================================================================

task1_tam_ly_5000.json  (Task 1 - 4.893 bản ghi, JSON array)
  {"type": "Psychology",
   "emotion_or_situation": "Anxiety / Academic pressure: <tình huống tiếng Việt>",
   "counseling_technique": "Question (Đặt câu hỏi), Affirmation and Reassurance (Khẳng định & Trấn an)
                            [ | CBT: Evidence-Based Questioning (Đặt câu hỏi dựa trên bằng chứng)]",
   "sample_response": "<câu trả lời mẫu tiếng Việt>",
   "ref": "ESConv Dataset (conv#1211 turn6; problem_type: ongoing depression)"}
  Nguồn: ESConv 2.376 | EmpatheticDialogues 1.088 | CACTUS 690 | counsel-chat 443 | Student-10K 296.

task2_dinh_duong_422_mon.json  (Task 2 - 422 bản ghi, JSON array)
  {"physical_symptom": "Tê bì; kim châm tay chân | ...",      1-4 triệu chứng, ngăn bởi " | "
   "nutrient_target": "Thiamin (Vitamin B1)",
   "food_sources": ["Gạo lứt"],                               mỗi bản ghi 1 thực phẩm
   "mechanism": "Gạo lứt cung cấp khoảng 10% DV ... cần đi khám để xác định nguyên nhân.",
   "ref": "PubMed: PMID 9557427 / EU Reg. 432/2012: '...' / Số liệu: VN_FCT_2007_MaSo_1005; xác nhận: USDA_FDC_2512380"}
  30 bản ghi có physical_symptom bắt đầu bằng "[CẢNH BÁO]" (đồ uống có đường, caffeine, đồ ăn vặt
  siêu chế biến): mechanism khuyên hạn chế, không dùng để cải thiện triệu chứng.

task3_hoi_thoai_5000.json  (Task 3 - 4.428 mẫu, JSON array, format "đầu ra bước 2" của spec)
  {"metadata": {"ref": "<nguồn Task 1> || <nguồn Task 2>", "scenario_type": "Nhóm B - Đủ dữ kiện"},
   "instruction": "<tin nhắn của teen>",
   "output": {"thought_process": {"detected_emotions": "...", "detected_symptoms": [...],
                                  "trigger_nutrition": true, "reasoning": "..."},
              "side_talk": "...",
              "nutritional_advice": "..."}}
  scenario_type: Nhóm A - Chỉ cảm xúc (1.495, trigger false, advice rỗng)
                 Nhóm B - Đủ dữ kiện (2.485, trigger true)
                 Nhóm C - Cần chuyển chuyên gia (448, trigger false, advice rỗng)

task3_hoi_thoai_5000_chatml.jsonl  (Task 3 - 4.428 dòng, JSONL ChatML; đóng gói từ task3_hoi_thoai_5000.json,
  bỏ metadata)
  {"messages": [{"role": "system", "content": "Bạn là người bạn đồng hành của học sinh."},
                {"role": "user", "content": "<instruction>"},
                {"role": "assistant", "content": "<output dạng chuỗi JSON>"}]}
