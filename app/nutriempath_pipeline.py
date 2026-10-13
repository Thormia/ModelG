"""NutriEmpath-VN: pipeline LLM1 (guard) -> LLM2 (fine-tuned) -> LLM3 (verifier).

Module này không phụ thuộc vào cách chạy mô hình. Notebook truyền vào một hàm
    llm(messages, adapter: bool, max_new_tokens: int, temperature: float) -> str
- adapter=False: mô hình gốc Qwen2.5-7B-Instruct (dùng cho LLM1 và LLM3, không fine-tune)
- adapter=True : mô hình đã fine-tune LoRA (LLM2)
Hai vai dùng chung một bản 7B trên GPU, chỉ bật/tắt LoRA, nên vừa 2 GPU T4.
"""
import json
import re
import time
import unicodedata

# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------
SYSTEM_LLM2 = "Bạn là người bạn đồng hành của học sinh."  # đúng system prompt lúc huấn luyện

# Số hotline cần xác minh lại trước khi triển khai (idea.docx mục 2.5)
HOTLINE_MESSAGE = (
    "📞 Nếu bạn đang gặp nguy hiểm hoặc có ý nghĩ làm hại bản thân, hãy nói ngay với bố mẹ, "
    "thầy cô hoặc một người lớn bạn tin tưởng. Bạn có thể gọi **111** (Tổng đài quốc gia bảo vệ trẻ em, "
    "miễn phí, 24/7) hoặc **115** (cấp cứu y tế) nếu cần giúp ngay."
)
DEFAULT_SIDE_TALK = (
    "Cảm ơn bạn đã chia sẻ với mình. Mình đang ở đây lắng nghe bạn. "
    "Bạn có muốn kể thêm một chút về điều đang làm bạn bận lòng không?"
)
DEFAULT_RED_SIDE_TALK = (
    "Mình rất coi trọng những gì bạn vừa chia sẻ, và mình lo cho bạn. "
    "Bạn không phải một mình trải qua chuyện này. Hãy nói ngay với một người lớn bạn tin tưởng hôm nay nhé."
)
COOLDOWN_TURNS = 3  # không gửi lại lời khuyên dinh dưỡng trong 3 lượt liền sau lần gửi gần nhất

RED_CATEGORIES = {
    "self_harm": "ý nghĩ tự hại, tự tử, không muốn sống",
    "medical_emergency": "dấu hiệu y tế nguy hiểm: ngất/suýt ngất, đau ngực, khó thở thật sự, co giật, nôn ra máu, sụt cân nhanh",
    "drug_request": "đòi thuốc ngủ, thuốc an thần, thuốc kê đơn hoặc hỏi liều lượng thuốc",
    "extreme_diet": "nhịn ăn, ép cân, cố nôn sau khi ăn, giảm cân cực đoan",
    "abuse_violence": "bị bạo hành, xâm hại, đe dọa, bắt nạt nghiêm trọng",
    "diagnosis_request": "ép chatbot chẩn đoán bệnh (ví dụ 'mình có bị trầm cảm không')",
}
# Các nhóm phải gắn thêm thông điệp hotline cố định
HOTLINE_CATEGORIES = {"self_harm", "medical_emergency", "abuse_violence", "extreme_diet"}

RED_OVERRIDE = {
    "self_harm": "Người dùng có dấu hiệu tự hại hoặc ý nghĩ không muốn sống. An ủi thật ấm áp, ghi nhận nỗi đau, "
                 "khuyên nói NGAY hôm nay với người lớn tin cậy và gọi tổng đài 111.",
    "medical_emergency": "Người dùng mô tả dấu hiệu y tế có thể nguy hiểm. An ủi ngắn, nói rõ đây là dấu hiệu cần "
                         "bác sĩ kiểm tra, khuyên báo ngay người lớn và đi khám; nếu đang rất nặng thì gọi 115.",
    "drug_request": "Người dùng hỏi về thuốc hoặc liều thuốc. Từ chối nhẹ nhàng, KHÔNG nêu tên thuốc hay liều lượng, "
                    "giải thích chỉ bác sĩ mới kê được, khuyên nói với bố mẹ hoặc đi khám.",
    "extreme_diet": "Người dùng nói về nhịn ăn, ép cân hoặc cố nôn. Không đưa kế hoạch ăn kiêng nào, an ủi, "
                    "khuyên nói với người lớn tin cậy và gặp nhân viên y tế.",
    "abuse_violence": "Người dùng có thể đang bị bạo hành, xâm hại hoặc đe dọa. Trấn an rằng đó không phải lỗi của họ, "
                      "khuyên tìm nơi an toàn, báo người lớn tin cậy và gọi 111.",
    "diagnosis_request": "Người dùng muốn được chẩn đoán. KHÔNG chẩn đoán, ghi nhận cảm xúc, khuyến khích gặp "
                         "chuyên gia tâm lý hoặc bác sĩ để được đánh giá đúng.",
}

# ---------------------------------------------------------------------------
# Tầng 0: từ khóa (chạy trước LLM1, chỉ những cụm rất rõ nghĩa)
# ---------------------------------------------------------------------------
def strip_accents(s):
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D").lower()

# Mỗi nhóm có 2 mẫu: (có dấu, không dấu). Mẫu không dấu chỉ dùng khi tin nhắn được gõ không dấu,
# vì bỏ dấu sẽ gây nhầm: "tự tử"/"từ từ" đều thành "tu tu", "xỉu"/"xíu" đều thành "xiu".
# Khớp HARD -> red luôn, không cần hỏi LLM1.
HARD_PATTERNS = {
    "self_harm": (r"tự tử|tự sát|tự (làm|gây) (đau|hại|tổn thương)|rạch (cổ )?tay|(không|chẳng) muốn sống|"
                  r"kết thúc cuộc (đời|sống)|quá liều",
                  r"\btu sat\b|rach (co )?tay|(khong|chang) muon song|ket thuc cuoc (doi|song)|qua lieu"),
    "drug_request": (r"xanax|sertraline|lexapro|prozac|fluoxetine|diazepam|seduxen|rivotril|"
                     r"(xin|mua|cho|chỉ).{0,20}thuốc (ngủ|an thần)",
                     r"xanax|sertraline|lexapro|prozac|fluoxetine|diazepam|seduxen|rivotril|"
                     r"(xin|mua|cho|chi).{0,20}thuoc (ngu|an than)"),
}
# SOFT: chỉ là gợi ý cho LLM1 tự cân nhắc, vì có thể là cách nói ẩn dụ ("lo muốn chết", "nghẹt thở vì áp lực")
SOFT_PATTERNS = {
    "self_harm": (r"muốn chết|biến mất|không tồn tại|tự tử", r"muon chet|bien mat|khong ton tai|\btu tu\b"),
    "medical_emergency": (r"ngất|xỉu|đau ngực|khó thở|co giật|nôn ra máu|sụt cân",
                          r"\bngat\b|\bxiu\b|dau nguc|kho tho|co giat|non ra mau|sut can"),
    "extreme_diet": (r"nhịn ăn|ép cân|cố nôn|móc họng|giảm cân", r"nhin an|ep can|co non|moc hong|giam can"),
    "abuse_violence": (r"đánh đập|bạo hành|xâm hại|đe dọa|đe doạ|bắt nạt|sàm sỡ",
                       r"danh dap|bao hanh|xam hai|de doa|bat nat|sam so"),
    "drug_request": (r"thuốc|liều", r"\bthuoc\b|\blieu\b"),
    "diagnosis_request": (r"trầm cảm không|có bị .{0,20}không|chẩn đoán",
                          r"tram cam khong|co bi .{0,20}khong|chan doan"),
}

def is_unaccented(text):
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(not c.isascii() for c in letters) / len(letters) < 0.02

def keyword_scan(text):
    t = unicodedata.normalize("NFC", text).lower()
    plain = strip_accents(text) if is_unaccented(text) else None
    def match(patterns):
        return [c for c, (acc, unacc) in patterns.items()
                if re.search(acc, t) or (plain is not None and re.search(unacc, plain))]
    return match(HARD_PATTERNS), match(SOFT_PATTERNS)

# Chặn tên thuốc / liều lượng trong đầu ra (kiểm tra cứng, độc lập với LLM3)
OUTPUT_BLOCKLIST = re.compile(
    r"xanax|sertraline|lexapro|prozac|fluoxetine|diazepam|melatonin|seduxen|rivotril|thuốc ngủ|thuốc an thần|"
    r"thực phẩm chức năng|viên uống|viên bổ sung|\b\d+([.,]\d+)?\s*(mg|mcg|µg|iu|ui)\b", re.I)

# ---------------------------------------------------------------------------
# Tiện ích JSON
# ---------------------------------------------------------------------------
def parse_json(text):
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None

def valid_llm2(o):
    try:
        tp = o["thought_process"]
        return (isinstance(tp["detected_emotions"], str) and isinstance(tp["detected_symptoms"], list)
                and isinstance(tp["trigger_nutrition"], bool) and isinstance(tp["reasoning"], str)
                and isinstance(o["side_talk"], str) and o["side_talk"].strip() != ""
                and isinstance(o.get("nutritional_advice", ""), str))
    except Exception:
        return False

# ---------------------------------------------------------------------------
# LLM1: phân loại an toàn (mô hình gốc, không fine-tune)
# ---------------------------------------------------------------------------
LLM1_SYSTEM = (
    "Bạn là bộ lọc an toàn cho một chatbot hỗ trợ tinh thần dành cho học sinh 13-18 tuổi. "
    "Nhiệm vụ: đọc tin nhắn của học sinh và quyết định đây là tình huống 'red' (cần xử lý khẩn cấp, an toàn) "
    "hay 'normal' (trò chuyện bình thường).\n"
    "Các nhóm 'red':\n" + "\n".join(f"- {k}: {v}" for k, v in RED_CATEGORIES.items()) + "\n"
    "Lưu ý: cách nói ẩn dụ hoặc cường điệu KHÔNG phải red, ví dụ 'lo muốn chết', 'đau lòng quá', "
    "'nghẹt thở vì áp lực', 'đau bụng muốn xỉu' khi tới tháng. Triệu chứng nhẹ như mất ngủ vài hôm, đau đầu, "
    "mệt mỏi, tê tay chân là 'normal'. Khi thật sự nghi ngờ có nguy hiểm thì chọn 'red'.\n"
    "Chỉ trả về đúng một JSON, không viết gì thêm: "
    '{"label": "red" hoặc "normal", "category": "<một nhóm ở trên hoặc none>", "reason": "<lý do ngắn>"}'
)
LLM1_FEWSHOT = [
    ("Mình lo thi muốn chết luôn, mấy hôm nay mất ngủ với đau đầu quá.",
     {"label": "normal", "category": "none", "reason": "Cách nói cường điệu; mất ngủ, đau đầu là triệu chứng nhẹ."}),
    ("Mấy tuần nay học thể dục mình hai lần suýt ngất, tối sầm mắt phải ngồi xuống.",
     {"label": "red", "category": "medical_emergency", "reason": "Suýt ngất lặp lại cần được khám."}),
    ("Cho mình xin tên thuốc ngủ nào mạnh mạnh, uống mấy viên thì ngủ được?",
     {"label": "red", "category": "drug_request", "reason": "Hỏi thuốc ngủ và liều lượng."}),
    ("Bạn trai chia tay, mình thấy đau lòng và nghẹt thở vì tủi thân.",
     {"label": "normal", "category": "none", "reason": "Ẩn dụ cảm xúc, không có nguy hiểm."}),
    ("Mình nhịn ăn 3 ngày rồi, ăn vào là móc họng cho nôn ra để giảm cân.",
     {"label": "red", "category": "extreme_diet", "reason": "Nhịn ăn và cố nôn."}),
    ("Nhiều đêm mình nghĩ nếu mình biến mất thì mọi người sẽ nhẹ nhõm hơn.",
     {"label": "red", "category": "self_harm", "reason": "Ý nghĩ muốn biến mất."}),
]

def run_llm1(llm, user_text, soft_hints):
    msgs = [{"role": "system", "content": LLM1_SYSTEM}]
    for u, a in LLM1_FEWSHOT:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": json.dumps(a, ensure_ascii=False)}]
    hint = ""
    if soft_hints:
        hint = "\n\n(Bộ lọc từ khóa thấy dấu hiệu có thể liên quan: " + ", ".join(soft_hints) + ". Hãy tự đánh giá ngữ cảnh.)"
    msgs.append({"role": "user", "content": user_text + hint})
    raw = ""
    for attempt in range(2):
        raw = llm(msgs, adapter=False, max_new_tokens=96, temperature=0.0 if attempt == 0 else 0.2)
        o = parse_json(raw)
        if isinstance(o, dict) and o.get("label") in ("red", "normal"):
            cat = o.get("category") if o.get("category") in RED_CATEGORIES else "none"
            if o["label"] == "red" and cat == "none":
                cat = soft_hints[0] if soft_hints else "self_harm"
            return {"label": o["label"], "category": cat, "reason": str(o.get("reason", "")), "raw": raw}
    # LLM1 không trả JSON hợp lệ: ưu tiên an toàn nếu có gợi ý từ khóa
    if soft_hints:
        return {"label": "red", "category": soft_hints[0], "reason": "LLM1 lỗi định dạng; có từ khóa nghi ngờ", "raw": raw}
    return {"label": "normal", "category": "none", "reason": "LLM1 lỗi định dạng; không có từ khóa nghi ngờ", "raw": raw}

# ---------------------------------------------------------------------------
# LLM2: mô hình fine-tune
# ---------------------------------------------------------------------------
def red_system_prompt(category):
    return (
        SYSTEM_LLM2 + "\n\n[CHẾ ĐỘ KHẨN CẤP] " + RED_OVERRIDE.get(category, RED_OVERRIDE["self_harm"]) +
        " Tuyệt đối không đưa lời khuyên dinh dưỡng, không chẩn đoán, không nêu tên thuốc hay liều lượng. "
        "Trả về JSON đúng schema với trigger_nutrition = false và nutritional_advice = \"\"."
    )

def run_llm2(llm, user_text, mode, category=None, feedback=None):
    system = SYSTEM_LLM2 if mode == "normal" else red_system_prompt(category)
    if feedback:
        system += ("\n\nCâu trả lời trước của bạn bị bộ kiểm duyệt từ chối vì: " + feedback +
                   " Hãy viết lại câu trả lời, sửa đúng các lỗi này.")
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_text}]
    raw = ""
    for attempt in range(2):  # mục 2.4: lỗi JSON thì thử lại một lần ở nhiệt độ thấp
        raw = llm(msgs, adapter=True, max_new_tokens=448, temperature=0.0 if attempt == 0 else 0.3)
        o = parse_json(raw)
        if valid_llm2(o):
            o.setdefault("nutritional_advice", "")
            return o, raw
    return None, raw

# ---------------------------------------------------------------------------
# LLM3: kiểm duyệt đầu ra (mô hình gốc, không fine-tune)
# ---------------------------------------------------------------------------
LLM3_SYSTEM = (
    "Bạn là bộ kiểm duyệt cuối cùng của một chatbot hỗ trợ tinh thần cho học sinh 13-18 tuổi. "
    "Bạn nhận tin nhắn của học sinh và câu trả lời chatbot ĐỊNH gửi đi (gồm side_talk và nutritional_advice). "
    "Kiểm tra:\n"
    "1. side_talk có liên quan đúng tới tin nhắn, không bịa thêm chi tiết người dùng không nói, thấu cảm, không phán xét.\n"
    "2. Không chẩn đoán bệnh, không nêu tên thuốc, liều lượng hay thực phẩm chức năng.\n"
    "3. nutritional_advice (nếu có) chỉ được đưa khi người dùng thật sự nêu triệu chứng thể chất; thực phẩm và vi chất "
    "phải khớp với dữ liệu tham chiếu, không phóng đại tác dụng, phù hợp lứa tuổi.\n"
    "4. Trong chế độ khẩn cấp: phải an ủi, khuyên tìm người lớn hoặc hỗ trợ y tế, và KHÔNG có lời khuyên dinh dưỡng.\n"
    "Chỉ trả về đúng một JSON: "
    '{"side_talk_ok": true/false, "advice_ok": true/false, "issues": ["<lỗi ngắn gọn>", ...]}. '
    "Nếu nutritional_advice rỗng thì advice_ok = true."
)

def run_llm3(llm, user_text, mode, candidate, references, rule_flags):
    ref_text = "\n".join(f"- {r}" for r in references) if references else "(không có dữ liệu tham chiếu khớp)"
    flags = ("\nCảnh báo tự động: " + "; ".join(rule_flags)) if rule_flags else ""
    content = (
        f"Chế độ: {'KHẨN CẤP' if mode == 'red' else 'bình thường'}\n"
        f"Tin nhắn học sinh: {user_text}\n\n"
        f"Triệu chứng chatbot phát hiện: {candidate['thought_process'].get('detected_symptoms', [])}\n"
        f"side_talk: {candidate['side_talk']}\n"
        f"nutritional_advice: {candidate.get('nutritional_advice', '')}\n\n"
        f"Dữ liệu tham chiếu dinh dưỡng (bảng Task 2):\n{ref_text}{flags}"
    )
    msgs = [{"role": "system", "content": LLM3_SYSTEM}, {"role": "user", "content": content}]
    raw = llm(msgs, adapter=False, max_new_tokens=160, temperature=0.0)
    o = parse_json(raw)
    if not isinstance(o, dict) or "side_talk_ok" not in o:
        # LLM3 lỗi định dạng: chỉ dựa vào luật cứng
        return {"side_talk_ok": not rule_flags, "advice_ok": not rule_flags,
                "issues": rule_flags or ["LLM3 lỗi định dạng"], "raw": raw}
    return {"side_talk_ok": bool(o.get("side_talk_ok")), "advice_ok": bool(o.get("advice_ok", True)),
            "issues": [str(x) for x in o.get("issues", [])][:5], "raw": raw}

# ---------------------------------------------------------------------------
# Tra cứu bảng Task 2 để LLM3 có căn cứ kiểm tra lời khuyên
# ---------------------------------------------------------------------------
class NutritionKB:
    def __init__(self, records):
        self.records = []
        for r in records or []:
            food = r["food_sources"][0]
            self.records.append({
                "food": food,
                "food_key": strip_accents(re.sub(r"\(.*?\)", "", food)).strip(),
                "symptom_key": strip_accents(r["physical_symptom"]),
                "line": f"{food} | vi chất: {r['nutrient_target']} | triệu chứng: {r['physical_symptom']}",
            })

    def lookup(self, symptoms, advice, k=8):
        adv = strip_accents(advice or "")
        hits = [r for r in self.records if r["food_key"] and r["food_key"] in adv]
        sym_tokens = set()
        for s in symptoms or []:
            sym_tokens |= {w for w in re.findall(r"\w+", strip_accents(s)) if len(w) > 2}
        if sym_tokens:
            scored = []
            for r in self.records:
                score = sum(1 for w in sym_tokens if w in r["symptom_key"])
                if score:
                    scored.append((score, r))
            scored.sort(key=lambda x: -x[0])
            for _, r in scored:
                if len(hits) >= k:
                    break
                if r not in hits:
                    hits.append(r)
        return [r["line"] for r in hits[:k]], bool(advice) and not any(r["food_key"] in adv for r in self.records)

# ---------------------------------------------------------------------------
# Điều phối một lượt chat
# ---------------------------------------------------------------------------
def new_session():
    return {"turn": 0, "last_advice_turn": None}

def respond(llm, user_text, session, kb=None, verify=True):
    """Trả về (câu trả lời hiển thị, trace để debug). session được cập nhật tại chỗ."""
    t0 = time.time()
    session["turn"] += 1
    trace = {"turn": session["turn"]}

    # Tầng 0 + LLM1
    hard, soft = keyword_scan(user_text)
    trace["keywords"] = {"hard": hard, "soft": soft}
    if hard:
        guard = {"label": "red", "category": hard[0], "reason": "Khớp từ khóa nguy hiểm", "raw": ""}
    else:
        guard = run_llm1(llm, user_text, soft)
    trace["llm1"] = guard
    mode = guard["label"]

    # LLM2 (+ LLM3, sửa tối đa một lần)
    feedback = None
    final = None
    for attempt in range(2):
        cand, raw2 = run_llm2(llm, user_text, mode, guard["category"], feedback)
        trace[f"llm2_try{attempt + 1}"] = raw2
        if cand is None:
            break
        if mode == "red" or not cand["thought_process"]["trigger_nutrition"]:
            cand["thought_process"]["trigger_nutrition"] = False
            cand["nutritional_advice"] = ""  # phòng thủ nhiều lớp: trigger false thì bỏ advice
        rule_flags = []
        if OUTPUT_BLOCKLIST.search(cand["side_talk"] + " " + cand["nutritional_advice"]):
            rule_flags.append("có tên thuốc, liều lượng hoặc thực phẩm chức năng")
        refs, unknown_food = (kb.lookup(cand["thought_process"]["detected_symptoms"], cand["nutritional_advice"])
                              if kb else ([], False))
        if unknown_food:
            rule_flags.append("thực phẩm trong lời khuyên không có trong bảng Task 2")
        if not verify:
            final = cand
            break
        check = run_llm3(llm, user_text, mode, cand, refs, rule_flags)
        trace[f"llm3_try{attempt + 1}"] = check
        if check["side_talk_ok"] and check["advice_ok"]:
            final = cand
            break
        if attempt == 0:
            feedback = "; ".join(check["issues"]) or "nội dung chưa phù hợp"
            continue
        # Lần sửa thứ hai vẫn chưa đạt: giữ phần đạt, thay phần lỗi bằng nội dung an toàn
        if not check["advice_ok"]:
            cand["nutritional_advice"] = ""
            cand["thought_process"]["trigger_nutrition"] = False
        if not check["side_talk_ok"]:
            cand["side_talk"] = DEFAULT_RED_SIDE_TALK if mode == "red" else DEFAULT_SIDE_TALK
        final = cand

    if final is None:  # LLM2 không sinh được JSON hợp lệ sau khi thử lại
        final = {"thought_process": {"trigger_nutrition": False, "detected_symptoms": []},
                 "side_talk": DEFAULT_RED_SIDE_TALK if mode == "red" else DEFAULT_SIDE_TALK,
                 "nutritional_advice": ""}
        trace["fallback"] = True

    # Cooldown 3 lượt cho lời khuyên dinh dưỡng
    advice = final.get("nutritional_advice", "")
    if advice and session["last_advice_turn"] is not None and \
            session["turn"] - session["last_advice_turn"] <= COOLDOWN_TURNS:
        trace["cooldown_dropped_advice"] = True
        advice = ""
    if advice:
        session["last_advice_turn"] = session["turn"]

    parts = [final["side_talk"].strip()]
    if advice:
        parts.append("🥗 **Gợi ý dinh dưỡng:** " + advice.strip())
    if mode == "red" and guard["category"] in HOTLINE_CATEGORIES:
        parts.append(HOTLINE_MESSAGE)
    trace["mode"] = mode
    trace["seconds"] = round(time.time() - t0, 1)
    return "\n\n".join(parts), trace
