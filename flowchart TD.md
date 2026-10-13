flowchart TD
  U[Tin nhắn người dùng] --> S{Kiểm tra<br/>dấu hiệu tự hại}
  S -- có --> H[Thông điệp hotline<br/>cố định]
  S -- không --> C[Đọc session store<br/>lấy lượt đã khuyên]
  C --> L[LLM fine-tuned<br/>sinh JSON]
  L --> V[Backend parse<br/>và kiểm tra schema]
  V --> G{trigger_nutrition true<br/>và hết cooldown 3 lượt?}
  V --> ST[(Session store)]
  G -- không --> B1[Hiển thị side_talk]
  G -- có --> B2[Hiển thị side_talk<br/>và nutritional_advice]
  B2 -. ghi lượt đã khuyên .-> ST
  ST -. lượt sau .-> C
  ST -.-> M[Log ẩn danh<br/>duyệt mẫu hàng tuần]
  H --> R[Phản hồi cho user]
  B1 --> R
  B2 --> R