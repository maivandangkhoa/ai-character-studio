# Mayalin Studio — Web UI (chạy trên VM)

Web UI để nhập prompt và nhận ảnh. Generate thật sự chạy trên **Kaggle batch** (free);
VM chỉ điều phối qua Kaggle API và hiển thị kết quả.

## Kiến trúc
```
Trình duyệt → FastAPI (VM) → Kaggle API → kernel "mayalin-generate" (GPU)
                                              ↑ đọc LoRA từ dataset mayalin-lora
        ảnh ← pull output ← /kaggle/working/images ←┘
```

## Cài đặt một lần (trên VM)

1. **Kaggle token**: Kaggle → Settings → Create New API Token → đặt file vào `~/.kaggle/kaggle.json` (`chmod 600`).
2. **HF token cho generate**: trên Kaggle, thêm Secret tên `HF_TOKEN` cho kernel (Add-ons → Secrets) — FLUX.1-dev là gated.
3. Cài deps web:
   ```bash
   pip install -r webapp/requirements.txt
   ```

## Sau khi train xong (lấy LoRA về rồi publish)

```bash
# 1) Kéo LoRA từ kernel train về VM (một lần)
kaggle kernels output maivandangkhoa/<train-kernel> -p ./trained

# 2) Đẩy LoRA lên dataset mà generate kernel sẽ đọc
python scripts/publish_lora.py ./trained/.../lora.safetensors
```

## Chạy web

```bash
uvicorn webapp.app:app --host 0.0.0.0 --port 8000
```

Mở `http://<vm-ip>:8000`, nhập prompt (mỗi dòng một ảnh), bấm **Tạo ảnh**.

## Lưu ý
- **Cold start ~20–40 phút/lượt** (Kaggle tải lại FLUX ~24GB mỗi lần). Gom nhiều prompt vào 1 lượt.
- Ăn vào **quota Kaggle ~30h/tuần**.
- Generate trên T4 **chưa được kiểm chứng** — FLUX diffusers nặng, có thể cần tối ưu fp8 (giống lúc train). Nếu kernel OOM, báo để chỉnh pipeline generate.
- Token Kaggle/HF là bí mật — đừng commit `kaggle.json`.
