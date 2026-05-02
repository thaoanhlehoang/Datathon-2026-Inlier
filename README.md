# Datathon-2026-Inlier

## 1. Cấu trúc thư mục repo

Repo này được sắp xếp theo mục đích sử dụng của từng nhóm file:

```text
Datathon-2026-Inlier/
├── data/
│   └── raw/                  # Dữ liệu gốc của cuộc thi: CSV đầu vào và sample_submission.csv
├── src/
│   └── modeling/             # Code chính để train model, forecast, calibration, CV và SHAP
├── notebooks/
│   ├── p1_mcq/               # Notebook giải câu hỏi trắc nghiệm vòng 1
│   ├── p2_eda/basic/         # Notebook EDA cơ bản
│   ├── eda/ai_generated/     # Notebook EDA chi tiết được tạo trong quá trình phân tích
│   ├── baseline/             # Notebook baseline
│   └── storytelling/         # Notebook phục vụ kể chuyện / trình bày insight
├── reports/
│   ├── visual_generators/    # Script/notebook tạo biểu đồ cho báo cáo
│   ├── visuals/              # Ảnh biểu đồ PNG/SVG đã tạo
│   ├── metrics/              # CSV metric dùng cho biểu đồ/báo cáo
│   └── figures/              # Hình phụ trợ cho báo cáo
├── outputs/
│   └── model/
│       ├── submissions/      # File submission đầu ra
│       ├── forecasts/        # Hình/kết quả forecast
│       └── explainability/   # Kết quả giải thích model, ví dụ SHAP
├── docs/
│   └── problem_statement/    # Đề bài / tài liệu mô tả cuộc thi
└── artifacts/
    └── cache/                # Cache và file tạm được sinh ra khi chạy script
```

## 2. Hướng dẫn chạy lại kết quả và tạo `submission.csv`

Chạy các lệnh sau từ thư mục gốc của repo:

```powershell
cd "D:\Code\Datathon 2026\Datathon-2026-Inlier"
python -m pip install -r requirements.txt
python src\modeling\our_method_forecast.py
Copy-Item outputs\model\submissions\submission_our_method.csv outputs\model\submissions\submission.csv -Force
```

Sau khi chạy xong, file nộp bài nằm tại:

```text
outputs/model/submissions/submission.csv
```
