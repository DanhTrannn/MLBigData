# Meal Recommender - Hệ thống Gợi ý Thực đơn

Hệ thống gợi ý thực đơn 1 ngày (sáng/trưa/tối) cho người bệnh và người bình thường, chạy hoàn toàn trên máy local.

## Tính năng

- Gợi ý thực đơn cá nhân hóa theo hồ sơ sức khỏe
- Hỗ trợ 3 bệnh lý: Tăng huyết áp, Tiểu đường, Gout
- Rule Engine lọc cứng dị ứng và an toàn thực phẩm
- Tối ưu đa mục tiêu (NSGA-II) cho năng lượng, dinh dưỡng, chi phí
- Giải thích lý do đề xuất
- Giao diện song ngữ Việt/Anh
- Không dùng database - lưu file JSON/JSONL local

## Cài đặt

```bash
# Tạo virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Cài dependencies
pip install -r requirements.txt
```

## Chạy ứng dụng

### Chạy nhanh (MVP end-to-end)
```bash
python run_local.py
```
Ứng dụng sẽ mở tại http://127.0.0.1:8000

### Chạy từng bước

```bash
# 1. Preprocess dữ liệu
python -m ml.pipelines.preprocess_foods

# 2. Validate dữ liệu
python -m ml.pipelines.validate_data

# 3. Khởi động server
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

### Pipeline ML (khi cần train model)

```bash
# Sinh dữ liệu giả lập
python -m ml.pipelines.generate_synthetic_users
python -m ml.pipelines.generate_interactions

# Build graph và train GNN
python -m ml.pipelines.build_graph
python -m ml.pipelines.train_gnn

# Build dataset và train MLP
python -m ml.pipelines.build_mlp_dataset
python -m ml.pipelines.train_mlp

# Đánh giá
python -m ml.pipelines.evaluate

# Export artifacts
python -m ml.pipelines.export_artifacts
```

## Kiểm thử

```bash
pytest tests/ -v
```

## Cấu trúc dự án

```
meal-recommender/
├── app.py                    # FastAPI entry point
├── backend/
│   ├── api/                  # API endpoints
│   ├── schemas/              # Pydantic models
│   ├── services/             # Business logic
│   │   ├── rule_engine.py    # Disease & allergy rules
│   │   ├── target_calculator.py  # Nutrient targets
│   │   ├── retriever.py      # Content-based retrieval
│   │   ├── optimizer.py      # Greedy + NSGA-II
│   │   ├── explainer.py      # Explanations
│   │   └── orchestrator.py   # Pipeline coordinator
│   ├── storage/              # JSONL file storage
│   └── ml_runtime/           # Model loading
├── frontend/                 # HTML/CSS/JS
├── ml/
│   ├── pipelines/            # Training scripts
│   ├── models/               # LightGCN, MLP
│   └── features/             # Feature engineering
├── configs/                  # YAML configurations
├── data/                     # Raw, processed, runtime
├── artifacts/                # Trained models
└── tests/                    # Test suite
```

## API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/health` | Liveness check |
| GET | `/ready` | Readiness check |
| GET | `/api/v1/catalog/diseases` | Danh sách bệnh hỗ trợ |
| GET | `/api/v1/catalog/allergens` | Danh sách dị ứng |
| POST | `/api/v1/recommendations/day` | Tạo thực đơn 1 ngày |
| POST | `/api/v1/recommendations/{id}/swap` | Đổi món |
| POST | `/api/v1/feedback` | Gửi phản hồi |

## Cấu hình

- `configs/disease_rules.yaml` - Quy tắc bệnh lý và dị ứng
- `configs/nutrient_targets.yaml` - Mục tiêu dinh dưỡng
- `configs/model.yaml` - Cấu hình ML models
- `configs/synthetic_generation.yaml` - Cấu hình sinh dữ liệu giả

## Giới hạn và Lưu ý an toàn

- Hệ thống **không thay thế** chẩn đoán hoặc chỉ định điều trị của chuyên gia y tế
- Các ngưỡng dinh dưỡng/bệnh lý cần được cấu hình và phê duyệt bởi chuyên gia y tế
- Dữ liệu giả lập chỉ dùng cho mục đích phát triển và demo
- Không tương tác thuốc-thực phẩm
- Không hỗ trợ thực đơn nhiều tuần hoặc quản lý tồn kho

## Công nghệ

- **Backend**: FastAPI, Pydantic, uvicorn
- **Data**: pandas, numpy, pyarrow, PyYAML
- **ML**: PyTorch, PyTorch Geometric, scikit-learn
- **Optimization**: pymoo (NSGA-II)
- **Explainability**: SHAP
- **Frontend**: HTML5, CSS3, JavaScript ES6+, Chart.js
- **Testing**: pytest, httpx
- **Storage**: JSON/JSONL (no database)
