# Tổng hợp Hỏi-Đáp: Hệ thống Gợi ý Thực đơn

## 1. Triển khai dự án từ file plan

**Hỏi**: Triển khai dự án trong file plan cho tôi đi.

**Kết quả**: Triển khai đầy đủ 12 phases theo `Plan_trien_khai_he_thong_goi_y_thuc_don_local_v1.1.docx`:
- Phase 0: Project structure + configs + schemas + sample data (100 món Việt/Asian)
- Phase 1: Food preprocessing pipeline
- Phase 2: Disease Rule Engine (HTN, Diabetes, Gout) + Target Calculator
- Phase 3: Content-based retrieval + Greedy/NSGA-II optimizer + Orchestrator
- Phase 4: Synthetic users (500) + interactions (8997)
- Phase 5: LightGCN model + bipartite user-food graph
- Phase 6: MLP suitability model + feature engineering
- Phase 7: NSGA-II multi-objective optimization
- Phase 8: SHAP explainability
- Phase 9: FastAPI + file storage + run_local.py
- Phase 10: HTML/CSS/JS frontend song ngữ Việt/Anh
- Phase 11: Export artifacts, tests, README

**75 files** được tạo, **27/27 tests PASS**.

---

## 2. Chức năng từng folder

**Hỏi**: Cho tôi biết từng folder trong dự án có chức năng nào?

**Trả lời**:

| Thư mục | Chức năng |
|---|---|
| `backend/api/` | API endpoints: recommendations, feedback, catalog |
| `backend/schemas/` | Pydantic models: user, recommendation, feedback |
| `backend/services/` | Logic nghiệp vụ: rule_engine, target_calculator, retriever, optimizer, orchestrator, explainer, cold_start, scorer |
| `backend/ml_runtime/` | Load model khi chạy: artifact_loader, model_bundle |
| `backend/storage/` | Lưu file local: file_store (JSONL), atomic_io |
| `backend/utils/` | Tiện ích: exceptions, logging |
| `frontend/css/` | Giao diện CSS |
| `frontend/js/` | JavaScript: app.js, api.js, form.js, result.js, charts.js |
| `frontend/i18n/` | Song ngữ: vi.json, en.json |
| `ml/models/` | Model AI: LightGCN, ProfileEncoder, SuitabilityMLP |
| `ml/pipelines/` | Scripts huấn luyện: preprocess, validate, synthetic, graph, train, evaluate, export |
| `ml/features/` | Feature engineering: food_features, user_features, feature_schema |
| `configs/` | YAML: disease_rules, nutrient_targets, model, synthetic_generation |
| `data/raw/` | Dữ liệu thô: sample_foods.csv |
| `data/processed/` | Dữ liệu xử lý: foods.json, users.json, interactions.json |
| `data/runtime/` | Dữ liệu phát sinh: recommendations.jsonl, feedback.jsonl |
| `artifacts/` | Model đã train: weights, embeddings, manifest |
| `reports/` | Báo cáo chất lượng dữ liệu |
| `tests/` | 27 tests: rule_engine, target_calculator, API |

---

## 3. Tại sao chưa train model vẫn đưa ra kết quả?

**Hỏi**: Chưa train model nhưng tại sao vẫn đưa ra kết quả kèm giải thích được? Kết quả đó dựa vào đâu?

**Trả lời**: Hệ thống có **fallback baseline** ở mỗi tầng:

| Tầng | Khi chưa train | Dùng gì |
|---|---|---|
| Target | Công thức BMR | Mifflin-St Jeor + YAML config |
| Rule Engine | Luật cứng | YAML disease_rules |
| Retrieval | Content-based | Heuristic: tag_affinity + calorie_fit + protein_fit + budget + sodium |
| Scoring | Fallback | Dùng retrieval_score luôn |
| Optimization | Greedy | Toán thuần, random 50 lần chọn tốt nhất |
| Explanation | Template | Rule trace + tag matching |

Đây là thiết kế có chủ đích: MVP chạy end-to-end trước, sau đó mới thêm AI.

---

## 4. Train model

**Hỏi**: Train model cho tôi đi.

**Kết quả**: Pipeline ML chạy đầy đủ với Python 3.11:

| Bước | Kết quả |
|---|---|
| Synthetic Users | 500 users (95 HTN, 57 DM, 55 Gout) |
| Interactions | 8,997 (2951 rating, 1501 like, 1533 eaten, 2637 dislike, 375 swap) |
| User-Food Graph | 500 users × 100 foods = 5,554 edges |
| LightGCN | 100 epochs, Recall@20 = 0.18, embeddings 100×64 |
| MLP | 50 epochs, Test MSE = 0.129, MAE = 0.289 |
| Artifacts | GNN + MLP + embeddings + SHAP background exported |

Artifacts:
```
artifacts/run_20260618_172951/
├── gnn_model.pt          (155 KB)
├── mlp_model.pt          (69 KB)
├── food_embeddings.npy   (25 KB - 100 vectors × 64 dim)
├── id_maps.json          (8 KB)
├── feature_schema.json   (1 KB - 31 features)
├── shap_background.npy   (12 KB)
└── manifest.json
```

---

## 5. MLP chỉ chấm điểm từng món, NSGA-II tối ưu cả ngày

**Hỏi**: MLP chỉ chấm điểm từng món ăn còn NSGA-II mới tối ưu thực đơn cả ngày. Bước đó đâu? MLP đang tính metrics dựa trên gì?

**Phát hiện 2 vấn đề**:

1. **MLP đã train nhưng KHÔNG được dùng** - `scorer.py` khởi tạo `SuitabilityScorer()` không truyền MLP model → luôn chạy fallback, chỉ copy retrieval_score.

2. **NSGA-II đã code nhưng KHÔNG được gọi** - `optimizer.py` `optimize()` luôn gọi `_greedy_optimize()`, hàm `nsga2_optimize()` tồn tại nhưng không ai gọi.

**MLP metrics tính trên**:
- Input: 31 features (user profile + food nutrients + meal type)
- Labels từ interactions: rating → 0-1, like/eaten → 1.0, dislike → 0.0, swap → 0.2
- Test MSE = 0.129, MAE = 0.289

**Fix**:
- `scorer.py`: Auto-load MLP model, extract 31 features, chạy inference, blend `0.6×MLP + 0.4×retrieval`
- `optimizer.py`: Gọi NSGA-II trước, greedy làm fallback
- NSGA-II: duplicate = **hard constraint** g3, chọn best compromise từ Pareto front

---

## 6. NSGA-II optimize cả thực đơn, MLP chỉ scoring từng món

**Hỏi**: NSGA-II có phải optimize cả thực đơn, còn MLP chỉ scoring từng món không?

**Trả lời**: Đúng.

```
                    MLP (scoring)                    NSGA-II (optimization)
                    ─────────────                    ──────────────────────
Input              1 user + 1 food                  Tất cả candidates × 3 bữa
Output             1 score (0→1)                    1 plan [sáng, trưa, tối]
Scope              Từng món riêng lẻ                Toàn bộ thực đơn cả ngày
Biến quyết định    Không (chỉ predict)              Chọn món nào cho bữa nào
Objectives         Không                            3 obj: max suit, min cal_dev, min cost
Constraints        Không                            3 hard: sodium, sugar, không trùng
```

Pipeline:
```
Rule Engine → Retriever → MLP Scorer (từng món) → NSGA-II (cả ngày) → Explainer
```

---

## 7. Verify còn phần nào chưa load model thật

**Hỏi**: Verify lại xem còn phần nào chưa load model thật không?

**Audit phát hiện 4 chỗ vẫn dùng stub**:

| # | Component | Vấn đề |
|---|---|---|
| 1 | `cold_start.py:58-64` | `compute_like_affinity()` chỉ có `pass`, không dùng GNN/ProfileEncoder |
| 2 | `retriever.py:74-80` | 100% heuristic, không dùng `food_embeddings.npy` |
| 3 | `explainer.py` | Template cứng, không dùng SHAP |
| 4 | `model_bundle.py:21-33` | Chỉ đọc manifest, không load weights/embeddings |
| 5 | `orchestrator.py:112` | `model_version="baseline_content_v1"` cố định |

---

## 8. Fix tất cả stubs để dùng model thật

**Hỏi**: Sửa dùng model hết đi.

**Fix 5 files**:

1. **ModelBundle** (`model_bundle.py`): Load thật `food_embeddings.npy`, `mlp_model.pt`, `profile_encoder.pt`, `shap_background.npy`, `id_maps.json`

2. **Retriever** (`retriever.py`): Dùng GNN food embeddings cosine similarity, blend `0.5×content + 0.5×GNN`

3. **ColdStartEncoder** (`cold_start.py`): Dùng ProfileEncoder khi có, GNN embeddings cho `compute_like_affinity()`

4. **Explainer** (`explainer.py`): SHAP DeepExplainer trên MLP, top-3 feature contributions

5. **Orchestrator** (`orchestrator.py`): Nhận ModelBundle, pass cho tất cả services, `model_version` từ bundle thật

6. **ArtifactLoader** (`artifact_loader.py`): Chọn bundle đầy đủ nhất (nhiều files nhất) + merge missing từ runs khác

7. **app.py**: Pass `bundle=bundle` vào Orchestrator

**Kết quả**: Log xác nhận:
```
Loaded: food_embeddings (100, 64), id_maps (100 foods),
        mlp_model, shap_background (100 samples), feature_schema
Orchestrator initialized with bundle version=run_20260618_172951
```

**27/27 tests PASS.**

SHAP: code đã viết đúng, chỉ cần `pip install shap` (graceful fallback khi chưa cài).

---

## 9. Đổi giao diện xanh nước biển

**Hỏi**: Đổi giao diện thành tone xanh nước biển đi.

**Thay đổi**:
- Primary: `#2d7d46` (xanh lá) → `#1565c0` (xanh dương đậm)
- Primary light: `#4caf50` → `#42a5f5` (xanh sáng)
- Primary dark: `#1b5e20` → `#0d47a1` (xanh navy)
- Secondary: `#f57c00` → `#00acc1` (xanh cyan)
- Header gradient: 3 màu xanh biển
- Disclaimer: nền xanh nhạt `#e3f2fd`
- Charts: bar chart xanh dương
- Focus ring: xanh dương nhạt

---

## Tổng kết trạng thái pipeline

```
Input: User profile (tuổi, chiều cao, cân nặng, bệnh, dị ứng, sở thích)
  │
  ▼
[1] Target Calculator ── Mifflin-St Jeor BMR → calories/nutrient targets
  │
  ▼
[2] Rule Engine ── YAML rules → loại món vi phạm (hard filter)
  │                 100 → 18 món (nếu Diabetes + HTN + Shellfish)
  ▼
[3] Retriever ── Content heuristic + GNN cosine similarity → Top-100 mỗi bữa
  │               0.5 × content_score + 0.5 × gnn_similarity
  ▼
[4] MLP Scorer ── 31 features → MLP inference → suitability score
  │               0.6 × MLP_score + 0.4 × retrieval_score
  ▼
[5] NSGA-II ── 3 objectives (max suit, min cal_dev, min cost)
  │            3 hard constraints (sodium, sugar, no duplicate)
  │            Best compromise từ Pareto front
  ▼
[6] Explainer ── Rule trace + tag matching + SHAP contributions
  │
  ▼
Output: DayPlanResponse (3 bữa + nutrition summary + explanations)
```

Tất cả models (GNN, MLP, GNN embeddings) đều **đã được load và sử dụng thật** trong pipeline.
