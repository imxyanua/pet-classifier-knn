# Changelog

Nhật ký quá trình làm dự án, mục mới nhất ở trên. Mỗi mục ứng với một pull request hoặc một đợt commit.

Trừ khi ghi khác, accuracy là accuracy trên tập test, trung bình ± độ lệch chuẩn qua 10 cách chia train/test (seed 0–9), với $k$ và khoảng cách được chọn bằng cross-validation trên tập train. IoU là chất lượng mask phân đoạn so với trimap của bộ Oxford.

## Tiến triển kết quả

| Mốc | Accuracy | IoU phân đoạn | Ghi chú |
|---|---:|---:|---|
| Phiên bản đầu (18/11) | 58.0% | 0.655 | $k$ chọn trên chính tập test, một seed: con số lạc quan, không so sánh trực tiếp được |
| PR #1 (18/11) | 59.3% ± 3.6 | 0.655 | Cách đánh giá đúng đầu tiên |
| PR #2 (22/11) | **61.1% ± 2.8** | **0.716** | Đổi cách khởi tạo GrabCut |
| PR #3, #5 (23/11) | 61.1% ± 2.8 | 0.716 | Chỉ thêm công cụ đo và ghi lại thử nghiệm |
| Oracle (tham khảo) | 72.5% ± 2.0 | 1.0 | Đặc trưng tính trên trimap, tức phân đoạn hoàn hảo; không thuộc pipeline |

## [Chưa phát hành]

## 2026-11-23 — Thử nghiệm cải thiện phân đoạn

PR [#3](https://github.com/imxyanua/pet-classifier-knn/pull/3), [#5](https://github.com/imxyanua/pet-classifier-knn/pull/5) (thay cho [#4](https://github.com/imxyanua/pet-classifier-knn/pull/4), bị GitHub tự đóng khi nhánh nền bị xóa).

Mục tiêu: nhiều ảnh chỉ phân đoạn được mặt con vật, hoặc lẫn nền vào con vật.

**Thêm**
- `evaluate_segmentation.py` xếp ảnh phân đoạn sai nặng thành *thiếu thân* (recall < 0.6), *dính nền* (precision < 0.6) hoặc *cả hai*.
- Tùy chọn `--split train|test` để chỉ tinh chỉnh trên tập train.
- Thí nghiệm oracle: phân đoạn hoàn hảo cho accuracy 72.5%, cao hơn hiện tại khoảng 11 điểm.

**Đã thử, không giữ** (không biến thể nào tăng accuracy vượt mức dao động)
- 256/320px, GrabCut hai lượt: thiếu thân 75 → 39 ảnh nhưng dính nền 67 → 83, accuracy cross-validation giảm ở 10/10 seed.
- Khởi tạo bằng saliency: dính nền 67 → 31 ảnh nhưng thiếu thân tăng gấp 2–2.5 lần.
- Giữ nhiều vùng, kênh kết cấu, giữ vùng chứa tâm, phép mở lớn hơn, tính màu trên lõi mask, tự chọn mask theo khác biệt màu.

**Kết luận:** lấy thêm thân thường kéo theo dính nền, và ngược lại. Phân đoạn vẫn là nguồn lỗi lớn nhất nhưng cần một phương pháp khác hẳn để thu hẹp khoảng cách tới oracle.

## 2026-11-22 — Cải thiện phân đoạn GrabCut

PR [#2](https://github.com/imxyanua/pet-classifier-knn/pull/2).

**Thay đổi**
- GrabCut khởi tạo bằng mask thay vì hình chữ nhật cách mép 8%: 1% sát mép là nền chắc chắn, dải 5% là "có thể là nền", elip 20% ở tâm là "chắc chắn là con vật"; 5 vòng thay vì 1; cố định seed để kết quả lặp lại được.
- Trích đặc trưng chạy song song trên nhiều nhân CPU (1.000 ảnh khoảng 45 giây).
- Biểu đồ trong app: số trên thanh màu, vạch chia, viền khung chuyển sang màu trắng để đọc được trên nền tối.

**Thêm**
- `download_data.py --trimaps` và `evaluate_segmentation.py`: chấm IoU, precision, recall của mask so với trimap. Trimap chỉ dùng để chấm điểm, không dùng để trích đặc trưng.

**Kết quả:** IoU 0.655 → 0.716, ảnh sai nặng 27.4% → 22.1%, accuracy 59.3% → 61.1%, accuracy cross-validation tăng ở 10/10 seed.

**Đã thử, không giữ:** chạy lại GrabCut khi mask quá lớn, tự cắt khung viền ảnh, GrabCut trên không gian màu Lab.

## 2026-11-18 — Đánh giá bằng cross-validation

PR [#1](https://github.com/imxyanua/pet-classifier-knn/pull/1).

**Sửa**
- Trước đây $k$ được chọn trên chính tập test rồi báo cáo trên tập test đó (rò rỉ dữ liệu). Giờ $k$ và khoảng cách được chọn bằng cross-validation 5 fold chỉ trên tập train, tập test chỉ dùng một lần, lặp 10 seed.

**Thêm**
- `knn/validation.py`: k-fold phân tầng, cross-validation, chọn tham số.
- Precision, recall, F1 theo từng lớp.
- `evaluate.py --seed --folds --repeats --features`.
- 15 unit test cho phần k-NN tự cài.

**Thay đổi**
- Dải $k$ mở rộng từ 1–11 lên 1–25, vì cross-validation hay chọn đúng giá trị lớn nhất của dải cũ.
- Bỏ Hu moments khỏi bộ đặc trưng mặc định (một mình chỉ gần mức đoán bừa); vẫn được cache để so sánh.

**Kết quả:** 59.3% ± 3.6. Sửa luôn một kết luận sai của phiên bản đầu: 5 đặc trưng hình học *có* giúp khi $k$ được chọn đúng cách.

## 2026-11-18 — Tài liệu

- Thư mục `docs/` gồm 6 phần: tổng quan, phân đoạn, đặc trưng, k-NN, đánh giá, hướng cải thiện.
- README viết lại theo pipeline và kết quả; công thức đổi sang cú pháp `$...$` để hiển thị được trên GitHub.

## 2026-11-18 — Pipeline đầu tiên

- Tiền xử lý (resize 192px, Gaussian blur) và phân đoạn bằng GrabCut.
- Đặc trưng 34 chiều: màu HSV (thống kê + histogram) và hình dạng (5 đặc trưng hình học + 7 Hu moments).
- k-NN tự cài bằng NumPy (Euclidean, Manhattan), chuẩn hóa z-score, chia tập phân tầng, accuracy, ma trận nhầm lẫn. Không dùng scikit-learn.
- Cache đặc trưng vào `features_cache.npz`.
- `evaluate.py` khảo sát $k$ và khoảng cách.
- Desktop app CustomTkinter: nhận dạng từng ảnh, xem k láng giềng, biểu đồ đánh giá.
- Code chia thành các package `features/`, `knn/`, `dataset/`, `gui/`.

**Kết quả:** 58.0% (Manhattan, $k=5$), con số lạc quan vì $k$ được chọn trên tập test.

## 2026-11-17 — Dữ liệu

- Bộ Oxford-IIIT Pet: `raw_data/images.tar.gz` (lưu bằng Git LFS), ảnh giải nén, file torrent dự phòng.
- `download_data.py` tải và lọc 5 giống: Pug, Samoyed, Bombay, Egyptian Mau, Sphynx.
- `data/`: 200 ảnh mỗi giống, tổng 1.000 ảnh.
