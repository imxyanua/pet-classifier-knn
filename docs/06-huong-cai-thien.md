# 6. Hướng cải thiện

Các hướng dưới đây dựa trên điểm yếu **đã đo được** ở [05-danh-gia.md](05-danh-gia.md), sắp theo thứ tự nên làm. Thứ tự này ưu tiên: (1) có thước đo đáng tin trước, (2) thay đổi nhỏ mà chắc chắn có lợi, (3) thay đổi lớn hơn sau.

## Tóm tắt

| Ưu tiên | Hướng | Trạng thái | Lợi ích |
|---:|---|---|---|
| 1 | Đánh giá đúng cách: chọn $k$ bằng cross-validation, báo cáo trung bình nhiều seed | **Đã làm** | Con số không thiên lệch: 59.3% ± 3.6 |
| 2 | Bỏ Hu moments | **Đã làm** | Vector gọn hơn; accuracy không đổi trong ngưỡng dao động |
| 3 | Sửa đặc trưng màu: H dạng vòng tròn, lọc pixel tối/nhạt, thêm Lab | Chưa | Chưa đo; tác động trực tiếp lên nhóm đặc trưng mạnh nhất |
| 4 | Đo và cải thiện phân đoạn bằng trimap của Oxford | **Đã làm phần lớn** | IoU 0.655 → 0.716; accuracy 59.3% → 61.1% |
| 5 | Cải tiến k-NN: vote theo khoảng cách, khoảng cách chi-square cho histogram | Chưa | Nhỏ đến vừa |
| 6 | Kỹ thuật: test, tham số dòng lệnh, xuất kết quả tự động | Một phần | Dễ thử nghiệm, dễ viết báo cáo |

Con số ở mục 1 và 2 là kết quả tại thời điểm làm, trước khi sửa phân đoạn ở mục 4.

Với độ dao động khoảng ±3 điểm giữa các cách chia, mọi cải tiến cần tăng **rõ hơn khoảng 3 điểm** trung bình 10 seed trên test, hoặc tăng nhất quán accuracy cross-validation ở mọi seed, thì mới coi là thật sự có lợi.

---

## 1. Đánh giá đúng cách — đã làm

**Vấn đề cũ.** Phiên bản đầu chọn cặp (khoảng cách, $k$) tốt nhất **dựa trên tập test**, rồi báo cáo accuracy trên chính tập test đó. Đây là rò rỉ dữ liệu: tập test đã được dùng để chọn mô hình nên không còn là dữ liệu "chưa thấy". Ngoài ra chỉ có một seed, trong khi kết quả dao động vài điểm giữa các cách chia.

**Đã thay bằng:**

- `knn/validation.py`: `stratified_k_fold`, `cross_val_accuracy` (chuẩn hóa lại trong từng fold), `select_params`.
- `knn/evaluate.py`: chọn tham số bằng CV 5 fold trên train, đánh giá test một lần, lặp 10 seed; các tham số `--seed`, `--folds`, `--repeats`, `--features`.
- `per_class_metrics`: precision, recall, F1 theo lớp.
- Dải $k$ mở rộng tới 25, vì với dải cũ (tới 11) CV thường chọn đúng giá trị lớn nhất, dấu hiệu $k$ tốt nhất nằm ngoài dải. Thử thêm tới $k=61$ cho thấy đường CV đạt đỉnh quanh $k=25$ rồi giảm.

**Kết quả:** 59.3% ± 3.6 (10 seed). Chi tiết ở [05-danh-gia.md](05-danh-gia.md).

**Phát hiện phụ:** cách đánh giá cũ đã che mất đóng góp của 5 đặc trưng hình học. Ở $k=5$ cố định chúng không giúp gì; khi CV được chọn $k$ lớn hơn, chúng thêm khoảng 3.7 điểm.

## 2. Bỏ Hu moments — đã làm

**Bằng chứng (theo quy trình CV, 10 seed):** Hu moments một mình đạt 25.9%, gần mức đoán bừa. Màu + hình học + Hu đạt 58.8% ± 2.7; bỏ Hu đạt 59.3% ± 3.6.

**Đã làm:** Hu vẫn được trích và lưu trong cache, nhưng `DEFAULT_FEATURE_GROUPS` trong `config.py` chỉ gồm `color` và `geometry`; cả `evaluate.py` và app đều dùng mặc định này. Vẫn thêm lại được bằng `--features color,geometry,hu` để so sánh.

**Lưu ý khi trình bày:** chênh lệch 0.5 điểm nằm trong độ dao động. Lý do bỏ Hu là chúng **không mang thông tin hữu ích** (gần mức đoán bừa khi dùng riêng), không phải vì bỏ đi làm tăng accuracy.

**Còn có thể thử:** đánh trọng số theo nhóm đặc trưng (ví dụ nhân nhóm hình học với hệ số $w$ sau chuẩn hóa) và chọn $w$ bằng CV.

## 3. Cải thiện đặc trưng màu

Màu là nhóm mang phần lớn thông tin (một mình đạt 58.6%), nên cải thiện ở đây có tác động trực tiếp nhất.

### 3a. Xử lý H như đại lượng vòng tròn

H = 0 và H = 179 đều là đỏ nhưng trung bình số học là 90. Thay $\mu_H$ bằng hai giá trị:

$$\bar{c} = \frac{1}{|P|}\sum \cos\left(\frac{2\pi H}{180}\right), \quad \bar{s} = \frac{1}{|P|}\sum \sin\left(\frac{2\pi H}{180}\right)$$

Độ dài vector $\sqrt{\bar{c}^2 + \bar{s}^2}$ còn cho biết màu có tập trung hay không, dùng thay cho $\sigma_H$.

Ảnh hưởng nhiều nhất tới Sphynx (da hồng nằm đúng vùng H quanh 0/179).

### 3b. Chỉ tính H trên pixel có màu thật

Pixel có S thấp (xám, trắng) hoặc V thấp (đen) có H gần như ngẫu nhiên. Chỉ đưa vào histogram H những pixel có, ví dụ, S > 40 và V > 40. Thêm một đặc trưng **tỷ lệ pixel "không màu"** (S hoặc V thấp): đặc trưng này tự nó đã phân biệt tốt Bombay (gần như toàn đen), Samoyed và Egyptian Mau (phần lớn không màu) với Pug và Sphynx (có màu).

### 3c. Thêm không gian Lab

Trong Lab, L là độ sáng, a là trục xanh lá–đỏ, b là trục xanh dương–vàng. Khoảng cách trong Lab gần với cảm nhận màu của mắt người hơn HSV. Trung bình và độ lệch chuẩn của a, b (4 chiều) phân biệt tốt tông kem (Pug, b dương) với tông hồng (Sphynx, a dương).

### 3d. Histogram 2D H–S

Thay vì hai histogram 1D riêng cho H và S, dùng histogram 2D (ví dụ 8×3 = 24 bin) để giữ được tương quan "màu gì và đậm cỡ nào". Cẩn thận với số chiều: đánh giá lại bằng cross-validation, vì thêm chiều có thể làm k-NN kém đi.

### 3e. (Ngoài phạm vi đề) Đặc trưng kết cấu

Cặp nhầm lẫn lớn nhất là Egyptian Mau ↔ Sphynx, và Mau còn hay bị nhầm với Samoyed. Đặc điểm phân biệt của Mau là **đốm**, tức là kết cấu chứ không phải màu hay hình dạng. Một histogram LBP (Local Binary Pattern) hoặc thống kê từ ma trận đồng xuất hiện mức xám (GLCM) sẽ mô tả được đốm. Đề chỉ yêu cầu màu và hình dạng, nên nếu thêm thì trình bày như một **thí nghiệm mở rộng**, so sánh có và không có kết cấu.

## 4. Đo và cải thiện phân đoạn — đã làm phần lớn

Chi tiết ở [02-tien-xu-ly-va-phan-doan.md](02-tien-xu-ly-va-phan-doan.md).

### 4a. Đo chất lượng mask bằng trimap — đã làm

`python download_data.py --trimaps` tải trimap của Oxford; `python evaluate_segmentation.py` tính IoU, precision, recall của mask GrabCut theo từng giống. Trimap chỉ dùng để chấm điểm, không dùng để trích đặc trưng.

Kết quả với cách khởi tạo cũ: IoU 0.655, 27.4% số ảnh sai nặng (IoU < 0.5), recall 0.75 (mask thường thiếu phần thân).

### 4b. Cải thiện GrabCut — đã làm

| Đã thử | Kết quả |
|---|---|
| Tăng số vòng lặp 1 → 5 | Giúp một chút khi đi kèm khởi tạo tốt hơn |
| Thu hẹp vùng nền chắc chắn còn 1% sát mép, thêm dải "có thể là nền" 5% | **Lợi ích chính**: recall 0.75 → 0.85 |
| Elip "chắc chắn là con vật" ở tâm (20%) | Giúp thêm; elip 30–40% kém hơn |
| Chạy lại khi mask phủ > 75% ảnh | Không cải thiện đáng kể |
| Tự động phát hiện và cắt khung viền | Không cải thiện đáng kể |
| GrabCut trên ảnh Lab thay vì BGR | Không cải thiện đáng kể |

Kết quả: IoU 0.716, 22.1% ảnh sai nặng; accuracy nhận dạng 59.3% → 61.1%, accuracy cross-validation tăng ở cả 10/10 seed.

### 4c. Còn lại

- **Thí nghiệm "oracle":** trích đặc trưng bằng trimap thay vì GrabCut rồi chạy lại k-NN, để biết **trần** accuracy nếu phân đoạn hoàn hảo. Đây chỉ là phân tích phụ (dùng mask người vẽ thay cho bước phân đoạn của đề), nhưng cho biết còn bao nhiêu lỗi nằm ở phân đoạn so với ở đặc trưng.
- **Nền đồng nhất:** 149 ảnh tệ đi khi đổi cách khởi tạo, chủ yếu ảnh có nền một màu (tuyết, phông tối) khiến mask tràn ra cả ảnh. Có thể kết hợp hai cách khởi tạo và chọn mask "hợp lý" hơn theo một tiêu chí không cần trimap (ví dụ diện tích mask nằm trong khoảng 10–80% ảnh).
- **Giới hạn của GrabCut:** khi con vật cùng màu với nền (thân Pug màu kem trên tường trắng), không cách khởi tạo nào sửa được, vì GrabCut chỉ dựa vào màu.

## 5. Cải tiến k-NN

Các thay đổi nhỏ trong `knn/knn.py`, phù hợp với yêu cầu "khảo sát ảnh hưởng của khoảng cách":

- **Vote theo khoảng cách:** mỗi láng giềng góp phiếu $1/(d + \varepsilon)$ thay vì 1. Láng giềng gần có tiếng nói lớn hơn, giảm ảnh hưởng của láng giềng xa khi $k$ lớn.
- **Khoảng cách chi-square** cho phần histogram:

  $$\chi^2(h, h') = \frac{1}{2}\sum_i \frac{(h_i - h'_i)^2}{h_i + h'_i + \varepsilon}$$

  Phù hợp với histogram hơn L1/L2 vì nó coi chênh lệch ở bin nhỏ là quan trọng hơn. Lưu ý chi-square tính trên histogram **chưa** chuẩn hóa z-score.
- **Khoảng cách cosine:** chỉ quan tâm hướng của vector, bỏ qua độ lớn. Thêm vào bảng khảo sát để so sánh.
- ~~**Khảo sát $k$ rộng hơn**~~ — đã làm cùng mục 1: dải $k$ hiện tới 25.

## 6. Cải tiến kỹ thuật

- ~~**Unit test**~~ — đã làm: `tests/` kiểm tra k-NN, phá hòa, chuẩn hóa, chia tập, cross-validation, metric và chọn nhóm đặc trưng (`python -m pytest`).
- ~~**Tham số dòng lệnh**~~ — đã làm: `--seed`, `--folds`, `--repeats`, `--features`.
- **Xuất kết quả tự động**: `evaluate.py` ghi bảng accuracy ra JSON và vẽ lại `docs/accuracy_k.png`, `docs/confusion_matrix.png`, để README luôn khớp với code.
- **Vector hóa đánh giá:** tính cả ma trận khoảng cách test × train trong một lệnh thay vì lặp từng ảnh. Không cần với 1.000 ảnh, nhưng hữu ích nếu mở rộng lên cả 37 giống (khoảng 7.400 ảnh).
- **Hiển thị trong app** các ảnh láng giềng gần nhất (không chỉ tên và khoảng cách), để thấy trực quan vì sao k-NN dự đoán như vậy. Rất có ích khi thuyết trình.

## Gợi ý lộ trình

Nếu thời gian có hạn, thứ tự sau cho nhiều giá trị nhất cho báo cáo:

1. ~~Mục 1 (cross-validation, nhiều seed)~~ — đã làm.
2. ~~Mục 2 (bỏ Hu)~~ — đã làm; bảng ablation có trong [05-danh-gia.md](05-danh-gia.md), nên đưa vào báo cáo.
3. ~~Mục 4a, 4b (đo IoU, cải thiện khởi tạo GrabCut)~~ — đã làm; bảng so sánh các cách khởi tạo nên đưa vào phần phân đoạn của báo cáo.
4. Mục 3a và 3b (sửa kênh H) — cải tiến nhỏ, có cơ sở lý thuyết rõ.
5. Mục 5 (vote theo khoảng cách, chi-square) — mở rộng phần khảo sát khoảng cách.
