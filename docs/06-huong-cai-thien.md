# 6. Hướng cải thiện

Các hướng dưới đây dựa trên điểm yếu **đã đo được** ở [05-danh-gia.md](05-danh-gia.md), sắp theo thứ tự nên làm. Thứ tự này ưu tiên: (1) có thước đo đáng tin trước, (2) thay đổi nhỏ mà chắc chắn có lợi, (3) thay đổi lớn hơn sau.

## Tóm tắt

| Ưu tiên | Hướng | Trạng thái | Lợi ích |
|---:|---|---|---|
| 1 | Đánh giá đúng cách: chọn $k$ bằng cross-validation, báo cáo trung bình nhiều seed | **Đã làm** | Con số không thiên lệch: 59.3% ± 3.6 |
| 2 | Bỏ Hu moments | **Đã làm** | Vector gọn hơn; accuracy không đổi trong ngưỡng dao động |
| 3 | Sửa đặc trưng màu: H dạng vòng tròn, lọc pixel tối/nhạt, thêm Lab | Chưa | Chưa đo; tác động trực tiếp lên nhóm đặc trưng mạnh nhất |
| 4 | Đo và cải thiện phân đoạn bằng trimap của Oxford | Chưa | Biết được trần cải thiện; có thể lớn nhất |
| 5 | Cải tiến k-NN: vote theo khoảng cách, khoảng cách chi-square cho histogram | Chưa | Nhỏ đến vừa |
| 6 | Kỹ thuật: test, tham số dòng lệnh, xuất kết quả tự động | Một phần | Dễ thử nghiệm, dễ viết báo cáo |

Với độ dao động ±3.6 điểm giữa các cách chia, mọi cải tiến ở các mục 3–5 cần tăng **rõ hơn khoảng 3–4 điểm** trung bình 10 seed thì mới coi là thật sự có lợi.

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

Màu là nhóm mang phần lớn thông tin (một mình đạt 55.6%), nên cải thiện ở đây có tác động trực tiếp nhất.

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

## 4. Đo và cải thiện phân đoạn

**Vấn đề.** Hình ở [02-tien-xu-ly-va-phan-doan.md](02-tien-xu-ly-va-phan-doan.md) cho thấy GrabCut chỉ lấy mặt Pug và lấy dính sàn ở Sphynx. Hiện chưa có con số nào cho biết phân đoạn sai bao nhiêu và làm accuracy giảm bao nhiêu.

### 4a. Dùng trimap của Oxford làm chuẩn

Bộ Oxford-IIIT Pet có file `annotations.tar.gz` chứa **trimap** cho từng ảnh: mỗi pixel được gán tiền cảnh, nền, hoặc viền. Đây là mask chuẩn do người gán nhãn.

Hai thí nghiệm nên làm:

1. **Đo chất lượng mask:** tính IoU (Intersection over Union) giữa mask GrabCut và trimap cho từng ảnh, rồi xem IoU trung bình theo từng giống. Dự đoán: Pug và Sphynx thấp nhất.
2. **Thí nghiệm "oracle":** trích đặc trưng bằng trimap thay vì GrabCut, rồi chạy lại k-NN. Accuracy thu được là **trần** của phương pháp nếu phân đoạn hoàn hảo.
   - Nếu trần cao hơn hẳn (ví dụ 70%), thì cải thiện phân đoạn là hướng đáng đầu tư nhất.
   - Nếu trần chỉ cao hơn vài điểm, thì giới hạn nằm ở đặc trưng, nên tập trung vào mục 3.

Thí nghiệm này rất có giá trị cho báo cáo vì nó tách được hai nguồn lỗi: lỗi phân đoạn và lỗi đặc trưng.

### 4b. Cải thiện GrabCut

- Tăng số vòng lặp từ 1 lên 3–5. Chậm hơn khi trích đặc trưng nhưng chỉ phải chạy một lần nhờ cache.
- Khởi tạo tốt hơn hình chữ nhật cố định 8%: ví dụ đánh dấu một vùng nhỏ ở tâm ảnh là "chắc chắn tiền cảnh" (`GC_FGD`) để GrabCut không bỏ mất thân con vật có màu gần nền.
- Phát hiện mask bất thường: nếu mask chiếm < 10% hoặc > 90% ảnh, coi là phân đoạn thất bại và dùng chiến lược dự phòng.

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
3. Mục 4a (IoU và thí nghiệm oracle với trimap) — trả lời được "lỗi nằm ở đâu".
4. Mục 3a và 3b (sửa kênh H) — cải tiến nhỏ, có cơ sở lý thuyết rõ.
5. Mục 5 (vote theo khoảng cách, chi-square) — mở rộng phần khảo sát khoảng cách.
