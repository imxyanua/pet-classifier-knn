# 6. Hướng cải thiện

Các hướng dưới đây dựa trên điểm yếu **đã đo được** ở [05-danh-gia.md](05-danh-gia.md), sắp theo thứ tự nên làm. Thứ tự này ưu tiên: (1) có thước đo đáng tin trước, (2) thay đổi nhỏ mà chắc chắn có lợi, (3) thay đổi lớn hơn sau.

## Tóm tắt

| Ưu tiên | Hướng | Công sức | Lợi ích dự kiến |
|---:|---|---|---|
| 1 | Đánh giá đúng cách: chọn $k$ bằng cross-validation, báo cáo trung bình nhiều seed | Nhỏ | Con số đáng tin, so sánh cải tiến được |
| 2 | Bỏ Hu moments, giảm trọng số hình dạng | Rất nhỏ | Khoảng +1–1.5 điểm (đã đo) |
| 3 | Sửa đặc trưng màu: H dạng vòng tròn, lọc pixel tối/nhạt, thêm Lab | Vừa | Chưa đo; tác động trực tiếp lên nhóm đặc trưng mạnh nhất |
| 4 | Đo và cải thiện phân đoạn bằng trimap của Oxford | Vừa–lớn | Biết được trần cải thiện; có thể lớn nhất |
| 5 | Cải tiến k-NN: vote theo khoảng cách, khoảng cách chi-square cho histogram | Nhỏ | Nhỏ đến vừa |
| 6 | Kỹ thuật: test, tham số dòng lệnh, xuất kết quả tự động | Nhỏ | Dễ thử nghiệm, dễ viết báo cáo |

---

## 1. Đánh giá đúng cách

**Vấn đề.** `knn/evaluate.py` chọn cặp (khoảng cách, $k$) tốt nhất **dựa trên tập test**, rồi báo cáo accuracy trên chính tập test đó. Đây là rò rỉ dữ liệu: tập test đã được dùng để chọn mô hình nên không còn là dữ liệu "chưa thấy". Thêm vào đó, kết quả với một seed dao động ±2–3 điểm (từ 51% đến 59.5%), lớn hơn nhiều cải tiến nhỏ muốn so sánh.

**Cách làm.**

1. Viết hàm `k_fold_indices(y, n_folds=5, seed)` chia **tập train** thành 5 phần phân tầng.
2. Với mỗi cặp (khoảng cách, $k$): train trên 4 phần, đánh giá trên phần còn lại, lặp 5 lần, lấy trung bình. Nhớ tính mean/std chuẩn hóa **trên 4 phần train của từng fold**.
3. Chọn cặp có accuracy cross-validation cao nhất.
4. Chỉ khi đó mới chạy một lần duy nhất trên tập test và báo cáo con số này.
5. Lặp toàn bộ quy trình với nhiều seed, báo cáo trung bình ± độ lệch chuẩn.

**Tại sao làm trước.** Không có bước này thì không phân biệt được một cải tiến thật với dao động ngẫu nhiên do cách chia tập.

Nên báo cáo thêm **precision, recall, F1 theo từng lớp** (tính được trực tiếp từ ma trận nhầm lẫn), vì accuracy chung che mất việc Egyptian Mau chỉ đạt 45%.

## 2. Bỏ Hu moments và giảm trọng số hình dạng

**Bằng chứng.** Hu moments một mình chỉ đạt 26.8%, gần mức đoán bừa. Bỏ chúng tăng accuracy từ 55.5% lên 57.0%. Giảm trọng số cả nhóm hình dạng xuống 0.25 cho 56.9%.

**Cách làm.**

- Đơn giản nhất: không đưa 7 chiều Hu vào vector (sửa `_shape_features` trong `features/features.py`), tăng `CACHE_VERSION`.
- Hoặc giữ lại nhưng thêm vector trọng số theo nhóm, nhân vào sau bước chuẩn hóa. Trọng số chọn bằng cross-validation ở mục 1.

**Đưa vào báo cáo.** Bảng ablation (chỉ màu / chỉ hình dạng / cả hai) là một phân tích có giá trị, trả lời câu hỏi "đặc trưng nào thực sự có ích", và phù hợp với yêu cầu khảo sát của đề.

## 3. Cải thiện đặc trưng màu

Màu là nhóm mang gần như toàn bộ thông tin, nên cải thiện ở đây có tác động trực tiếp nhất.

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

Egyptian Mau bị nhầm nhiều nhất vì đặc điểm phân biệt của nó là **đốm**, tức là kết cấu chứ không phải màu hay hình dạng. Một histogram LBP (Local Binary Pattern) hoặc thống kê từ ma trận đồng xuất hiện mức xám (GLCM) sẽ mô tả được đốm. Đề chỉ yêu cầu màu và hình dạng, nên nếu thêm thì trình bày như một **thí nghiệm mở rộng**, so sánh có và không có kết cấu.

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
- **Khảo sát $k$ rộng hơn:** khi chỉ dùng màu, $k=15$ còn tốt hơn $k=11$. Nên khảo sát tới khoảng $k=25$.

## 6. Cải tiến kỹ thuật

- **Unit test** cho `knn/knn.py` với dữ liệu tự tạo (hai cụm điểm tách rời phải đạt 100%, vote hòa phải chọn đúng lớp gần hơn). Đảm bảo phần tự cài không có lỗi khi sửa sau này.
- **Tham số dòng lệnh** cho `evaluate.py`: `--seed`, `--k`, `--metric`, `--features color,shape`. Hiện các thí nghiệm bổ sung trong [05-danh-gia.md](05-danh-gia.md) phải chạy bằng đoạn code riêng.
- **Xuất kết quả tự động**: `evaluate.py` ghi bảng accuracy ra JSON và vẽ lại `docs/accuracy_k.png`, `docs/confusion_matrix.png`, để README luôn khớp với code.
- **Vector hóa đánh giá:** tính cả ma trận khoảng cách test × train trong một lệnh thay vì lặp từng ảnh. Không cần với 1.000 ảnh, nhưng hữu ích nếu mở rộng lên cả 37 giống (khoảng 7.400 ảnh).
- **Hiển thị trong app** các ảnh láng giềng gần nhất (không chỉ tên và khoảng cách), để thấy trực quan vì sao k-NN dự đoán như vậy. Rất có ích khi thuyết trình.

## Gợi ý lộ trình

Nếu thời gian có hạn, thứ tự sau cho nhiều giá trị nhất cho báo cáo:

1. Mục 1 (cross-validation, nhiều seed) — để mọi con số sau đó đáng tin.
2. Mục 2 (bỏ Hu) và đưa bảng ablation vào báo cáo.
3. Mục 4a (IoU và thí nghiệm oracle với trimap) — trả lời được "lỗi nằm ở đâu".
4. Mục 3a và 3b (sửa kênh H) — cải tiến nhỏ, có cơ sở lý thuyết rõ.
5. Mục 5 (vote theo khoảng cách, chi-square) — mở rộng phần khảo sát khoảng cách.
