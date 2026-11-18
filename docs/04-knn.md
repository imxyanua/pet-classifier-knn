# 4. Thuật toán k-NN

Code: [`knn/knn.py`](../knn/knn.py). Không dùng scikit-learn.

## Ý tưởng

k-NN (k láng giềng gần nhất) phân loại một mẫu mới bằng cách:

1. tính khoảng cách từ mẫu đó đến **mọi** mẫu trong tập train,
2. lấy $k$ mẫu gần nhất,
3. chọn nhãn xuất hiện nhiều nhất trong $k$ mẫu đó.

k-NN là phương pháp **học lười** (lazy learning): bước "huấn luyện" chỉ là lưu lại tập train, không học tham số nào. Toàn bộ chi phí nằm ở bước dự đoán. Vì không học gì, k-NN hoàn toàn dựa vào giả định: **hai ảnh có vector đặc trưng gần nhau thì cùng giống**. Giả định này đúng đến đâu là do chất lượng đặc trưng quyết định.

## Chuẩn hóa z-score

```python
mean = X_train.mean(axis=0)
std  = X_train.std(axis=0)          # std ≈ 0 được thay bằng 1 để tránh chia 0
X'   = (X - mean) / std
```

Mỗi chiều sau chuẩn hóa có trung bình 0, độ lệch chuẩn 1 trên tập train.

**Vì sao bắt buộc:** các đặc trưng có thang đo rất khác nhau. $\mu_V$ nằm trong khoảng 0–255, bin histogram trong 0–1, aspect ratio quanh 0.5–2. Nếu không chuẩn hóa, khoảng cách gần như chỉ do $\mu_V$ và $\sigma_V$ quyết định, các đặc trưng khác bị bỏ qua.

**Chỉ dùng mean/std của tập train**, sau đó áp dụng cùng giá trị cho tập test và cho ảnh mới trong app. Nếu tính mean/std trên cả tập test, thông tin của test bị rò rỉ vào mô hình và kết quả bị lạc quan.

## Khoảng cách

Với $x$ là mẫu cần dự đoán và $x_i$ là một mẫu train, cả hai $d$ chiều:

**Euclidean (L2)**

$$d_2(x, x_i) = \sqrt{\sum_{j=1}^{d} (x_j - x_{ij})^2}$$

**Manhattan (L1)**

$$d_1(x, x_i) = \sum_{j=1}^{d} |x_j - x_{ij}|$$

Khác biệt chính: L2 bình phương độ lệch nên **một chiều lệch lớn chi phối cả tổng**. L1 cộng tuyến tính nên mỗi chiều đóng góp đều hơn. Khi vector gộp nhiều nhóm đặc trưng và có một số chiều nhiễu (như Hu moments), L1 thường ổn định hơn. Kết quả thí nghiệm khớp với điều này: Manhattan tốt hơn hoặc bằng Euclidean ở mọi $k$.

Cả hai hàm đều tính vector hóa: `X_train - x` dùng broadcasting của NumPy để trừ $x$ khỏi mọi dòng cùng lúc, cho ra $N$ khoảng cách trong một lệnh.

## Tìm k láng giềng

```python
dists = DISTANCES[metric](X_train, x)
nn = np.argpartition(dists, k - 1)[:k]     # k chỉ số nhỏ nhất, chưa sắp xếp
nn = nn[np.argsort(dists[nn])]              # sắp xếp k phần tử đó
```

`argpartition` tìm $k$ phần tử nhỏ nhất trong $O(N)$ thay vì sắp xếp toàn bộ $O(N \log N)$. Sau đó chỉ sắp xếp $k$ phần tử để hiển thị láng giềng theo thứ tự trong app.

## Vote

Chọn nhãn xuất hiện nhiều nhất trong $k$ láng giềng. Nếu hai nhãn **hòa số phiếu**, chọn nhãn có khoảng cách trung bình nhỏ hơn.

Ví dụ $k=4$, láng giềng là [Pug 0.8, Bombay 1.0, Pug 1.5, Bombay 1.1]:

- Pug: 2 phiếu, khoảng cách trung bình 1.15
- Bombay: 2 phiếu, khoảng cách trung bình 1.05
- kết quả: **Bombay**

Với 5 lớp, hòa có thể xảy ra ngay cả khi $k$ lẻ (ví dụ $k=5$ chia 2–2–1), nên quy tắc phá hòa là cần thiết.

## Ảnh hưởng của k

| $k$ | Hành vi |
|---|---|
| Nhỏ (1–3) | Biên quyết định rất gồ ghề, nhạy với từng mẫu nhiễu hoặc gán sai. Dễ **overfit**. |
| Vừa | Cân bằng giữa chi tiết và ổn định. |
| Lớn | Biên trơn, nhưng láng giềng bắt đầu lấy mẫu từ lớp khác. Dễ **underfit**; khi $k$ tiến tới $N$, luôn dự đoán lớp đông nhất. |

Trên dữ liệu này, $k=1$ luôn kém nhất (khoảng 50% accuracy CV). Với Manhattan, accuracy tăng tới khoảng $k=7$ rồi gần như phẳng tới $k=25$; với Euclidean, đạt đỉnh ở $k=13$ rồi giảm. Chi tiết ở [05-danh-gia.md](05-danh-gia.md).

## Chọn k bằng cross-validation

Code: [`knn/validation.py`](../knn/validation.py).

$k$ và khoảng cách là **siêu tham số**: chúng không được học từ dữ liệu mà phải chọn trước. Cách sai phổ biến là thử mọi $k$ trên tập test rồi báo cáo $k$ cho kết quả cao nhất. Làm vậy, tập test đã bị dùng để ra quyết định, nên con số báo cáo bị lạc quan: nó đo xem cấu hình được chọn khớp với **chính 200 ảnh đó** tốt đến đâu, chứ không đo khả năng trên ảnh mới.

Cross-validation $K$ fold (ở đây $K=5$) giải quyết bằng cách chỉ dùng tập train:

1. `stratified_k_fold` chia 800 ảnh train thành 5 phần, mỗi phần giữ tỷ lệ lớp (mỗi giống 32 ảnh mỗi phần).
2. Với mỗi phần $f$: dùng 4 phần còn lại làm "train tạm", phần $f$ làm "validation".
3. `cross_val_accuracy` chạy k-NN và tính accuracy trên từng phần validation, rồi lấy trung bình 5 lần.
4. `select_params` lặp bước 2–3 cho mọi cặp (khoảng cách, $k$) và chọn cặp có trung bình cao nhất. Hòa thì giữ cặp gặp trước, tức $k$ nhỏ hơn.

Chi tiết quan trọng: **mean/std chuẩn hóa được tính lại trên 4 phần train của từng fold**, không tính một lần trên cả 800 ảnh. Nếu tính trên cả 800 ảnh, thông tin của phần validation đã lọt vào bước chuẩn hóa, và accuracy CV lại bị lạc quan, giống lỗi ở trên nhưng ở quy mô nhỏ hơn.

Sau khi chọn xong, chuẩn hóa lại trên toàn bộ 800 ảnh train và dự đoán 200 ảnh test **đúng một lần**.

Chi phí: 2 khoảng cách × 10 giá trị $k$ × 5 fold = 100 lần chạy k-NN, mỗi lần 160 ảnh validation × 640 ảnh train. Toàn bộ chỉ mất vài giây.

## Chia tập phân tầng

`stratified_split` chia **riêng từng lớp**: xáo trộn chỉ số của lớp đó (theo `seed`) rồi lấy 20% làm test. Mỗi giống có đúng 40 ảnh test và 160 ảnh train. Nếu chia ngẫu nhiên toàn bộ, một lớp có thể bị thiếu mẫu test, làm accuracy và ma trận nhầm lẫn lệch.

## Độ phức tạp

Với $N$ mẫu train, $d$ chiều, $M$ mẫu test:

- huấn luyện: $O(1)$ (chỉ lưu dữ liệu) + $O(Nd)$ cho chuẩn hóa
- dự đoán một mẫu: $O(Nd)$
- đánh giá toàn bộ test: $O(MNd)$ = 200 × 800 × 34 ≈ 5,4 triệu phép toán, chạy dưới 1 giây

k-NN chậm với dữ liệu lớn (hàng triệu ảnh) vì phải so với mọi mẫu, nhưng với 1.000 ảnh thì không đáng kể. Phần chậm thật sự của project là GrabCut khi trích đặc trưng.
