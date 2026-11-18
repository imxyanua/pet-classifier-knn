# 5. Đánh giá

Code: [`knn/evaluate.py`](../knn/evaluate.py), hàm `accuracy_score` và `confusion_matrix` trong [`knn/knn.py`](../knn/knn.py).

## Cách đánh giá

- 1.000 ảnh, chia phân tầng 800 train / 200 test với `seed=42` (mỗi giống 160 / 40).
- Chuẩn hóa z-score bằng mean/std của tập train.
- Chạy k-NN với mọi cặp (khoảng cách, $k$): Euclidean và Manhattan, $k$ = 1, 3, 5, 7, 9, 11.

### Độ chính xác

$$\text{accuracy} = \frac{\text{số ảnh dự đoán đúng}}{\text{tổng số ảnh test}}$$

Vì 5 lớp cân bằng (mỗi lớp 40 ảnh test), accuracy là thước đo hợp lý. Mốc so sánh: đoán ngẫu nhiên đạt 20%.

### Ma trận nhầm lẫn

Ma trận $C$ kích thước 5×5, $C_{ij}$ là số ảnh **thực tế** thuộc lớp $i$ nhưng **dự đoán** là lớp $j$.

- Đường chéo: dự đoán đúng. Tổng mỗi hàng bằng 40.
- Ngoài đường chéo: cho biết **cặp lớp nào hay bị nhầm**, và theo chiều nào.
- Recall của lớp $i$ = $C_{ii} / \sum_j C_{ij}$ (đúng bao nhiêu trong số ảnh thật của lớp đó).
- Precision của lớp $j$ = $C_{jj} / \sum_i C_{ij}$ (đúng bao nhiêu trong số ảnh bị đoán là lớp đó).

## Kết quả chính

| Khoảng cách | k=1 | k=3 | k=5 | k=7 | k=9 | k=11 |
|---|---:|---:|---:|---:|---:|---:|
| Euclidean | 54.0% | 55.5% | 57.5% | 53.0% | 54.5% | 52.5% |
| Manhattan | 55.5% | 55.5% | **58.0%** | **58.0%** | 55.5% | 56.5% |

![Accuracy theo k](accuracy_k.png)

Cấu hình tốt nhất: **Manhattan, $k=5$, 58.0%**.

### Ảnh hưởng của khoảng cách

Manhattan tốt hơn hoặc bằng Euclidean ở cả 6 giá trị $k$, chênh lệch lớn nhất ở $k=7$ (58.0% so với 53.0%). Đường Manhattan cũng ít dao động hơn. Lý do: vector có một số chiều nhiễu (Hu moments, xem bên dưới), và L2 bình phương độ lệch nên các chiều nhiễu này bị khuếch đại.

### Ảnh hưởng của k

- $k=1$ không phải tốt nhất với cả hai khoảng cách: một láng giềng duy nhất quá nhạy với ảnh phân đoạn lỗi.
- Đỉnh ở $k$ = 5–7.
- Với Euclidean, $k$ lớn hơn 5 làm accuracy giảm, vì láng giềng xa bắt đầu đến từ lớp khác.

## Đọc ma trận nhầm lẫn

![Ma trận nhầm lẫn](confusion_matrix.png)

| Giống | Đúng / 40 | Recall | Hay bị nhầm thành |
|---|---:|---:|---|
| Bombay | 29 | 72.5% | Pug (7) |
| Samoyed | 27 | 67.5% | Egyptian Mau (4), Sphynx (4) |
| Pug | 22 | 55.0% | Samoyed (7), Bombay (6), Sphynx (5) |
| Sphynx | 20 | 50.0% | Pug (7), Egyptian Mau (5) |
| Egyptian Mau | 18 | 45.0% | Samoyed (9), Sphynx (8) |

Giải thích:

- **Bombay tốt nhất** vì lông đen tuyền cho $\mu_V$ thấp hẳn các giống khác. Khi nhầm, Bombay thường bị nhầm thành Pug, có lẽ vì mặt Pug đen và mask Pug thường chỉ lấy phần mặt.
- **Samoyed** tách tốt nhờ V cao, S thấp.
- **Egyptian Mau kém nhất.** Lông xám bạc có S thấp, V khá cao, giống Samoyed về màu. Đặc trưng phân biệt chính của Mau là **đốm**, tức là kết cấu (texture), mà vector hiện tại không mô tả.
- **Pug ↔ Sphynx** nhầm lẫn hai chiều (5 và 7): cả hai có tông màu da/kem tương tự, và lỗi phân đoạn ở cả hai giống làm đặc trưng thêm nhiễu.

## Thí nghiệm bổ sung

Các thí nghiệm dưới đây dùng lại cache đặc trưng, Manhattan, và lặp lại với **10 seed chia tập khác nhau** (0–9) để có trung bình và độ lệch chuẩn thay vì một con số duy nhất.

### Độ ổn định của kết quả

Với Manhattan, $k=5$, đủ 34 chiều: **55.5% ± 2.3%** (thấp nhất 51.0%, cao nhất 59.5%).

Con số 58.0% với `seed=42` nằm ở phía trên của khoảng này. Ngoài ra $k$ đang được chọn dựa trên chính tập test, nên 58.0% là ước lượng hơi lạc quan. Con số đáng tin hơn để báo cáo là **khoảng 55–56%**.

### Đóng góp của từng nhóm đặc trưng (ablation)

| Đặc trưng dùng | Số chiều | Accuracy (10 seed) |
|---|---:|---:|
| Chỉ hình dạng | 12 | 29.1% ± 1.7 |
| — chỉ Hu moments | 7 | 26.8% ± 2.8 |
| — chỉ 5 đặc trưng hình học | 5 | 31.8% ± 2.9 |
| Chỉ màu | 22 | 55.5% ± 3.4 |
| Màu + hình dạng | 34 | 55.5% ± 2.3 |
| Màu + 5 hình học (bỏ Hu) | 27 | **57.0% ± 2.4** |

Kết luận:

- **Màu mang gần như toàn bộ thông tin.** Chỉ dùng màu đã đạt mức của cả vector.
- **Hình dạng một mình gần mức đoán ngẫu nhiên** (29% so với 20%). Lý do đã phân tích ở [03-dac-trung.md](03-dac-trung.md): tư thế và lỗi phân đoạn làm hình dạng trong cùng một giống thay đổi quá nhiều.
- **Hu moments gây nhiễu.** Bỏ Hu moments tăng khoảng 1.5 điểm. Chênh lệch này nhỏ hơn độ lệch chuẩn nên chưa thể khẳng định chắc chắn, nhưng xu hướng nhất quán với việc Hu một mình chỉ đạt 26.8%.
- Hình dạng có giảm độ dao động (độ lệch chuẩn 2.3 so với 3.4 khi chỉ dùng màu), tức là giúp kết quả ổn định hơn dù không tăng trung bình.

### Trọng số nhóm hình dạng

Nhân các chiều hình dạng với hệ số $w$ sau chuẩn hóa (màu giữ hệ số 1):

| $w$ | Accuracy (10 seed) |
|---:|---:|
| 0 (bỏ hình dạng) | 55.5% ± 3.4 |
| 0.25 | **56.9% ± 3.6** |
| 0.5 | 56.1% ± 3.5 |
| 1 (hiện tại) | 55.5% ± 2.3 |

Giảm trọng số hình dạng giúp một chút, cho thấy hình dạng có ích ở mức nhỏ nhưng đang bị cho trọng số quá lớn so với lượng thông tin nó mang.

### k tối ưu khi chỉ dùng màu

| $k$ | 1 | 3 | 5 | 7 | 9 | 11 | 15 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Accuracy | 50.2% | 52.8% | 55.5% | 56.5% | 56.0% | 57.2% | **57.5%** |

Khi vector ít nhiễu hơn, $k$ lớn hơn lại tốt hơn: vote trên nhiều láng giềng làm mượt sai số của từng ảnh phân đoạn lỗi. Vì vậy $k$ tối ưu phụ thuộc vào bộ đặc trưng và phải được chọn lại mỗi khi đổi đặc trưng.

## Tái tạo các con số

```bash
python evaluate.py
```

in bảng kết quả chính và ma trận nhầm lẫn. Các thí nghiệm bổ sung chạy trên `features_cache.npz` bằng cách chọn cột của `X` (0–21 màu, 22–26 hình học, 27–33 Hu) rồi gọi lại các hàm trong `knn/knn.py` với `seed` từ 0 đến 9.
