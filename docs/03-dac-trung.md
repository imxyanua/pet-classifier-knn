# 3. Đặc trưng ảnh

Code: [`features/features.py`](../features/features.py), các hàm `extract_features`, `_shape_features`, `_safe_hist`.

Mỗi ảnh được biểu diễn bằng một vector 67 chiều:

| Chỉ số | Nhóm | Số chiều |
|---|---|---:|
| 0–2 | Trung bình H, S, V | 3 |
| 3–5 | Độ lệch chuẩn H, S, V | 3 |
| 6 | Tỷ lệ pixel không màu | 1 |
| 7–22 | Histogram H, 16 bin (chỉ pixel có màu) | 16 |
| 23–38 | Histogram S, 16 bin | 16 |
| 39–54 | Histogram V, 16 bin | 16 |
| 55–59 | Hình học contour | 5 |
| 60–66 | Hu moments | 7 |

Chiều 0–54 là **màu**, 55–66 là **hình dạng**.

Cả 67 chiều đều được trích và lưu trong cache, nhưng **mặc định chỉ dùng 60 chiều đầu** (màu + hình học); Hu moments bị bỏ. Danh sách nhóm dùng mặc định nằm ở `DEFAULT_FEATURE_GROUPS` trong [`config.py`](../config.py), vị trí từng nhóm ở `FEATURE_GROUPS` trong [`features/features.py`](../features/features.py).

## Đặc trưng màu

### Vì sao dùng HSV

Trong không gian BGR, cả ba kênh đều thay đổi khi ánh sáng thay đổi. HSV tách thông tin màu thành:

- **H (Hue, sắc độ):** "màu gì" — vàng, đỏ, xanh. OpenCV dùng thang 0–179.
- **S (Saturation, độ bão hòa):** màu đậm hay nhạt. 0 là xám, 255 là màu thuần.
- **V (Value, độ sáng):** 0 là đen, 255 là sáng nhất.

Nhờ vậy có thể phát biểu trực tiếp các khác biệt giữa các giống:

| Giống | Kỳ vọng |
|---|---|
| Bombay | V rất thấp |
| Samoyed | V rất cao, S thấp |
| Pug | H vùng vàng/cam, S trung bình |
| Sphynx | H vùng hồng/đỏ, S khá cao |
| Egyptian Mau | S thấp, V trung bình, độ lệch chuẩn V cao do đốm |

Số liệu thực tế (trung bình trên 200 ảnh mỗi giống) khớp với kỳ vọng ở kênh V và S:

| Giống | H trung bình | S trung bình | V trung bình |
|---|---:|---:|---:|
| Bombay | 60.7 | 67.9 | **76.7** |
| Egyptian Mau | 57.9 | 49.0 | 134.2 |
| Sphynx | 38.0 | **77.4** | 148.1 |
| Pug | 53.2 | 54.6 | 126.6 |
| Samoyed | 51.1 | **35.9** | **159.6** |

Kênh H thì khác biệt ít hơn mong đợi; xem phần "Vấn đề của kênh H" bên dưới.

### Thống kê

Với tập pixel $P$ nằm trong mask:

- $\mu_S$, $\mu_V$, $\sigma_S$, $\sigma_V$ trên toàn $P$.
- $\mu_H$, $\sigma_H$ chỉ trên pixel **có màu** ($S \ge 40$ và $V \ge 40$), vì H trên pixel đen/xám gần như nhiễu.
- **Tỷ lệ không màu** $= |\{p \in P: S < 40 \lor V < 40\}| / |P|$ — cao với Bombay (đen), Samoyed/Mau (trắng/xám), thấp hơn với Pug/Sphynx có màu rõ.

### Histogram

Đếm số pixel rơi vào từng khoảng giá trị, rồi chia cho tổng để thành tỷ lệ (tổng các bin bằng 1). Histogram H chỉ dùng pixel có màu; S và V dùng toàn mask.

Ban đầu dùng 8/4/4 bin; sau đó **16/16/16** (61.1% → 65.4%). Với ~60 chiều và 800 mẫu train, k-NN vẫn xử lý được.

### Vấn đề của kênh H

1. **H là đại lượng vòng tròn.** Giá trị 0 và 179 đều là đỏ; trung bình số học có thể lệch (đã thử $\cos$/$\sin$, không giữ).
2. **H vô nghĩa khi S hoặc V thấp** — đã xử lý bằng cách lọc pixel có màu + đặc trưng tỷ lệ không màu (PR #9).

## Đặc trưng hình dạng

Tính trên contour lớn nhất của mask.

### 5 đặc trưng hình học

Ký hiệu: $A$ là diện tích contour, $P$ là chu vi, $w \times h$ là bounding box thẳng, $A_{hull}$ là diện tích bao lồi.

| Đặc trưng | Công thức | Ý nghĩa |
|---|---|---|
| Aspect ratio | $w / h$ | > 1: dáng ngang (nằm, đứng nghiêng); < 1: dáng dọc (ngồi) |
| Extent | $A / (w h)$ | Con vật lấp đầy khung bao bao nhiêu |
| Solidity | $A / A_{hull}$ | Gần 1: khối đặc, tròn trịa (Pug); thấp: nhiều phần nhô ra như chân, đuôi, tai (Sphynx, Mau) |
| Circularity | $4\pi A / P^2$ | Bằng 1 với hình tròn, nhỏ với hình dài hoặc viền răng cưa |
| Eccentricity | cạnh ngắn / cạnh dài của hình chữ nhật xoay nhỏ nhất | Độ thon dài, không phụ thuộc góc xoay |

### Hu moments

Moment ảnh bậc $(p, q)$ của mask:

$$m_{pq} = \sum_x \sum_y x^p y^q \, I(x, y)$$

Từ moment trung tâm chuẩn hóa, Hu (1962) xây dựng 7 bất biến không đổi khi **tịnh tiến, co giãn và xoay** hình. Vì giá trị Hu chênh nhau nhiều bậc độ lớn (từ $10^{-1}$ đến $10^{-20}$), project biến đổi:

$$h_i' = -\text{sign}(h_i) \cdot \log_{10}(|h_i| + 10^{-12})$$

để đưa về cùng thang đo trước khi chuẩn hóa.

### Hạn chế của đặc trưng hình dạng

Ý tưởng là phân biệt dáng tròn (Pug) với dáng mảnh (Sphynx). Thí nghiệm ở [05-danh-gia.md](05-danh-gia.md) cho thấy:

- **Một mình hình dạng rất yếu:** 5 đặc trưng hình học đạt khoảng 31%, Hu moments khoảng 29%, so với 20% của đoán bừa.
- **5 đặc trưng hình học vẫn có ích khi gộp với màu:** thêm khoảng 2.5 điểm so với chỉ dùng màu.
- **Hu moments không có ích:** gộp thêm vào không tăng accuracy, nên bị bỏ khỏi bộ mặc định.

Lý do hình dạng yếu:

- cùng một giống nhưng nằm, ngồi, đứng cho contour hoàn toàn khác nhau,
- ảnh chỉ chụp mặt và ảnh chụp toàn thân cho hình dạng khác hẳn,
- lỗi phân đoạn (thiếu thân, dính nền) làm contour sai trực tiếp.

Nói cách khác, sự khác biệt hình dạng **trong** một giống lớn hơn sự khác biệt **giữa** các giống. Đặc trưng hình dạng toàn cục phù hợp hơn với bài toán vật thể cứng, có tư thế cố định (lá cây, linh kiện, chữ số).

Hu moments chịu ảnh hưởng nặng nhất vì chúng mô tả chi tiết phân bố khối của toàn bộ mask: chỉ cần đuôi hay chân bị cắt mất là giá trị thay đổi lớn. 5 đặc trưng hình học thô hơn (tỷ lệ khung, độ đặc, độ tròn) nên ít nhạy với các lỗi nhỏ đó.
