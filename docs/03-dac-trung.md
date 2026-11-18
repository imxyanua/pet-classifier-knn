# 3. Đặc trưng ảnh

Code: [`features/features.py`](../features/features.py), các hàm `extract_features`, `_shape_features`, `_safe_hist`.

Mỗi ảnh được biểu diễn bằng một vector 34 chiều:

| Chỉ số | Nhóm | Số chiều |
|---|---|---:|
| 0–2 | Trung bình H, S, V | 3 |
| 3–5 | Độ lệch chuẩn H, S, V | 3 |
| 6–13 | Histogram H, 8 bin | 8 |
| 14–17 | Histogram S, 4 bin | 4 |
| 18–21 | Histogram V, 4 bin | 4 |
| 22–26 | Hình học contour | 5 |
| 27–33 | Hu moments | 7 |

Chiều 0–21 là **màu**, 22–33 là **hình dạng**.

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

- trung bình $\mu_c = \frac{1}{|P|}\sum_{p \in P} p_c$ với $c \in$ {H, S, V}
- độ lệch chuẩn $\sigma_c$ đo độ đồng đều màu. Bombay và Samoyed có $\sigma$ thấp (một màu), Egyptian Mau có $\sigma_V$ cao (đốm sáng tối xen kẽ).

### Histogram

Đếm số pixel rơi vào từng khoảng giá trị, rồi chia cho tổng để thành tỷ lệ (tổng các bin bằng 1). Chuẩn hóa này làm histogram không phụ thuộc kích thước con vật trong ảnh.

Số bin được chọn ít (8/4/4) vì k-NN chịu ảnh hưởng của **lời nguyền số chiều**: càng nhiều chiều thì khoảng cách giữa các điểm càng gần nhau, láng giềng "gần nhất" càng kém ý nghĩa. Với 800 mẫu train, vector vài chục chiều là hợp lý.

### Vấn đề của kênh H

Hai vấn đề làm kênh H kém tin cậy hơn S và V:

1. **H là đại lượng vòng tròn.** Giá trị 0 và 179 đều là màu đỏ, nằm sát nhau trên vòng màu, nhưng trung bình số học của chúng là 90 (màu xanh lam). Da hồng của Sphynx nằm đúng vùng đỏ này, nên $\mu_H$ của Sphynx có thể bị kéo lệch.
2. **H vô nghĩa khi S hoặc V thấp.** Pixel gần đen (Bombay) hoặc gần trắng/xám (Samoyed, Egyptian Mau) có H gần như ngẫu nhiên. Ví dụ Bombay có $\mu_H$ = 60.7, đây chủ yếu là nhiễu chứ không phải màu thật.

Cách khắc phục nằm trong [06-huong-cai-thien.md](06-huong-cai-thien.md).

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

Ý tưởng là phân biệt dáng tròn (Pug) với dáng mảnh (Sphynx). Trên thực tế, thí nghiệm ở [05-danh-gia.md](05-danh-gia.md) cho thấy nhóm này **gần như không giúp phân loại**:

- cùng một giống nhưng nằm, ngồi, đứng cho contour hoàn toàn khác nhau,
- ảnh chỉ chụp mặt và ảnh chụp toàn thân cho hình dạng khác hẳn,
- lỗi phân đoạn (thiếu thân, dính nền) làm contour sai trực tiếp.

Nói cách khác, sự khác biệt hình dạng **trong** một giống lớn hơn sự khác biệt **giữa** các giống. Đặc trưng hình dạng toàn cục phù hợp hơn với bài toán vật thể cứng, có tư thế cố định (lá cây, linh kiện, chữ số), không phù hợp với động vật.
