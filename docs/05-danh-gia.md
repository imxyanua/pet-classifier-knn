# 5. Đánh giá

Code: [`knn/evaluate.py`](../knn/evaluate.py); các hàm metric trong [`knn/knn.py`](../knn/knn.py); cross-validation trong [`knn/validation.py`](../knn/validation.py).

## Quy trình

1. Chia 1.000 ảnh thành 800 train / 200 test, phân tầng (mỗi giống 160 / 40).
2. Trên 800 ảnh train, chạy cross-validation 5 fold cho mọi cặp khoảng cách (Euclidean, Manhattan) × $k$ (1, 3, 5, 7, 9, 11, 13, 15, 19, 25). Chọn cặp có accuracy CV cao nhất.
3. Chuẩn hóa trên 800 ảnh train, chạy k-NN với cặp đã chọn, đánh giá **một lần** trên 200 ảnh test.
4. Lặp bước 1–3 với seed 0–9 (10 cách chia khác nhau), báo cáo trung bình ± độ lệch chuẩn.

Lý do của từng bước được giải thích ở [04-knn.md](04-knn.md), mục "Chọn k bằng cross-validation". Tóm tắt: tập test không tham gia vào bất kỳ quyết định nào, nên accuracy trên test là ước lượng không thiên lệch cho ảnh mới. Lặp nhiều seed cho biết con số đó dao động bao nhiêu.

Đặc trưng mặc định: màu + hình học, 27 chiều (xem [03-dac-trung.md](03-dac-trung.md)).

### Các thước đo

**Accuracy**

$$\text{accuracy} = \frac{\text{số ảnh dự đoán đúng}}{\text{tổng số ảnh test}}$$

5 lớp cân bằng (mỗi lớp 40 ảnh test) nên accuracy là thước đo hợp lý. Mốc so sánh: đoán ngẫu nhiên đạt 20%.

**Ma trận nhầm lẫn**

Ma trận $C$ kích thước 5×5, $C_{ij}$ là số ảnh **thực tế** thuộc lớp $i$ nhưng **dự đoán** là lớp $j$.

- Đường chéo: dự đoán đúng. Tổng mỗi hàng bằng 40.
- Ngoài đường chéo: cho biết **cặp lớp nào hay bị nhầm**, và theo chiều nào.

**Precision, recall, F1 theo lớp** (hàm `per_class_metrics`)

- Recall của lớp $i$ = $C_{ii} / \sum_j C_{ij}$: trong số ảnh thật của lớp $i$, đoán đúng bao nhiêu.
- Precision của lớp $j$ = $C_{jj} / \sum_i C_{ij}$: trong số ảnh bị đoán là lớp $j$, đúng bao nhiêu.
- F1 = $2 \cdot \text{precision} \cdot \text{recall} / (\text{precision} + \text{recall})$: trung bình điều hòa của hai giá trị trên.

Accuracy chung có thể che mất một lớp yếu; các thước đo theo lớp chỉ ra chính xác lớp nào yếu và yếu theo kiểu nào (bị bỏ sót hay bị nhận nhầm).

## Kết quả chính

| | Accuracy trên test |
|---|---:|
| Trung bình 10 seed | **59.3% ± 3.6%** |
| Thấp nhất / cao nhất | 54.5% / 68.0% |
| `seed=42` | 59.0% |

CV chọn **Manhattan ở cả 10 seed**. Giá trị $k$ được chọn: 25 (5 lần), 13 (2), 15 (2), 19 (1).

Khoảng dao động 54.5%–68.0% là lớn. Nguyên nhân chính là tập test nhỏ: với 200 ảnh, mỗi ảnh đúng/sai thay đổi accuracy 0.5 điểm, và việc ảnh "khó" (phân đoạn lỗi) rơi vào test nhiều hay ít phụ thuộc may rủi. Vì vậy **chênh lệch dưới khoảng 3–4 điểm giữa hai phương án không đủ để kết luận phương án nào tốt hơn**.

### Ảnh hưởng của k và khoảng cách

![Accuracy CV theo k](accuracy_k.png)

Accuracy cross-validation trên 800 ảnh train với `seed=42`:

| $k$ | 1 | 3 | 5 | 7 | 9 | 11 | 13 | 15 | 19 | 25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Euclidean | 49.5 | 53.9 | 54.0 | 55.9 | 55.8 | 56.4 | 57.4 | 56.8 | 55.9 | 55.4 |
| Manhattan | 51.3 | 55.1 | 56.6 | **58.4** | 57.4 | 57.6 | 57.8 | 58.3 | 57.9 | **58.4** |

- **Manhattan tốt hơn Euclidean ở mọi $k$.** L2 bình phương độ lệch nên một chiều lệch lớn chi phối cả khoảng cách; L1 cộng tuyến tính nên các chiều đóng góp đều hơn. Với vector trộn màu và hình học, nơi một vài chiều có thể nhiễu do phân đoạn lỗi, L1 bền hơn.
- **$k$ nhỏ kém rõ rệt.** $k=1$ thấp hơn đỉnh khoảng 7 điểm: một láng giềng duy nhất quá nhạy với ảnh bị phân đoạn sai hoặc ảnh lạ.
- **Với Manhattan, đường gần như phẳng từ $k=7$ đến $k=25$** (dao động trong khoảng 1 điểm). Đó là lý do CV chọn $k$ khác nhau giữa các seed: các giá trị này gần như tương đương, và sai số ngẫu nhiên quyết định cái nào nhỉnh hơn.
- **Với Euclidean, đạt đỉnh ở $k=13$ rồi giảm**: láng giềng xa bắt đầu đến từ lớp khác.

Với `seed=42`, $k=7$ và $k=25$ hòa nhau (58.38%), CV giữ $k=7$ vì gặp trước.

## Đọc ma trận nhầm lẫn

![Ma trận nhầm lẫn](confusion_matrix.png)

`seed=42`, Manhattan, $k=7$, accuracy 59.0%:

| Giống | Đúng / 40 | Precision | Recall | F1 | Hay bị nhầm thành |
|---|---:|---:|---:|---:|---|
| Bombay | 28 | 71.8% | 70.0% | **70.9%** | Pug (8) |
| Samoyed | 26 | 60.5% | 65.0% | 62.7% | Egyptian Mau (5), Pug (4) |
| Pug | 23 | 53.5% | 57.5% | 55.4% | Bombay (5), Sphynx (5) |
| Egyptian Mau | 21 | 53.8% | 52.5% | 53.2% | Samoyed (7), Sphynx (6) |
| Sphynx | 20 | 55.6% | 50.0% | 52.6% | Egyptian Mau (8), Samoyed (6) |

Giải thích:

- **Bombay tốt nhất** nhờ lông đen tuyền cho $\mu_V$ thấp hẳn các giống khác. Khi sai, Bombay hay bị đoán là Pug, có lẽ vì mặt Pug đen và mask Pug thường chỉ lấy được phần mặt.
- **Sphynx ↔ Egyptian Mau** là cặp nhầm lẫn lớn nhất (8 và 6 ảnh). Cả hai có tông xám/nhạt, dáng mảnh, tai to. Nét phân biệt chính của Mau là **đốm**, tức kết cấu, mà vector hiện tại không mô tả.
- **Egyptian Mau ↔ Samoyed** (7 và 5): lông xám bạc của Mau có S thấp, V khá cao, gần với lông trắng của Samoyed.
- **Pug** có precision thấp nhất trong nhóm chó (53.5%): nhiều ảnh của giống khác bị đoán là Pug (8 Bombay, 5 Sphynx, 4 Samoyed). Mask Pug thường không đầy đủ nên đặc trưng của Pug trải rộng, "hút" các ảnh lạ về phía nó.

## Đóng góp của từng nhóm đặc trưng (ablation)

Cùng quy trình (CV chọn tham số, 10 seed), chỉ thay đổi nhóm đặc trưng:

```bash
python evaluate.py --features color,geometry,hu
python evaluate.py --features color
...
```

| Đặc trưng | Số chiều | Accuracy test (10 seed) |
|---|---:|---:|
| Màu + hình học + Hu | 34 | 58.8% ± 2.7 |
| **Màu + hình học (mặc định)** | 27 | **59.3% ± 3.6** |
| Chỉ màu | 22 | 55.6% ± 3.2 |
| Hình học + Hu | 12 | 29.1% ± 3.3 |
| Chỉ hình học | 5 | 31.2% ± 3.3 |
| Chỉ Hu moments | 7 | 25.9% ± 2.5 |

Kết luận:

- **Màu mang phần lớn thông tin.** Chỉ màu đã đạt 55.6%.
- **5 đặc trưng hình học bổ sung khoảng 3.7 điểm** khi gộp với màu, dù một mình chỉ đạt 31%. Hai nhóm mang thông tin khác nhau: màu cho biết "lông màu gì", hình học cho biết "dáng tròn hay mảnh".
- **Hu moments không có ích.** Một mình gần mức đoán bừa (25.9% so với 20%), và thêm vào không tăng accuracy. Chúng bị bỏ khỏi bộ mặc định vì làm vector dài hơn mà không mang thêm thông tin. Mức chênh 58.8% so với 59.3% nằm trong độ dao động, nên **không** được coi là bằng chứng bỏ Hu làm tăng accuracy.

### So với cách đánh giá cũ

Phiên bản đầu tiên chọn $k$ trực tiếp trên tập test với một seed duy nhất và chỉ thử $k \le 11$. Cách đó cho 58.0% (Manhattan, $k=5$) và dẫn tới kết luận sai rằng "hình dạng gần như không giúp": ở $k=5$ cố định, màu + hình dạng và chỉ màu cho cùng 55.5%. Khi cho phép CV chọn $k$ lớn hơn, lợi ích của hình học mới lộ ra, có lẽ vì vote trên nhiều láng giềng làm mượt nhiễu của từng chiều hình dạng.

Bài học: một cấu hình cố định có thể che mất đóng góp của một nhóm đặc trưng. So sánh đặc trưng nên được làm **sau khi** chọn siêu tham số riêng cho từng phương án.

## Tái tạo các con số

```bash
python evaluate.py                                # kết quả chính và bảng CV
python evaluate.py --features color,geometry,hu   # một dòng trong bảng ablation
python evaluate.py --repeats 0                     # chỉ chạy seed=42, bỏ phần lặp
```

Lần đầu mất 1–2 phút để trích đặc trưng; các lần sau dùng cache, mỗi lần chạy khoảng 30 giây.
