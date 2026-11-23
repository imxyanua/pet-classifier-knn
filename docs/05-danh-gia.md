# 5. Đánh giá

Code: [`knn/evaluate.py`](../knn/evaluate.py); các hàm metric trong [`knn/knn.py`](../knn/knn.py); cross-validation trong [`knn/validation.py`](../knn/validation.py).

## Quy trình

1. Chia 1.000 ảnh thành 800 train / 200 test, phân tầng (mỗi giống 160 / 40).
2. Trên 800 ảnh train, chạy cross-validation 5 fold cho mọi cặp khoảng cách (Euclidean, Manhattan) × $k$ (1, 3, 5, 7, 9, 11, 13, 15, 19, 25). Chọn cặp có accuracy CV cao nhất.
3. Chuẩn hóa trên 800 ảnh train, chạy k-NN với cặp đã chọn, đánh giá **một lần** trên 200 ảnh test.
4. Lặp bước 1–3 với seed 0–9 (10 cách chia khác nhau), báo cáo trung bình ± độ lệch chuẩn.

Lý do của từng bước được giải thích ở [04-knn.md](04-knn.md), mục "Chọn k bằng cross-validation". Tóm tắt: tập test không tham gia vào bất kỳ quyết định nào, nên accuracy trên test là ước lượng không thiên lệch cho ảnh mới. Lặp nhiều seed cho biết con số đó dao động bao nhiêu.

Đặc trưng mặc định: màu + hình học, 59 chiều (xem [03-dac-trung.md](03-dac-trung.md)).

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
| Trung bình 10 seed | **66.6% ± 2.9%** |
| Thấp nhất / cao nhất | 62.0% / 72.0% |
| `seed=42` | 66.0% |

CV chọn **Manhattan + vote theo khoảng cách ở cả 10 seed**. Giá trị $k$ được chọn: 19 (5 lần), 25 (3), 15 (1), 13 (1 lần). Cosine và χ² có trong lưới tìm kiếm nhưng không bao giờ thắng.

Khoảng dao động 62.0%–72.0% vẫn đáng kể. Nguyên nhân chính là tập test nhỏ: với 200 ảnh, mỗi ảnh đúng/sai thay đổi accuracy 0.5 điểm, và việc ảnh "khó" (phân đoạn lỗi) rơi vào test nhiều hay ít phụ thuộc may rủi. Vì vậy **chênh lệch dưới khoảng 3–4 điểm giữa hai phương án không đủ để kết luận phương án nào tốt hơn**; bằng chứng mạnh hơn là CV tăng nhất quán (vote theo khoảng cách thắng vote đều trên lưới $k$).

### Ảnh hưởng của k và khoảng cách

![Accuracy CV theo k](accuracy_k.png)

Accuracy cross-validation trên 800 ảnh train với `seed=42`:

Bảng dưới là **vote theo khoảng cách** (`seed=42`), vì CV luôn chọn kiểu vote này:

| $k$ | 1 | 3 | 5 | 7 | 9 | 11 | 13 | 15 | 19 | 25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Euclidean | 58.5 | 61.1 | 61.5 | 63.0 | 63.9 | 63.4 | 63.3 | 64.1 | 63.5 | 64.6 |
| Manhattan | 59.9 | 62.8 | 64.6 | 65.9 | 66.0 | 66.6 | 66.4 | **66.8** | 66.4 | 65.8 |
| Cosine | 57.8 | 62.1 | 62.6 | 63.5 | 65.0 | 63.6 | 63.3 | 64.4 | 63.8 | 63.6 |
| Chi-square | 53.6 | 55.1 | 58.4 | 61.0 | 59.9 | 61.0 | 60.4 | 60.6 | 63.0 | 61.9 |

- **Manhattan tốt nhất**; cosine gần Euclidean; χ² kém rõ dù đã giữ histogram thô.
- **Vote theo khoảng cách** nâng Manhattan khoảng 1–2 điểm CV so với vote đều (đỉnh 66.8% so với 65.1%).
- **$k$ nhỏ kém rõ rệt.** $k=1$ thấp hơn đỉnh khoảng 7 điểm.
- **Với Manhattan, đường gần như phẳng từ $k=11$ đến $k=19$**. CV chọn $k$ khác nhau giữa các seed vì các giá trị gần tương đương.

## Đọc ma trận nhầm lẫn

![Ma trận nhầm lẫn](confusion_matrix.png)

`seed=42`, Manhattan, vote theo khoảng cách, $k=15$, accuracy 66.0%:

| Giống | Đúng / 40 | Precision | Recall | F1 | Hay bị nhầm thành |
|---|---:|---:|---:|---:|---|
| Bombay | 31 | 75.6% | 77.5% | **76.5%** | Pug (7), Sphynx (1) |
| Sphynx | 23 | 88.5% | 57.5% | 69.7% | Samoyed (7), Egyptian Mau (6) |
| Samoyed | 29 | 58.0% | 72.5% | 64.4% | Pug (8), Egyptian Mau (3) |
| Pug | 27 | 52.9% | 67.5% | 59.3% | Samoyed (8), Bombay (4) |
| Egyptian Mau | 22 | 68.8% | 55.0% | 61.1% | Pug (5), Samoyed (6), Bombay (6) |

Giải thích:

- **Bombay tốt nhất** nhờ lông đen tuyền cho $\mu_V$ thấp hẳn các giống khác. **Samoyed** và **Pug** có recall cao (70% và 67.5%) nhưng precision thấp hơn vì bị các giống khác "đổ" vào.
- **Pug "hút" ảnh của các giống khác**: precision chỉ 50%. Mask Pug thường không đầy đủ (chỉ lấy mặt, hoặc dính nền), nên các vector Pug trải rộng trong không gian đặc trưng và trở thành láng giềng của nhiều ảnh lạ.
- **Egyptian Mau yếu nhất** (recall 47.5%): hay bị nhầm thành Pug, Samoyed hoặc Bombay. Nét phân biệt chính của Mau là **đốm**, tức kết cấu, mà vector hiện tại không mô tả.
- **Sphynx → Samoyed** (7 ảnh) và **Sphynx → Egyptian Mau** (6 ảnh): da hồng nhạt có S thấp, V cao, gần với lông trắng/xám; Sphynx cũng là giống có mask dính nền nhiều nhất (xem [02-tien-xu-ly-va-phan-doan.md](02-tien-xu-ly-va-phan-doan.md)), nên màu nền sáng lẫn vào đặc trưng.

## Đóng góp của từng nhóm đặc trưng (ablation)

Cùng quy trình (CV chọn tham số, 10 seed), chỉ thay đổi nhóm đặc trưng:

```bash
python evaluate.py --features color,geometry,hu
python evaluate.py --features color
...
```

| Đặc trưng | Số chiều | Accuracy test (10 seed) |
|---|---:|---:|
| Màu + hình học + Hu | 66 | 66.6% ± 2.4 |
| **Màu + hình học (mặc định)** | 59 | **66.6% ± 2.9** |
| Chỉ màu | 54 | 63.7% ± 2.5 |
| Hình học + Hu | 12 | 32.4% ± 2.4 |
| Chỉ hình học | 5 | 31.1% ± 3.5 |
| Chỉ Hu moments | 7 | 29.3% ± 0.8 |

Kết luận:

- **Màu mang phần lớn thông tin.** Chỉ màu đã đạt 63.7%.
- **5 đặc trưng hình học bổ sung khoảng 1.7 điểm** khi gộp với màu, dù một mình chỉ đạt 31%. Hai nhóm mang thông tin khác nhau: màu cho biết "lông màu gì", hình học cho biết "dáng tròn hay mảnh".
- **Hu moments một mình gần mức đoán bừa** (29.3% so với 20%). Thêm Hu vào mặc định hiện cho 66.6% ± 2.4 — khoảng 1 điểm trên 65.4% — nằm trong độ dao động, nên mặc định vẫn bỏ Hu để vector gọn hơn.

## Các phiên bản

| Phiên bản | Thay đổi | Accuracy báo cáo |
|---|---|---:|
| 1 | Chọn $k \le 11$ trực tiếp trên tập test, một seed | 58.0% (lạc quan) |
| 2 | Chọn $k \le 25$ bằng cross-validation trên train, 10 seed, bỏ Hu | 59.3% ± 3.6 |
| 3 | Khởi tạo GrabCut bằng mask viền/tâm, 5 vòng | 61.1% ± 2.8 |
| 4 | Histogram HSV 16/16/16 (thay 8/4/4) | 65.4% ± 2.1 |
| 5 | Vote theo khoảng cách; khảo sát cosine / χ² | **66.6% ± 2.9** |

Con số của phiên bản 1 không so sánh trực tiếp được với hai phiên bản sau, vì nó dùng tập test để chọn $k$.

Phiên bản 1 còn dẫn tới một kết luận sai: "hình dạng gần như không giúp". Ở $k=5$ cố định, màu + hình dạng và chỉ màu cho cùng 55.5%. Khi cho phép CV chọn $k$ lớn hơn, lợi ích của hình học mới lộ ra, có lẽ vì vote trên nhiều láng giềng làm mượt nhiễu của từng chiều hình dạng. Bài học: một cấu hình cố định có thể che mất đóng góp của một nhóm đặc trưng; so sánh đặc trưng nên được làm **sau khi** chọn siêu tham số riêng cho từng phương án.

Ảnh hưởng của phân đoạn (phiên bản 3) được phân tích ở [02-tien-xu-ly-va-phan-doan.md](02-tien-xu-ly-va-phan-doan.md).

## Tái tạo các con số

```bash
python evaluate.py                                # kết quả chính và bảng CV
python evaluate.py --features color,geometry,hu   # một dòng trong bảng ablation
python evaluate.py --repeats 0                     # chỉ chạy seed=42, bỏ phần lặp
```

Lần đầu mất khoảng 1 phút để trích đặc trưng (chạy song song); các lần sau dùng cache, mỗi lần chạy khoảng 30 giây.
