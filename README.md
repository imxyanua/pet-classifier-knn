# Pet-Classifier-kNN: Nhận dạng giống thú cưng bằng k-NN trên đặc trưng ảnh

<img width="576" alt="segmentation" src="docs/segmentation.png" />

Nhận dạng 5 giống chó/mèo trong bộ Oxford-IIIT Pet bằng thuật toán k láng giềng gần nhất (k-NN), dựa trên đặc trưng **màu** và **hình dạng** trích từ ảnh.

Đây là bài tập môn xử lý ảnh (đề 34: *Nhận dạng đối tượng bằng k-NN dựa trên đặc trưng ảnh*). Toàn bộ phần k-NN và đánh giá được tự cài bằng NumPy, không dùng scikit-learn. OpenCV chỉ dùng cho các thao tác xử lý ảnh (đổi không gian màu, GrabCut, contour, moments).

Project vẫn đang được cải thiện dần, mục tiêu là dễ hiểu từ đầu đến cuối chứ không phải độ chính xác cao nhất.

## Cách hoạt động

Mỗi ảnh đi qua pipeline sau:

    ảnh gốc -> resize (cạnh dài 192px) -> Gaussian blur 5×5 ↩
            -> GrabCut + morphology + giữ vùng lớn nhất -> mask đối tượng ↩
            -> đặc trưng màu (HSV) + đặc trưng hình dạng (contour) -> vector 34 chiều ↩
            -> chuẩn hóa z-score -> k-NN (Euclidean / Manhattan) -> nhãn giống

## Tổng quan

Đề bài yêu cầu bốn phần, và project đáp ứng như sau:

| Yêu cầu | Cách làm |
|---|---|
| Tiền xử lý và phân đoạn đối tượng | Resize, làm mờ, GrabCut, đóng/mở hình thái học, giữ contour lớn nhất |
| Trích đặc trưng màu và hình dạng | Thống kê và histogram HSV trên vùng đối tượng; tỷ lệ khung, độ đặc, độ tròn, Hu moments |
| Cài đặt k-NN, khảo sát ảnh hưởng của \(k\) hoặc khoảng cách | k-NN tự viết; khảo sát \(k \in \{1,3,5,7,9,11\}\) với cả Euclidean và Manhattan |
| Đánh giá bằng độ chính xác và ma trận nhầm lẫn | Accuracy và confusion matrix 5×5 tự cài |

Ngoài ra có một desktop app (CustomTkinter) để nhận dạng từng ảnh và xem kết quả đánh giá trực quan.

## Kết quả

Chia 800 ảnh train / 200 ảnh test (stratified, `seed=42`, mỗi lớp 40 ảnh test).

| Khoảng cách | k=1 | k=3 | k=5 | k=7 | k=9 | k=11 |
|---|---:|---:|---:|---:|---:|---:|
| Euclidean | 54.0% | 55.5% | 57.5% | 53.0% | 54.5% | 52.5% |
| Manhattan | 55.5% | 55.5% | **58.0%** | **58.0%** | 55.5% | 56.5% |

Kết quả tốt nhất: **Manhattan, k=5, accuracy 58.0%**. Đoán ngẫu nhiên với 5 lớp chỉ đạt 20%.

| Accuracy theo k | Ma trận nhầm lẫn |
|---|---|
| <img width="420" alt="accuracy_k" src="docs/accuracy_k.png" /> | <img width="380" alt="confusion_matrix" src="docs/confusion_matrix.png" /> |

Nhận xét:

- **Manhattan ổn định hơn Euclidean** ở hầu hết các giá trị \(k\). Khoảng cách L1 ít bị chi phối bởi một vài chiều lệch lớn, phù hợp khi vector trộn nhiều loại đặc trưng.
- \(k\) quá nhỏ (k=1) dễ bị nhiễu bởi một láng giềng sai; \(k\) quá lớn (k=11 với Euclidean) kéo thêm mẫu của lớp khác vào vote. Vùng \(k\) = 5–7 cho kết quả tốt nhất.
- **Bombay** nhận dạng tốt nhất (29/40) vì lông đen tuyền khác hẳn các giống còn lại.
- **Egyptian Mau** kém nhất (18/40), hay nhầm sang Samoyed (9) và Sphynx (8). Lông xám nhạt lốm đốm có histogram màu gần với lông trắng và da hồng/xám.
- **Pug** bị nhầm sang Samoyed (7) và Bombay (6). Phần lớn lỗi đến từ phân đoạn: GrabCut thường chỉ lấy được phần mặt hoặc lấy lẫn nền.

## Các giống được chọn

| Thư mục | Giống | Đặc điểm nổi bật |
|---|---|---|
| `pug` | Chó Pug | Lông vàng kem, mặt ngắn, thân tròn |
| `samoyed` | Chó Samoyed | Lông trắng muốt, xù |
| `Bombay` | Mèo Bombay | Lông đen tuyền |
| `Egyptian_Mau` | Mèo Ai Cập | Lông xám đốm vằn, tai nhọn |
| `Sphynx` | Mèo không lông | Da hồng/xám, thân mảnh |

Năm giống này được chọn vì khác nhau rõ về màu và dáng, phù hợp với đặc trưng màu và hình dạng mà đề yêu cầu.

## Đặc trưng ảnh

Vector đặc trưng có 34 chiều, chỉ tính trên các pixel thuộc mask đối tượng:

**Màu (22 chiều), không gian HSV**

- trung bình H, S, V (3)
- độ lệch chuẩn H, S, V (3)
- histogram H 8 bin, S 4 bin, V 4 bin, đã chuẩn hóa về tổng bằng 1 (16)

**Hình dạng (12 chiều), từ contour lớn nhất**

- aspect ratio: rộng / cao của bounding box
- extent: diện tích / diện tích bounding box
- solidity: diện tích / diện tích bao lồi
- circularity: \(4\pi A / P^2\)
- eccentricity: cạnh ngắn / cạnh dài của hình chữ nhật xoay nhỏ nhất
- 7 Hu moments (lấy log để cùng thang đo)

Trước khi đưa vào k-NN, mỗi chiều được chuẩn hóa z-score bằng mean/std tính trên tập train, để các đặc trưng có thang đo khác nhau đóng góp ngang nhau vào khoảng cách.

## k-NN

Cài trong [`knn/knn.py`](knn/knn.py):

- `euclidean_distances`, `manhattan_distances`: khoảng cách từ một mẫu đến toàn bộ tập train
- `knn_predict_one`: chọn k mẫu gần nhất bằng `argpartition`, vote theo đa số; nếu hòa thì chọn lớp có khoảng cách trung bình nhỏ hơn
- `stratified_split`: chia train/test giữ tỷ lệ mỗi lớp
- `standardize_fit`, `standardize_apply`: chuẩn hóa z-score
- `accuracy_score`, `confusion_matrix`: đánh giá

## Cấu trúc thư mục

```text
pet-classifier-knn/
├── app.py               # chạy desktop app
├── evaluate.py          # chạy khảo sát k / khoảng cách trên terminal
├── download_data.py     # tải và lọc bộ Oxford-IIIT Pet
├── config.py            # đường dẫn, danh sách giống, các giá trị k
├── requirements.txt
├── features/            # tiền xử lý, phân đoạn, trích đặc trưng
├── knn/                 # k-NN tự cài và script đánh giá
├── dataset/             # đọc thư mục data/ và cache đặc trưng
├── gui/                 # giao diện CustomTkinter
├── docs/                # hình dùng trong README
├── data/                # 5 giống × 200 ảnh (đầu vào train/test)
└── raw_data/            # bộ ảnh gốc Oxford-IIIT Pet (Git LFS)
```

## Yêu cầu

- Python 3.10+
- NumPy
- OpenCV (`opencv-python`)
- Pillow
- Matplotlib
- CustomTkinter

Cài đặt:

```bash
pip install -r requirements.txt
```

## Dữ liệu

Dùng bộ [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) của Đại học Oxford (37 giống, khoảng 7.400 ảnh). Project chỉ lấy 5 giống, mỗi giống 200 ảnh.

Thư mục `data/` đã có sẵn trong repo. Nếu muốn tải lại từ đầu:

```bash
python download_data.py
```

Script sẽ:

1. tải `images.tar.gz` vào `raw_data/` (có link dự phòng nếu link chính lỗi)
2. giải nén
3. copy ảnh của 5 giống đã chọn vào `data/<tên giống>/`

Repo cũng kèm file torrent `oxford-iiit-pet-b18bbd9ba03d50b0f7f479acc9f4228a408cecc1.torrent` để tải bằng BitTorrent khi server Oxford chậm.

> `raw_data/images.tar.gz` (~755 MB) được lưu bằng Git LFS. Khi clone, cài Git LFS trước hoặc chỉ cần thư mục `data/` là đủ để chạy.

## Đánh giá trên terminal

```bash
python evaluate.py
```

Script sẽ:

1. trích đặc trưng cho 1.000 ảnh (lần đầu mất khoảng 1–2 phút), lưu vào `features_cache.npz`
2. chia 800 train / 200 test
3. chạy k-NN với mọi cặp (khoảng cách, k)
4. in bảng accuracy, cặp tốt nhất và confusion matrix

Những lần chạy sau dùng lại cache nên gần như tức thì. Xóa `features_cache.npz` nếu đã sửa code trích đặc trưng.

## Desktop app

```bash
python app.py
```

**Tab Nhận dạng**

- **Mở ảnh…**: chọn một ảnh bất kỳ; app hiển thị ảnh gốc và vùng đối tượng sau phân đoạn
- thanh trượt **k** (1–11) và nút chọn **Euclidean / Manhattan**
- **Nhận dạng**: hiện giống dự đoán và danh sách k láng giềng gần nhất kèm khoảng cách

**Tab Đánh giá**

- **Trích đặc trưng & đánh giá**: chạy k-NN trên tập test với khoảng cách đang chọn
- **Tính lại đặc trưng**: bỏ qua cache và trích lại toàn bộ
- hiển thị accuracy, biểu đồ accuracy theo \(k\) và ma trận nhầm lẫn

Nếu chưa có cache, hãy chạy tab Đánh giá (hoặc `python evaluate.py`) trước khi nhận dạng.

## Hạn chế

- **Phân đoạn chưa ổn định.** GrabCut được khởi tạo bằng một hình chữ nhật cố định (cách mép 8%). Với ảnh như Pug ở hình đầu trang, mask chỉ lấy được phần mặt; với Sphynx thì lấy dính cả sàn. Mask sai làm lệch cả đặc trưng màu lẫn hình dạng.
- **Đặc trưng hình dạng phụ thuộc tư thế.** Con vật nằm, ngồi hay đứng cho contour rất khác nhau, nên Hu moments và aspect ratio dao động mạnh trong cùng một giống.
- **Chỉ một lần chia train/test.** Kết quả có thể thay đổi vài phần trăm khi đổi `seed`; cross-validation sẽ cho số liệu tin cậy hơn.
- **Chọn k trên tập test.** Bảng khảo sát dùng trực tiếp tập test để so sánh \(k\), nên 58% là ước lượng hơi lạc quan.

## Hướng phát triển

- thêm không gian màu Lab, histogram màu 2D (H–S) hoặc histogram mịn hơn
- khởi tạo GrabCut tốt hơn hoặc dùng mask có sẵn trong bộ Oxford (trimaps) để so sánh
- đánh trọng số cho từng nhóm đặc trưng trước khi tính khoảng cách
- k-fold cross-validation và tập validation riêng để chọn \(k\)
- vote có trọng số theo khoảng cách (\(1/d\))
