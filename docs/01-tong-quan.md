# 1. Tổng quan

## Bài toán

Đề 34: *Nhận dạng đối tượng bằng k-NN dựa trên đặc trưng ảnh*.

> Tiền xử lý và phân đoạn đối tượng; trích xuất một số đặc trưng màu và hình dạng; cài đặt k-NN và khảo sát ảnh hưởng của $k$ hoặc khoảng cách. Đánh giá bằng độ chính xác và ma trận nhầm lẫn.

Đối tượng được chọn là thú cưng: cho một ảnh chó hoặc mèo, dự đoán nó thuộc giống nào trong 5 giống.

Điểm cốt lõi của đề là **đặc trưng do mình tự thiết kế** chứ không phải đặc trưng do mạng neuron tự học. k-NN không có tham số để huấn luyện, nên chất lượng nhận dạng phụ thuộc gần như hoàn toàn vào việc vector đặc trưng có tách được các giống hay không.

## Dữ liệu

Nguồn: [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), 37 giống, khoảng 7.400 ảnh. Project lấy 5 giống, mỗi giống 200 ảnh, tổng 1.000 ảnh.

| Giống | Loài | Đặc điểm có thể phân biệt bằng màu / hình dạng |
|---|---|---|
| Pug | Chó | Lông vàng kem, mặt đen, thân tròn |
| Samoyed | Chó | Lông trắng, xù, sáng |
| Bombay | Mèo | Lông đen tuyền |
| Egyptian Mau | Mèo | Lông xám bạc, đốm |
| Sphynx | Mèo | Không lông, da hồng/xám, thân mảnh |

Ảnh trong bộ này là ảnh chụp đời thường: nền lộn xộn, con vật ở nhiều tư thế, có ảnh chỉ thấy mặt, có ảnh có người hoặc đồ vật. Đây là lý do bước phân đoạn quan trọng và cũng là nguồn lỗi lớn nhất.

## Cấu trúc code

```text
config.py              đường dẫn, danh sách giống, các giá trị k, tên hiển thị
download_data.py       tải images.tar.gz, giải nén, lọc 5 giống vào data/

features/features.py   load ảnh, tiền xử lý, phân đoạn, trích đặc trưng
dataset/dataset.py     duyệt data/<giống>/, trích đặc trưng hàng loạt, cache .npz
knn/knn.py             khoảng cách, chuẩn hóa, chia tập, k-NN, accuracy, confusion matrix
knn/evaluate.py        khảo sát mọi cặp (khoảng cách, k), in bảng kết quả
gui/app.py             desktop app CustomTkinter

app.py, evaluate.py    điểm chạy ở thư mục gốc, chỉ gọi vào package tương ứng
```

Các module phụ thuộc theo một chiều, không vòng:

```text
config ──► features ──► dataset ──► knn/evaluate, gui
                 knn ───────────────┘
```

`knn/knn.py` không phụ thuộc vào xử lý ảnh: nó chỉ nhận ma trận số. Nhờ vậy có thể kiểm tra thuật toán k-NN độc lập với dữ liệu thật.

## Luồng dữ liệu

```text
data/<giống>/*.jpg
   │  load_bgr
   ▼
ảnh BGR
   │  preprocess          resize cạnh dài 192px, Gaussian 5×5
   │  segment_pet         GrabCut → morphology → giữ vùng lớn nhất
   ▼
(ảnh 192px, mask nhị phân)
   │  extract_features    màu HSV trong mask + hình dạng contour
   ▼
vector 34 chiều
   │  load_or_extract     lặp 1.000 ảnh, lưu features_cache.npz
   ▼
X (1000×34), y (1000)
   │  stratified_split    800 train / 200 test, giữ tỷ lệ lớp
   │  standardize_fit     mean, std tính trên train
   ▼
X_train, X_test đã chuẩn hóa
   │  knn_predict_batch   với mỗi (khoảng cách, k)
   ▼
dự đoán ──► accuracy_score, confusion_matrix
```

## Thư viện dùng ở đâu

| Thư viện | Dùng cho | Không dùng cho |
|---|---|---|
| OpenCV | đọc ảnh, resize, blur, HSV, GrabCut, morphology, contour, moments | phân loại |
| NumPy | phép toán ma trận, histogram | (thuật toán tự viết bằng NumPy) |
| Matplotlib | vẽ biểu đồ, ma trận nhầm lẫn | tính toán |
| CustomTkinter | giao diện | — |
| scikit-learn | **không dùng** | k-NN, chuẩn hóa, chia tập, metric đều tự cài |

## Cache đặc trưng

Trích đặc trưng 1.000 ảnh mất khoảng 1–2 phút, chủ yếu do GrabCut. Kết quả được lưu vào `features_cache.npz` (gồm `X`, `y`, `paths`, `version`) ở thư mục gốc.

Cache **không tự cập nhật** khi sửa code trong `features/`. Sau khi sửa đặc trưng, cần xóa file cache, nhấn "Tính lại đặc trưng" trong app, hoặc tăng `CACHE_VERSION` trong `dataset/dataset.py`.
