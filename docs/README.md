# Tài liệu dự án

Bộ tài liệu này giải thích chi tiết từng bước của project, lý do chọn cách làm, cách đọc kết quả và hướng cải thiện. README ở thư mục gốc chỉ tóm tắt và hướng dẫn chạy; các file ở đây đi sâu hơn.

Nên đọc theo thứ tự:

| # | File | Nội dung |
|---|---|---|
| 1 | [01-tong-quan.md](01-tong-quan.md) | Bài toán, dữ liệu, cấu trúc code, luồng dữ liệu |
| 2 | [02-tien-xu-ly-va-phan-doan.md](02-tien-xu-ly-va-phan-doan.md) | Resize, làm mờ, GrabCut, hình thái học, chấm điểm phân đoạn bằng trimap |
| 3 | [03-dac-trung.md](03-dac-trung.md) | Từng đặc trưng màu và hình dạng, ý nghĩa và công thức |
| 4 | [04-knn.md](04-knn.md) | Thuật toán k-NN, khoảng cách, chuẩn hóa, vote |
| 5 | [05-danh-gia.md](05-danh-gia.md) | Cách đánh giá, kết quả, đọc ma trận nhầm lẫn, thí nghiệm bổ sung |
| 6 | [06-huong-cai-thien.md](06-huong-cai-thien.md) | Các điểm yếu đã đo được và hướng cải thiện theo thứ tự ưu tiên |

Hình minh họa dùng trong tài liệu:

- [`segmentation.png`](segmentation.png): ảnh gốc, mask, ảnh chồng mask cho mỗi giống
- [`segmentation_before_after.png`](segmentation_before_after.png): mask của cách khởi tạo GrabCut cũ và mới, các ảnh thay đổi nhiều nhất
- [`accuracy_k.png`](accuracy_k.png): accuracy cross-validation trên tập train theo $k$, Euclidean và Manhattan, `seed=42`
- [`confusion_matrix.png`](confusion_matrix.png): ma trận nhầm lẫn trên test với cấu hình CV chọn (Manhattan, vote theo khoảng cách, $k=15$, `seed=42`)
