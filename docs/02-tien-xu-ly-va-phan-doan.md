# 2. Tiền xử lý và phân đoạn

Code: [`features/features.py`](../features/features.py), các hàm `load_bgr`, `preprocess`, `segment_pet`.

## Đọc ảnh

```python
data = np.fromfile(path, dtype=np.uint8)
img = cv2.imdecode(data, cv2.IMREAD_COLOR)
```

Dùng `np.fromfile` + `cv2.imdecode` thay cho `cv2.imread` vì `cv2.imread` trên Windows không đọc được đường dẫn có ký tự Unicode (tên thư mục tiếng Việt chẳng hạn).

## Tiền xử lý

1. **Resize** sao cho cạnh dài bằng 192px, giữ nguyên tỷ lệ khung.
   - Ảnh gốc có kích thước rất khác nhau (từ khoảng 200px đến hơn 1000px). Đưa về cùng tỷ lệ giúp các đặc trưng phụ thuộc kích thước (như số pixel trong mask) so sánh được với nhau.
   - 192px là điểm cân bằng giữa tốc độ và chất lượng: GrabCut ở 256px mất khoảng 0.38 giây/ảnh, ở 192px khoảng 0.12 giây/ảnh.
2. **Gaussian blur 5×5** để giảm nhiễu và chi tiết lông quá nhỏ, giúp GrabCut ổn định hơn và histogram màu mượt hơn.

## Phân đoạn bằng GrabCut

Mục tiêu: tạo mask nhị phân, 255 ở pixel thuộc con vật, 0 ở pixel nền. Mọi đặc trưng sau đó chỉ tính trên vùng mask.

### Ý tưởng GrabCut

GrabCut mô hình hóa màu của tiền cảnh và hậu cảnh bằng hai mô hình hỗn hợp Gauss (GMM), rồi dùng **graph cut** để gán nhãn từng pixel sao cho:

- pixel có màu giống GMM tiền cảnh thì được gán tiền cảnh,
- pixel cạnh nhau có màu giống nhau thì nên cùng nhãn (ràng buộc làm trơn).

Thuật toán lặp: gán nhãn → học lại GMM → gán nhãn lại.

### Khởi tạo

GrabCut cần biết đâu chắc chắn là nền. Project dùng một hình chữ nhật cách mỗi mép ảnh 8%:

```python
mx, my = int(w * 0.08), int(h * 0.08)
rect = (mx, my, w - 2 * mx, h - 2 * my)
cv2.grabCut(img, mask, rect, bgd, fgd, 1, cv2.GC_INIT_WITH_RECT)
```

Mọi pixel ngoài hình chữ nhật bị coi là nền chắc chắn. Pixel bên trong được GrabCut quyết định. Giả định ngầm: **con vật nằm gần giữa ảnh và không chạm mép**.

Chỉ chạy 1 vòng lặp để giữ tốc độ. Pixel có nhãn `GC_FGD` (chắc chắn tiền cảnh) hoặc `GC_PR_FGD` (có thể là tiền cảnh) được giữ lại.

Nếu GrabCut lỗi (ví dụ ảnh quá nhỏ), mask mặc định là toàn bộ ảnh.

### Hậu xử lý bằng hình thái học

Mask thô từ GrabCut thường có lỗ thủng và đốm nhiễu. Dùng phần tử cấu trúc hình elip 5×5:

1. **Đóng (close)** 2 lần: giãn rồi co, lấp các lỗ nhỏ bên trong con vật (ví dụ mắt, vùng lông tối).
2. **Mở (open)** 1 lần: co rồi giãn, xóa các đốm nhỏ lẻ ở nền.
3. **Giữ contour lớn nhất** và tô đặc nó. Bỏ mọi vùng rời rạc khác.

Bước 3 giả định ảnh chỉ có một con vật chính, đúng với hầu hết ảnh trong bộ Oxford-IIIT Pet.

## Kết quả và giới hạn

![Phân đoạn](segmentation.png)

Mỗi hàng: ảnh sau tiền xử lý, mask, ảnh chồng mask (vùng xanh là tiền cảnh). Thứ tự từ trên xuống: Pug, Samoyed, Bombay, Egyptian Mau, Sphynx.

| Ảnh | Kết quả | Lý do |
|---|---|---|
| Pug | Chỉ lấy được phần mặt, mất cả thân | Thân màu kem gần với màu tường phía sau nên bị GMM gán là nền |
| Samoyed | Tốt | Lông trắng tương phản rõ với nền trời và đất |
| Bombay | Rất tốt | Mèo đen trên nền trắng |
| Egyptian Mau | Tốt, giữ được cả chân và đuôi | Nền tối đồng nhất |
| Sphynx | Lấy dính một phần sàn | Da hồng/xám gần với màu thảm |

Quy luật chung: GrabCut làm tốt khi **màu con vật khác hẳn màu nền**, và thất bại khi chúng gần nhau. Đây không phải lỗi cài đặt mà là giới hạn của phương pháp dựa trên màu.

Hệ quả cho các bước sau:

- mask thiếu phần thân thì đặc trưng màu chỉ phản ánh phần mặt,
- mask dính nền thì histogram màu bị pha màu nền,
- đặc trưng hình dạng bị ảnh hưởng nặng nhất vì nó đo trực tiếp đường viền của mask.

Xem [06-huong-cai-thien.md](06-huong-cai-thien.md) cho cách đo và cải thiện chất lượng phân đoạn.
