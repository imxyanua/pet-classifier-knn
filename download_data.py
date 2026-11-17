#code by xyanua - github.com/imxyanua
import os
import shutil
import tarfile
import urllib.request

RAW_DIR = "raw_data"
DATA_DIR = "data"
TAR_PATH = os.path.join(RAW_DIR, "images.tar.gz")

# Link tải trực tiếp chính thức từ Đại học Oxford 
URLS = [
    "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz",
    "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
]

SELECTED_BREEDS = [
    "pug",           # Chó Pug: lông vàng kem, mặt ngắn, thân tròn vuông
    "samoyed",       # Chó Samoyed: lông trắng muốt, xù
    "Bombay",        # Mèo Bombay: lông đen tuyền
    "Egyptian_Mau",  # Mèo Ai Cập: lông xám đốm vằn, tai nhọn
    "Sphynx",        # Mèo không lông: màu da hồng/xám, thân mảnh
]

def download_file():
    os.makedirs(RAW_DIR, exist_ok=True)
    if os.path.exists(TAR_PATH):
        print(f"Đã tìm thấy file {TAR_PATH}, bỏ qua bước tải.")
        return

    for url in URLS:
        try:
            print(f"Đang kết nối tới: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 1024 * 256  # 256 KB

                with open(TAR_PATH, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = downloaded * 100 / total_size
                            mb_done = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            print(f"\rĐang tải: {percent:5.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)", end="", flush=True)
            print("\nTải hoàn tất!")
            return
        except Exception as e:
            print(f"\nKhông tải được từ link này ({e}), đang thử link dự phòng...")
            if os.path.exists(TAR_PATH):
                os.remove(TAR_PATH)

    raise RuntimeError("Không thể tải dữ liệu từ cả hai máy chủ.")

def extract_and_filter():
    images_dir = os.path.join(RAW_DIR, "images")
    if not os.path.exists(images_dir):
        print("Đang giải nén images.tar.gz (mất khoảng 10-20 giây)...")
        with tarfile.open(TAR_PATH, "r:gz") as tar:
            tar.extractall(path=RAW_DIR)

    os.makedirs(DATA_DIR, exist_ok=True)
    counts = {breed: 0 for breed in SELECTED_BREEDS}

    for filename in os.listdir(images_dir):
        if not filename.lower().endswith(".jpg"):
            continue
        for breed in SELECTED_BREEDS:
            if filename.startswith(breed + "_"):
                breed_dir = os.path.join(DATA_DIR, breed)
                os.makedirs(breed_dir, exist_ok=True)
                shutil.copy2(
                    os.path.join(images_dir, filename),
                    os.path.join(breed_dir, filename)
                )
                counts[breed] += 1
                break

    print("\nHoàn tất chuẩn bị dữ liệu trong thư mục 'data/':")
    for breed, count in counts.items():
        print(f"  - {breed}: {count} ảnh")

if __name__ == "__main__":
    download_file()
    extract_and_filter()