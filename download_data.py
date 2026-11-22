# code by xyanua - github.com/imxyanua
import argparse
import os
import shutil
import tarfile
import urllib.request

from config import (
    ANNOTATIONS_TAR_PATH,
    DATA_DIR,
    RAW_DIR,
    SELECTED_BREEDS,
    TAR_PATH,
    TRIMAP_DIR,
)

URLS = [
    "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz",
    "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
]
ANNOTATION_URLS = [
    "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz",
    "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
]


def download_file(urls=URLS, dest=TAR_PATH):
    os.makedirs(RAW_DIR, exist_ok=True)
    if os.path.exists(dest):
        print(f"Đã tìm thấy file {dest}, bỏ qua bước tải.")
        return

    for url in urls:
        try:
            print(f"Đang kết nối tới: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 1024 * 256

                with open(dest, "wb") as f:
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
                            print(
                                f"\rĐang tải: {percent:5.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)",
                                end="",
                                flush=True,
                            )
            print("\nTải hoàn tất!")
            return
        except Exception as e:
            print(f"\nKhông tải được từ link này ({e}), đang thử link dự phòng...")
            if os.path.exists(dest):
                os.remove(dest)

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
                    os.path.join(breed_dir, filename),
                )
                counts[breed] += 1
                break

    print("\nHoàn tất chuẩn bị dữ liệu trong thư mục 'data/':")
    for breed, count in counts.items():
        print(f"  - {breed}: {count} ảnh")


def extract_trimaps():
    """Giải nén trimap (mask do người gán nhãn), chỉ dùng để chấm điểm phân đoạn."""
    if os.path.isdir(TRIMAP_DIR) and os.listdir(TRIMAP_DIR):
        print(f"Đã có trimap trong {TRIMAP_DIR}, bỏ qua bước giải nén.")
        return
    with tarfile.open(ANNOTATIONS_TAR_PATH, "r:gz") as tar:
        members = [
            m for m in tar.getmembers()
            if m.name.startswith("annotations/trimaps/")
            and not os.path.basename(m.name).startswith("._")
        ]
        tar.extractall(path=RAW_DIR, members=members)
    print(f"Đã giải nén {len(members)} trimap vào {TRIMAP_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(description="Tải và lọc bộ Oxford-IIIT Pet.")
    parser.add_argument(
        "--trimaps",
        action="store_true",
        help="tải thêm trimap để chấm điểm phân đoạn bằng evaluate_segmentation.py",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_file()
    extract_and_filter()
    if args.trimaps:
        download_file(ANNOTATION_URLS, ANNOTATIONS_TAR_PATH)
        extract_trimaps()
