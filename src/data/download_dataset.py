import os
import subprocess
import shutil

def download_image_verification_corpus():
    print("--- BƯỚC 1: KIỂM TRA THƯ MỤC CHỨA DỮ LIỆU ---")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data', 'raw')
    os.makedirs(data_dir, exist_ok=True)
    
    repo_dir = os.path.join(data_dir, 'image-verification-corpus')
    
    print("\n--- BƯỚC 2: TẢI DATASET TỪ GITHUB (IMAGE VERIFICATION CORPUS) ---")
    if os.path.exists(repo_dir):
        print(f"Dataset đã được tải sẵn tại: {repo_dir}")
        print("Đang dọn dẹp để tải lại từ đầu...")
        shutil.rmtree(repo_dir, ignore_errors=True)
        
    print("Đang tải dữ liệu chuẩn xác 100% từ Github của cô giáo...")
    print("Vui lòng chờ (dung lượng khoảng 50MB)...")
    
    try:
        # Sử dụng lệnh git clone để tải toàn bộ repo
        subprocess.run(["git", "clone", "https://github.com/MKLab-ITI/image-verification-corpus.git", repo_dir], check=True)
        print("✅ Đã tải thành công kho dữ liệu!")
    except Exception as e:
        print(f"❌ Lỗi khi tải dữ liệu: {e}")
        print("Lưu ý: Máy tính của bạn cần được cài đặt 'git' trước để chạy lệnh này.")
        return

    print("\n--- BƯỚC 3: KIỂM TRA CÁC FILE QUAN TRỌNG ---")
    devset_dir = os.path.join(repo_dir, 'mediaeval2015', 'devset')
    tweets_file = os.path.join(devset_dir, 'tweets.txt')
    images_rar = os.path.join(devset_dir, 'MediaEval2015_DevSet_Images.rar')
    
    if os.path.exists(tweets_file) and os.path.exists(images_rar):
        print(f"✅ Đã tìm thấy file văn bản: {tweets_file}")
        print(f"✅ Đã tìm thấy file hình ảnh nén: {images_rar}")
        print("\nTẤT CẢ DỮ LIỆU ĐÃ SẴN SÀNG!")
        print(f"Thư mục lưu trữ: {repo_dir}")
    else:
        print("❌ Cảnh báo: Không tìm thấy các file dữ liệu bên trong repo.")

if __name__ == "__main__":
    download_image_verification_corpus()
