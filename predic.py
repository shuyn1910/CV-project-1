import cv2
from ultralytics import YOLO

def main():
    # 1. Nạp trọng số mô hình tốt nhất
    model_path = "weights/best.pt"
    model = YOLO(model_path)

    # 2. Chọn cổng camera Iriun
    # 0: Thường là Webcam có sẵn của laptop
    # 1: Camera ảo Iriun (Nếu mở lên vẫn là webcam laptop thì đổi thành 2)
    cam_index = 1
    
    print(f"[*] Đang kết nối tới Camera Iriun (cổng {cam_index})...")
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"[-] Không thể mở camera tại cổng {cam_index}.")
        print("[*] Hãy thử đổi cam_index = 2 hoặc kiểm tra cửa sổ Iriun trên PC đã hiện hình chưa.")
        return

    # 3. Tạo cửa sổ hiển thị chuẩn kích thước
    window_name = "YOLOv8 Detection - Iriun Phone Camera"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    print("[+] Kết nối thành công! Đưa camera điện thoại quay Bàn phím / Chuột.")
    print("[*] Nhấn phím 'q' trên cửa sổ hiển thị để dừng chương trình.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[-] Mất tín hiệu khung hình từ camera.")
            break

        # 4. Dự đoán thời gian thực trên GPU (RTX)
        results = model.predict(source=frame, conf=0.65, device=0, verbose=False)

        # 5. Phủ khung Bounding Box và Nhãn đối tượng lên ảnh
        annotated_frame = results[0].plot()

        # 6. Hiển thị hình ảnh
        cv2.imshow(window_name, annotated_frame)

        # Thoát nếu nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()