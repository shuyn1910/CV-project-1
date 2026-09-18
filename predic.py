import cv2
from ultralytics import YOLO
import platform

def main():
  # 1. Nạp trọng số mô hình tốt nhất
  model_path = "weights/best.pt"
  model = YOLO(model_path)

  cam_index = 2  # 0 là webcam mặc định

  if platform.system() == "Windows":
      print(f"[*] Chạy trên Windows: Mở Camera (cổng {cam_index}) qua DirectShow...")
      cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
  else:
      print(
          f"[*] Chạy trên Linux/Jetson: Mở Camera (/dev/video{cam_index}) qua"
          " V4L2..."
      )
      cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
  if not cap.isOpened():
    print(f"[-] Không thể mở camera tại cổng /dev/video{cam_index}.")
    print(
        "[*] Hãy kiểm tra lại lệnh: ls /dev/video* hoặc quyền sudo usermod -aG"
        " video $USER"
    )
    return

  # Khóa độ phân giải camera ở mức 640x480 để xử lý mượt và tăng FPS
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

  # 3. Tạo cửa sổ hiển thị
  window_name = "YOLOv8 Detection - Jetson Nano"
  cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
  cv2.resizeWindow(window_name, 640, 480)

  print("[+] Kết nối camera thành công!")
  print("[*] Nhấn phím 'q' trên cửa sổ hiển thị để dừng chương trình.")

  while True:
    ret, frame = cap.read()
    if not ret:
      print("[-] Mất tín hiệu khung hình từ camera.")
      break

    # 4. Dự đoán thời gian thực
    # Lưu ý: Đổi device='cpu' nếu Jetson Nano chưa cài đặt PyTorch build riêng cho CUDA
    results = model.predict(source=frame, conf=0.65, device=0, verbose=False)

    # 5. Phủ khung Bounding Box và Nhãn đối tượng lên ảnh
    annotated_frame = results[0].plot()

    # 6. Hiển thị hình ảnh
    cv2.imshow(window_name, annotated_frame)

    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()