import argparse
import os
import cv2
import numpy as np

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save(path, img):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]:
        cv2.imwrite(path, img)
    else:
        cv2.imwrite(path + ".jpg", img)

def expand_bbox(x, y, w, h, scale: float, W: int, H: int):
    dx = int(round(w * scale))
    dy = int(round(h * scale))
    x2 = max(0, x - dx)
    y2 = max(0, y - dy)
    x3 = min(W, x + w + dx)
    y3 = min(H, y + h + dy)
    return x2, y2, (x3 - x2), (y3 - y2)

def detect_face_bbox(img_bgr):
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        raise RuntimeError("Не удалось загрузить каскад Хаара. Проверьте установку OpenCV.")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60, 60))
    if len(faces) == 0:
        gray_eq = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=5,
                                              flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60, 60))
    if len(faces) == 0:
        raise RuntimeError("Лицо не найдено. Попробуйте другое фото или освещение.")
    areas = [w*h for (x, y, w, h) in faces]
    idx = int(np.argmax(areas))
    return faces[idx]  
def remove_small_edges(edge_img, min_w=10, min_h=10):
    # edge_img — бинарное (0/255)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((edge_img > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(edge_img)
    for i in range(1, num_labels): 
        x, y, w, h, area = stats[i]
        if w >= min_w and h >= min_h:
            out[labels == i] = 255
    return out

def add_corners_to_edges(edges, corners_xy):
    # edges — single channel 0/255
    out = edges.copy()
    for (cx, cy) in corners_xy:
        if 0 <= cy < out.shape[0] and 0 <= cx < out.shape[1]:
            cv2.circle(out, (cx, cy), 2, 255, -1, lineType=cv2.LINE_AA)
    return out

def gaussian_mask_from_binary(bin_img, ksize=5):
    # 0/255 в 0..1, Гауссом
    mask = (bin_img.astype(np.float32) / 255.0)
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    mask = np.clip(mask, 0.0, 1.0)
    return mask

def smooth_F1(face_bgr):
    f = cv2.bilateralFilter(face_bgr, d=7, sigmaColor=75, sigmaSpace=75)
    f = cv2.GaussianBlur(f, (5, 5), 0)
    return f

def enhance_F2(face_bgr):
    # CLAHE по L-каналу + unsharp masking
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # Unsharp mask
    blur = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    usm = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)
    return usm

def blend_per_formula(M, F1, F2):
    # F1/F2: HxWx3 (uint8)
    F1f = F1.astype(np.float32)
    F2f = F2.astype(np.float32)
    M3 = np.dstack([M, M, M]).astype(np.float32)
    R = M3 * F2f + (1.0 - M3) * F1f
    return np.clip(R, 0, 255).astype(np.uint8)

def seamless_back(original_bgr, face_proc_bgr, bbox):
    x, y, w, h = bbox
    center = (x + w // 2, y + h // 2)
    mask = np.ones((h, w), dtype=np.uint8) * 255

    if face_proc_bgr.shape[0] != h or face_proc_bgr.shape[1] != w:
        face_proc_bgr = cv2.resize(face_proc_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
    mixed = cv2.seamlessClone(face_proc_bgr, original_bgr, mask, center, cv2.MIXED_CLONE)
    return mixed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Путь к исходному изображению (jpg/png и т.п.)")
    parser.add_argument("--out", default="out", help="Папка для сохранения результатов")
    parser.add_argument("--canny_low", type=int, default=80, help="Нижний порог Canny")
    parser.add_argument("--canny_high", type=int, default=160, help="Верхний порог Canny")
    parser.add_argument("--min_edge_w", type=int, default=10, help="Мин. ширина компонента для сохранения")
    parser.add_argument("--min_edge_h", type=int, default=10, help="Мин. высота компонента для сохранения")
    parser.add_argument("--dilate_ksize", type=int, default=5, help="Размер ядра dilation (квадрат)")
    args = parser.parse_args()

    ensure_dir(args.out)

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Не удалось открыть изображение: {args.input}")

    H, W = img.shape[:2]
    save(os.path.join(args.out, "00_original.jpg"), img)

    # 1) Обнаружение лица и расширение bbox на 10%
    x, y, w, h = detect_face_bbox(img)
    x2, y2, w2, h2 = expand_bbox(x, y, w, h, scale=0.10, W=W, H=H)

    img_bbox = img.copy()
    cv2.rectangle(img_bbox, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
    save(os.path.join(args.out, "01_face_bbox.jpg"), img_bbox)

    face_roi = img[y2:y2 + h2, x2:x2 + w2].copy()
    save(os.path.join(args.out, "02_face_crop.jpg"), face_roi)

    # 2) Края Canny на фрагменте лица
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_face, args.canny_low, args.canny_high, L2gradient=True)
    save(os.path.join(args.out, "03_edges_canny.png"), edges)


    edges_clean = remove_small_edges(edges, min_w=args.min_edge_w, min_h=args.min_edge_h)
    save(os.path.join(args.out, "04_edges_cleaned.png"), edges_clean)

    # 3) Угловые точки на исходном изображении
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray_full, maxCorners=500, qualityLevel=0.01, minDistance=5, blockSize=7)
    corners_xy = []
    if corners is not None:
        corners = np.int32(corners)
        for c in corners:
            cx, cy = c.ravel()
          
            if (x2 <= cx < x2 + w2) and (y2 <= cy < y2 + h2):
                corners_xy.append((cx - x2, cy - y2)) 

    edges_plus_corners = add_corners_to_edges(edges_clean, corners_xy)
    save(os.path.join(args.out, "05_edges_plus_corners.png"), edges_plus_corners)

    # 4) Морфологическое наращивание
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (args.dilate_ksize, args.dilate_ksize))
    dilated = cv2.dilate(edges_plus_corners, k, iterations=1)
    save(os.path.join(args.out, "06_dilated.png"), dilated)

    # 5) Гауссово сглаживание
    M = gaussian_mask_from_binary(dilated, ksize=5)
    # визуализация маски
    save(os.path.join(args.out, "07_mask_M_float.png"), (M * 255.0).astype(np.uint8))

    # 6) Обработка лица: F1 (сглаженное) и F2 (повышенная резкость/контраст)
    F1 = smooth_F1(face_roi)
    save(os.path.join(args.out, "08_F1_smooth.jpg"), F1)

    F2 = enhance_F2(face_roi)
    save(os.path.join(args.out, "09_F2_enhanced.jpg"), F2)

    # 7) Result = M*F2 + (1-M)*F1
    result_face = blend_per_formula(M, F1, F2)
    save(os.path.join(args.out, "10_result_face.jpg"), result_face)

    seamless = seamless_back(img, result_face, (x2, y2, w2, h2))
    save(os.path.join(args.out, "11_seamless_back.jpg"), seamless)

    print("Готово. Результаты сохранены в:", os.path.abspath(args.out))

if __name__ == "__main__":
    main()
