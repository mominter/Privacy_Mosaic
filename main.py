import cv2
import time
from pathlib import Path
from datetime import datetime

from modules.face_detection import initialize_face_model, detect_faces, resize_image
from modules.text_detection import initialize_ocr_model, detect_text
from modules.privacy_detection import detect_privacy
from modules.privacy_filter import apply_privacy_filter

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

STRENGTH = 12
CAMERA_WIDTH, CAMERA_HEIGHT, TARGET_FPS = 640, 480, 30
FACE_INTERVAL, OCR_INTERVAL = 2, 15
WINDOW_NAME = "Privacy Filter"
CLICK_PADDING, IOU_THRESHOLD = 8, 0.25
cv2.setNumThreads(2)

current_regions = []

def make_detections(face_results, privacy_results):
    detections = []
    for item in face_results:
        if isinstance(item, dict) and item.get("bbox") is not None:
            detections.append({"kind": "FACE", "bbox": tuple(map(int, item["bbox"])), "source": item})
    for item in privacy_results:
        if isinstance(item, dict) and item.get("bbox") is not None:
            detections.append({"kind": "PRIVACY", "bbox": tuple(map(int, item["bbox"])), "source": item})
    return detections

def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0

def update_tracks(detections, previous):
    updated, used = [], set()
    for det in detections:
        best_idx, best_score = None, 0.0
        for i, old in enumerate(previous):
            if i in used or old["kind"] != det["kind"]:
                continue
            score = box_iou(det["bbox"], old["bbox"])
            if score > best_score:
                best_idx, best_score = i, score
        if best_idx is not None and best_score >= IOU_THRESHOLD:
            old = previous[best_idx]
            used.add(best_idx)
            region_id, selected = old["id"], old["selected"]
        else:
            region_id, selected = None, False
        updated.append({
            "id": region_id, "kind": det["kind"], "bbox": det["bbox"],
            "source": det["source"], "selected": selected
        })

    next_id = max((r["id"] for r in previous), default=0) + 1
    for region in updated:
        if region["id"] is None:
            region["id"] = next_id
            next_id += 1
    return updated

def active_results(regions):
    faces, privacy = [], []
    for region in regions:
        if region["selected"]:
            continue
        if region["kind"] == "FACE":
            faces.append(region["source"])
        else:
            privacy.append(region["source"])
    return faces, privacy

def render_result(original, regions, draw_boxes=True):
    faces, privacy = active_results(regions)
    result = apply_privacy_filter(original, faces, privacy, strength=STRENGTH)
    if result is None:
        return None
    # 결과 이미지에는 탐지 영역의 테두리나 MOSAIC / UNMOSAIC 문구를 표시하지 않습니다.
    # 클릭을 통한 모자이크 해제/복구는 current_regions의 좌표를 이용하므로 그대로 동작합니다.
    return result

def mouse_callback(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    candidates = []
    for region in current_regions:
        x1, y1, x2, y2 = region["bbox"]
        if x1 - CLICK_PADDING <= x <= x2 + CLICK_PADDING and y1 - CLICK_PADDING <= y <= y2 + CLICK_PADDING:
            candidates.append((max(1, (x2 - x1) * (y2 - y1)), region))
    if not candidates:
        print("[Click] 탐지 영역 바깥입니다.")
        return
    _, selected = min(candidates, key=lambda x: x[0])
    selected["selected"] = not selected["selected"]
    print(f"[Click] {selected['kind']}: " +
          ("모자이크 해제" if selected["selected"] else "모자이크 복구"))
    refresh = param.get("refresh")
    if refresh:
        refresh()

def initialize_models():
    print("\n[모델] InsightFace 초기화 중...")
    face_model = initialize_face_model()
    print("[모델] InsightFace 준비 완료")
    print("\n[모델] PaddleOCR 초기화 중...")
    ocr_model = initialize_ocr_model()
    print("[모델] PaddleOCR 준비 완료")
    return face_model, ocr_model

def run_image_mode(face_model, ocr_model):
    global current_regions
    path_text = input(
        "\n이미지 경로 입력 (프로젝트 폴더의 soccer.jpg는 Enter): "
    ).strip().strip('"')
    image_path = Path(path_text) if path_text else BASE_DIR / "soccer.jpg"
    if not image_path.is_absolute():
        image_path = BASE_DIR / image_path
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[오류] 이미지를 읽을 수 없습니다: {image_path}")
        return

    print("[이미지] 얼굴 탐지 중...")
    faces, processed = detect_faces(image, face_model)
    print("[이미지] 텍스트 탐지 중...")
    texts = detect_text(processed, ocr_model)
    privacy = detect_privacy(texts)
    print(f"[탐지] 얼굴 {len(faces)}개 / 개인정보 {len(privacy)}개")

    current_regions = update_tracks(make_detections(faces, privacy), [])
    output_path = RESULT_DIR / f"{image_path.stem}_privacy_result.jpg"

    def refresh():
        result = render_result(processed, current_regions)
        cv2.imshow(WINDOW_NAME, result)
        if cv2.imwrite(str(output_path), result):
            print(f"[저장] {output_path}")
        else:
            print("[경고] 이미지 저장 실패")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, {"refresh": refresh})
    refresh()
    print("\n[이미지 모드] 클릭: 모자이크 해제/복구 | Q 또는 ESC: 종료")
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
    current_regions = []
    cv2.destroyWindow(WINDOW_NAME)

def open_camera():
    print("\n카메라 선택\n1. PC 웹캠\n2. 휴대폰 IP 카메라")
    choice = input("선택 (1/2): ").strip()
    if choice == "1":
        value = input("웹캠 번호 (기본값 0): ").strip()
        source = int(value) if value else 0
    elif choice == "2":
        source = input("휴대폰 스트림 URL (예: http://192.168.0.10:8080/video): ").strip()
        if not source:
            print("[오류] URL이 비어 있습니다.")
            return None
    else:
        print("[오류] 잘못된 선택입니다.")
        return None

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[오류] 카메라 연결 실패. 웹캠 번호 또는 휴대폰 스트림 URL을 확인하세요.")
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def run_camera_mode(face_model, ocr_model):
    global current_regions
    cap = open_camera()
    if cap is None:
        return

    frame_count = 0
    cached_faces, cached_privacy, tracks = [], [], []
    face_ms = ocr_ms = fps = 0.0
    previous_end = None
    start = time.perf_counter()

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, mouse_callback, {"refresh": None})
        print("\n[카메라 모드] 클릭: 모자이크 해제/복구 | S: 화면 저장 | Q/ESC: 종료")
        while True:
            frame_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[경고] 카메라 프레임을 받지 못했습니다.")
                break
            frame_count += 1

            if frame_count == 1 or (frame_count - 1) % FACE_INTERVAL == 0:
                t = time.perf_counter()
                cached_faces, processed = detect_faces(frame, face_model)
                face_ms = (time.perf_counter() - t) * 1000
            else:
                processed = resize_image(frame)

            if frame_count == 1 or (frame_count - 1) % OCR_INTERVAL == 0:
                t = time.perf_counter()
                texts = detect_text(processed, ocr_model)
                cached_privacy = detect_privacy(texts)
                ocr_ms = (time.perf_counter() - t) * 1000

            tracks = update_tracks(make_detections(cached_faces, cached_privacy), tracks)
            current_regions = tracks
            result = render_result(processed, tracks)
            if result is None:
                continue

            now = time.perf_counter()
            if previous_end is not None and now > previous_end:
                instant = 1.0 / (now - previous_end)
                fps = instant if fps == 0 else fps * 0.8 + instant * 0.2
            previous_end = now

            lines = [
                f"FPS: {fps:.1f} | Face: {len(cached_faces)} | Privacy: {len(cached_privacy)}",
                f"Face: {face_ms:.1f}ms | OCR(last): {ocr_ms:.1f}ms",
                "Click: Toggle mosaic | S: Save | Q/ESC: Quit",
            ]
            for i, line in enumerate(lines):
                cv2.putText(result, line, (10, 22 + i * 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, result)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = RESULT_DIR / f"camera_privacy_{stamp}.jpg"
                if cv2.imwrite(str(save_path), result):
                    print(f"[저장] {save_path}")
            if frame_count % 150 == 0:
                elapsed = time.perf_counter() - start
                print(f"[성능] 평균 FPS: {frame_count / elapsed:.2f} | "
                      f"최근 프레임: {(time.perf_counter() - frame_start) * 1000:.1f}ms")
            if key in (ord("q"), 27):
                break
    finally:
        current_regions = []
        cap.release()
        cv2.destroyAllWindows()
        elapsed = time.perf_counter() - start
        print(f"\n[카메라 종료] 프레임: {frame_count} | "
              f"평균 FPS: {frame_count / elapsed if elapsed else 0:.2f}")

def main():
    print("=" * 58)
    print("개인정보 보호 모자이크 시스템")
    print("=" * 58)
    print("1. 이미지 파일 모자이크")
    print("2. 실시간 카메라 모자이크 (웹캠/휴대폰)")
    print("0. 종료")
    mode = input("\n실행 모드 선택: ").strip()

    if mode == "0":
        return
    if mode not in ("1", "2"):
        print("[오류] 올바른 메뉴를 선택해주세요.")
        return

    try:
        face_model, ocr_model = initialize_models()
        if mode == "1":
            run_image_mode(face_model, ocr_model)
        else:
            run_camera_mode(face_model, ocr_model)
    except KeyboardInterrupt:
        print("\n사용자 요청으로 종료합니다.")
    except Exception as exc:
        print(f"\n[오류] {type(exc).__name__}: {exc}")
        print("모델 설치 상태와 modules 폴더의 함수 구성을 확인해주세요.")
    finally:
        cv2.destroyAllWindows()
    print("프로그램 종료")

if __name__ == "__main__":
    main()