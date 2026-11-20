import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

def initialize_face_model():
    providers = ort.get_available_providers()

    if "CUDAExecutionProvider" in providers:
        selected_providers = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]
        ctx_id = 0
        print("[Face] GPU 모드")
    else:
        selected_providers = ["CPUExecutionProvider"]
        ctx_id = -1
        print("[Face] CPU 모드")

    model = FaceAnalysis(
        name="buffalo_l",
        providers=selected_providers
    )

    model.prepare(
        ctx_id=ctx_id,
        det_size=(640, 640)
    )

    return model

def resize_image(image, target_size=640):
    """원본 비율을 유지하며 정사각형 letterbox 이미지로 변환합니다."""
    if image is None:
        raise ValueError("이미지를 불러오지 못했습니다.")

    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("이미지 크기가 올바르지 않습니다.")

    # 두 변 모두 기준 크기 이하라면 원본 크기를 유지합니다.
    if width <= target_size and height <= target_size:
        return image

    # 원본 비율을 유지하면서 긴 변만 target_size에 맞춥니다.
    scale = target_size / max(width, height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation
    )

    # scale = min(target_size / width, target_size / height)
    # new_width = max(1, int(round(width * scale)))
    # new_height = max(1, int(round(height * scale)))

    # interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    # resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    # canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    # x_offset = (target_size - new_width) // 2
    # y_offset = (target_size - new_height) // 2
    # canvas[
    #     y_offset:y_offset + new_height,
    #     x_offset:x_offset + new_width
    # ] = resized

    # return canvas

def detect_faces(image, model):
    """얼굴 탐지 결과와 모델 입력용 640x640 이미지를 반환합니다."""
    if image is None:
        raise ValueError("이미지가 없습니다.")

    resized_image = resize_image(image)
    faces = model.get(resized_image)

    face_results = []
    image_height, image_width = resized_image.shape[:2]
    
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)

        x1 = max(0, min(640, x1))
        y1 = max(0, min(640, y1))
        x2 = max(0, min(640, x2))
        y2 = max(0, min(640, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        face_results.append({
            "bbox": (x1, y1, x2, y2),
            "score": float(face.det_score)
        })

    return face_results, resized_image

def draw_face_boxes(image, face_results):
    result_image = image.copy()
    for face in face_results:
        x1, y1, x2, y2 = face["bbox"]
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            result_image,
            f"Face: {face['score']:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
    return result_image