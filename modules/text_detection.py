import paddle
import cv2
from paddleocr import PaddleOCR

# 1. OCR 모델 초기화
def initialize_ocr_model():

    print("GPU OCR 모델 초기화 중...")

    # PaddlePaddle GPU 확인
    if not paddle.is_compiled_with_cuda():
        raise RuntimeError(
            "현재 PaddlePaddle은 CPU 버전입니다. "
            "GPU 버전으로 설치해주세요."
        )

    gpu_count = paddle.device.cuda.device_count()

    if gpu_count == 0:
        raise RuntimeError(
            "사용 가능한 GPU가 없습니다."
        )

    print("사용 가능한 GPU 개수:", gpu_count)

    # GPU 0번 장치 사용
    paddle.device.set_device("gpu:0")

    # OCR 모델 초기화
    ocr_model = PaddleOCR(
        lang="korean",
        use_textline_orientation=True,
        device="gpu:0"
    )

    print("OCR GPU 초기화 완료!")

    return ocr_model

# 2. 이미지 텍스트 탐지
def detect_text(image, ocr_model):

    if image is None:
        raise ValueError("이미지가 없습니다.")

    # 이미지 크기 확인
    height, width = image.shape[:2]

    # OpenCV BGR 이미지를 PaddleOCR에 전달
    results = ocr_model.predict(image)

    text_results = []

    # 3. OCR 결과 정리
    for result in results:

        data = result.json

        # PaddleOCR 결과 구조 확인
        if callable(data):
            data = data()

        if isinstance(data, dict):
            data = data.get("res", data)

        if not isinstance(data, dict):
            continue

        texts = data.get("rec_texts", [])
        scores = data.get("rec_scores", [])
        boxes = data.get("rec_polys", [])

        # 텍스트, 신뢰도, 좌표 저장
        for text, score, box in zip(texts, scores, boxes):

            # 좌표를 정수형으로 변환
            points = [
                [int(point[0]), int(point[1])]
                for point in box
            ]

            # 텍스트 영역의 사각형 좌표 계산
            x_coords = [point[0] for point in points]
            y_coords = [point[1] for point in points]

            x1 = max(0, min(width, min(x_coords)))
            y1 = max(0, min(height, min(y_coords)))
            x2 = max(0, min(width, max(x_coords)))
            y2 = max(0, min(height, max(y_coords)))

            text_results.append({
                "text": str(text),
                "score": float(score),
                "bbox": (x1, y1, x2, y2),
                "polygon": points
            })

    return text_results

# 4. OCR 탐지 결과 표시
def draw_text_boxes(image, text_results):

    result_image = image.copy()

    for item in text_results:

        x1, y1, x2, y2 = item["bbox"]

        # 텍스트 영역 표시
        cv2.rectangle(
            result_image,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        # 탐지된 텍스트 표시
        label = f"{item['text']} ({item['score']:.2f})"

        cv2.putText(
            result_image,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1
        )

    return result_image