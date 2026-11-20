import cv2


def _clip_bbox(image_shape, bbox):
    height, width = image_shape[:2]
    if bbox is None or len(bbox) != 4:
        return None

    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x1 >= x2 or y1 >= y2:
        return None
    return x1, y1, x2, y2


def apply_mosaic(image, bbox, strength=12):
    """지정 영역에 모자이크를 적용합니다. 입력 이미지를 직접 수정합니다."""
    if image is None:
        raise ValueError("이미지가 없습니다.")
    if not isinstance(strength, int) or strength < 2:
        raise ValueError("모자이크 강도는 정수 2 이상이어야 합니다.")

    clipped = _clip_bbox(image.shape, bbox)
    if clipped is None:
        return image

    x1, y1, x2, y2 = clipped
    roi = image[y1:y2, x1:x2]
    roi_height, roi_width = roi.shape[:2]

    small_width = max(1, roi_width // strength)
    small_height = max(1, roi_height // strength)

    small = cv2.resize(
        roi, (small_width, small_height),
        interpolation=cv2.INTER_AREA
    )
    mosaic = cv2.resize(
        small, (roi_width, roi_height),
        interpolation=cv2.INTER_NEAREST
    )
    image[y1:y2, x1:x2] = mosaic
    return image


def apply_privacy_filter(image, face_results, privacy_results, strength=12):
    """얼굴 및 개인정보 영역에 모자이크를 적용합니다."""
    if image is None:
        raise ValueError("이미지가 없습니다.")
    if not isinstance(face_results, list):
        raise TypeError("face_results는 리스트 형태여야 합니다.")
    if not isinstance(privacy_results, list):
        raise TypeError("privacy_results는 리스트 형태여야 합니다.")
    if not isinstance(strength, int) or strength < 2:
        raise ValueError("모자이크 강도는 정수 2 이상이어야 합니다.")

    # 전체 이미지 복사는 여기서 한 번만 수행합니다.
    result_image = image.copy()

    for item in face_results:
        if isinstance(item, dict):
            apply_mosaic(result_image, item.get("bbox"), strength)

    for item in privacy_results:
        if isinstance(item, dict):
            apply_mosaic(result_image, item.get("bbox"), strength)

    return result_image
