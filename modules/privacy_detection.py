import re

# 1. 전화번호 정규식
PHONE_PATTERN = re.compile(
    r"(?<!\d)"
    r"(?:01[016789]|02|0[3-6]\d|070)"
    r"[\s().-]*\d{3,4}"
    r"[\s().-]*\d{4}"
    r"(?!\d)"
)

# 2. 이메일 정규식
EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+-]+"
    r"@[a-zA-Z0-9.-]+"
    r"\.[a-zA-Z]{2,}"
)

# 3. 개인정보 탐지
def detect_privacy(text_results):

    if not isinstance(text_results, list):
        raise TypeError("text_results는 리스트 형태여야 합니다.")

    privacy_results = []

    for item in text_results:

        # OCR 결과 데이터 확인
        if not isinstance(item, dict):
            continue

        text = item.get("text", "")

        if not isinstance(text, str) or not text.strip():
            continue

        # OCR에서 전달받은 좌표와 신뢰도 유지
        bbox = item.get("bbox")

        if not bbox or len(bbox) != 4:
            continue

        score = item.get("score", 0.0)

        # 전화번호 탐지
        phone_matches = PHONE_PATTERN.finditer(text)

        for match in phone_matches:

            privacy_results.append({
                "type": "phone_number",
                "text": text,
                "matched_text": match.group(),
                "score": float(score),
                "bbox": tuple(bbox)
            })

        # 이메일 탐지
        # OCR에서 이메일 중간에 공백이 들어가는 경우 보정
        normalized_text = re.sub(r"\s+", "", text)

        email_matches = EMAIL_PATTERN.finditer(normalized_text)

        for match in email_matches:

            privacy_results.append({
                "type": "email",
                "text": text,
                "matched_text": match.group(),
                "score": float(score),
                "bbox": tuple(bbox)
            })

    return privacy_results