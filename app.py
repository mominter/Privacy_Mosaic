import io
import hashlib

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from modules.face_detection import initialize_face_model, detect_faces
from modules.text_detection import detect_text
from modules.privacy_detection import detect_privacy
from modules.privacy_filter import apply_privacy_filter

# ---------------------------------------------------------
# Streamlit 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="개인정보 보호 모자이크",
    page_icon="🔒",
    layout="wide",
)

MOSAIC_STRENGTH = 12

# ---------------------------------------------------------
# 모델 초기화
# ---------------------------------------------------------
@st.cache_resource(show_spinner="얼굴 탐지 모델을 불러오는 중입니다...")
def load_face_model():
    return initialize_face_model()

@st.cache_resource(show_spinner="텍스트 인식 모델을 불러오는 중입니다...")
def load_ocr_model():
    import paddle
    from paddleocr import PaddleOCR

    if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
        paddle.device.set_device("gpu:0")
        device = "gpu:0"
        st.info("PaddleOCR 실행 장치: GPU")
    else:
        device = "cpu"
        st.info("PaddleOCR 실행 장치: CPU")

    return PaddleOCR(
        lang="korean",
        use_textline_orientation=True,
        device=device,
    )

# ---------------------------------------------------------
# 이미지 변환 및 처리
# ---------------------------------------------------------
def decode_image(image_bytes):
    """업로드/카메라 이미지 바이트를 OpenCV BGR 이미지로 변환합니다."""
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    rgb_image = np.array(pil_image)
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

def process_image(image_bgr, face_model, ocr_model):
    """얼굴/텍스트를 탐지하고 개인정보 영역에 모자이크를 적용할 수 있도록 결과를 만듭니다."""
    face_results, processed_image = detect_faces(image_bgr, face_model)

    text_results = detect_text(processed_image, ocr_model)
    privacy_results = detect_privacy(text_results)

    return {
        "image": processed_image,
        "faces": face_results,
        "privacy": privacy_results,
        "texts": text_results,
    }

def make_result(image, faces, privacy, face_unmosaic, privacy_unmosaic):
    """체크된 영역은 모자이크 대상에서 제외하고 결과 이미지를 만듭니다."""
    active_faces = [
        item for index, item in enumerate(faces)
        if index not in face_unmosaic
    ]
    active_privacy = [
        item for index, item in enumerate(privacy)
        if index not in privacy_unmosaic
    ]

    result = apply_privacy_filter(
        image.copy(),
        active_faces,
        active_privacy,
        strength=MOSAIC_STRENGTH,
    )

    if result is None:
        raise RuntimeError("모자이크 결과 이미지를 생성하지 못했습니다.")

    return result

def encode_jpg(image):
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    if not success:
        raise RuntimeError("결과 이미지를 JPG로 변환하지 못했습니다.")
    return encoded.tobytes()

# ---------------------------------------------------------
# 화면
# ---------------------------------------------------------
st.title("🔒 개인정보 보호 모자이크")
st.caption("사진을 촬영하거나 업로드하면 얼굴과 개인정보를 자동으로 탐지하고 모자이크 처리합니다.")

with st.sidebar:
    st.subheader("처리 안내")
    st.write(
        "1. 사진을 업로드하거나 카메라로 촬영합니다.\n\n"
        "2. 사진이 입력되면 별도 실행 버튼 없이 자동으로 분석합니다.\n\n"
        "3. 얼굴과 개인정보에 모자이크가 먼저 적용됩니다.\n\n"
        "4. 모자이크를 해제하고 싶은 영역에 체크한 뒤 결과를 저장합니다."
    )
    st.caption("긴 변이 640px를 초과하는 이미지만 비율을 유지해 축소합니다.")

input_method = st.radio(
    "이미지 입력 방법",
    ["사진 업로드", "카메라 촬영"],
    horizontal=True,
)

if input_method == "사진 업로드":
    uploaded_file = st.file_uploader(
        "모자이크 처리할 사진을 선택하세요.",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
    )
else:
    uploaded_file = st.camera_input("카메라로 사진을 촬영하세요.")

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    current_image_key = hashlib.sha256(image_bytes).hexdigest()

    try:
        source_image = decode_image(image_bytes)
    except Exception as exc:
        st.error(f"이미지를 읽을 수 없습니다: {exc}")
        st.stop()

    original_h, original_w = source_image.shape[:2]
    st.write(f"입력 이미지 크기: **{original_w} × {original_h}**")

    # 새 이미지가 들어온 경우에만 자동 분석합니다.
    # Streamlit 체크박스 조작으로 화면이 다시 실행되어도 AI 분석을 반복하지 않습니다.
    result_data = st.session_state.get("privacy_result_data")
    result_key = st.session_state.get("privacy_result_key")

    if result_key != current_image_key or result_data is None:
        try:
            with st.spinner("사진을 분석하고 모자이크를 적용하고 있습니다..."):
                face_model = load_face_model()
                ocr_model = load_ocr_model()
                result_data = process_image(source_image, face_model, ocr_model)

            st.session_state["privacy_result_data"] = result_data
            st.session_state["privacy_result_key"] = current_image_key
            st.success("자동 탐지 및 모자이크 처리가 완료되었습니다.")

        except Exception as exc:
            st.error(f"처리 중 오류가 발생했습니다: {type(exc).__name__}: {exc}")
            st.exception(exc)
            st.stop()

    faces = result_data["faces"]
    privacy = result_data["privacy"]
    processed_image = result_data["image"]

    st.subheader("탐지 결과")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("얼굴 탐지", f"{len(faces)}개")
    metric2.metric("개인정보 탐지", f"{len(privacy)}개")
    metric3.metric("전체 탐지", f"{len(faces) + len(privacy)}개")

    st.caption(
        f"분석 이미지 크기: {processed_image.shape[1]} × {processed_image.shape[0]}"
    )

    # 체크박스는 기본값이 해제 상태이므로 최초 결과에는 모든 탐지 영역이 모자이크됩니다.
    # 체크하면 해당 영역만 모자이크 대상에서 제외됩니다.
    face_unmosaic = set()
    privacy_unmosaic = set()

    if faces or privacy:
        with st.expander("모자이크 해제할 영역 선택", expanded=True):
            st.caption("체크한 영역은 모자이크가 사라집니다. 체크를 해제하면 다시 모자이크됩니다.")

            if faces:
                st.markdown("**얼굴 영역**")
                face_columns = st.columns(2)

                for index, face in enumerate(faces):
                    x1, y1, x2, y2 = face["bbox"]
                    column = face_columns[index % 2]
                    if column.checkbox(
                        f"얼굴 {index + 1} 모자이크 해제",
                        key=f"face_unmosaic_{current_image_key}_{index}",
                        value=False,
                    ):
                        face_unmosaic.add(index)

            if privacy:
                st.markdown("**개인정보 영역**")
                privacy_columns = st.columns(2)

                for index, item in enumerate(privacy):
                    column = privacy_columns[index % 2]
                    if column.checkbox(
                        f"개인정보 {index + 1} 모자이크 해제",
                        key=f"privacy_unmosaic_{current_image_key}_{index}",
                        value=False,
                    ):
                        privacy_unmosaic.add(index)

    try:
        final_image = make_result(
            processed_image,
            faces,
            privacy,
            face_unmosaic,
            privacy_unmosaic,
        )
        result_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

        st.subheader("모자이크 결과")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**처리 전**")
            st.image(
                cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

        with col2:
            st.markdown("**처리 후**")
            st.image(result_rgb, use_container_width=True)

        st.download_button(
            label="모자이크 사진 다운로드",
            data=encode_jpg(final_image),
            file_name="privacy_mosaic_result.jpg",
            mime="image/jpeg",
            type="primary",
            use_container_width=True,
        )

    except Exception as exc:
        st.error(f"결과 이미지 생성 중 오류가 발생했습니다: {type(exc).__name__}: {exc}")
        st.exception(exc)

else:
    st.info("사진을 업로드하거나 카메라로 촬영하면 자동으로 탐지 및 모자이크 처리가 시작됩니다.")