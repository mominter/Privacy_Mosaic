# 🔒 개인정보 보호 모자이크 (Privacy Mosaic)

사진을 업로드하거나 카메라로 촬영하면 얼굴과 개인정보가 포함된 텍스트 영역을 자동으로 탐지하고 모자이크 처리하는 Streamlit 웹 애플리케이션입니다.

사용자는 탐지된 얼굴 및 개인정보 영역을 확인하고, 모자이크를 유지하거나 원하는 영역의 모자이크를 해제한 뒤 결과 이미지를 다운로드할 수 있습니다.

---

## 주요 기능

- **사진 업로드**: JPG, JPEG, PNG, WEBP, BMP 이미지 입력 지원
- **카메라 촬영**: 웹 브라우저의 카메라를 이용한 사진 촬영
- **얼굴 탐지**: 얼굴 탐지 모듈을 이용해 얼굴 영역 탐지
- **텍스트 인식(OCR)**: PaddleOCR를 이용해 이미지 내 텍스트 인식
- **개인정보 판별**: 인식된 텍스트를 개인정보 탐지 모듈로 전달해 대상 영역 판별
- **자동 모자이크**: 탐지된 얼굴과 개인정보 영역에 기본 모자이크 적용
- **선택적 모자이크 해제**: 체크박스를 선택해 특정 얼굴 또는 개인정보 영역의 모자이크 해제
- **결과 다운로드**: 모자이크가 적용된 이미지를 JPG 파일로 다운로드

## 동작 흐름

1. 사용자가 사진을 업로드하거나 카메라로 촬영합니다.
2. 입력 이미지를 RGB에서 OpenCV 처리용 BGR 형식으로 변환합니다.
3. 얼굴 탐지 모듈이 얼굴 영역을 탐지합니다.
4. PaddleOCR가 이미지의 텍스트를 인식합니다.
5. 개인정보 판별 모듈이 인식 결과에서 개인정보 영역을 판별합니다.
6. 얼굴과 개인정보 영역에 모자이크를 적용합니다.
7. 사용자가 체크한 영역은 모자이크 대상에서 제외합니다.
8. 최종 이미지를 화면에 표시하고 JPG로 다운로드할 수 있도록 제공합니다.

## 프로젝트 구조

아래는 권장 저장소 구조입니다. 실제 모듈 파일명과 배치에 맞게 조정하세요.

```text
Privacy_Mosaic/
├── app.py
├── requirements.txt
├── README.md
└── modules/
    ├── __init__.py
    ├── face_detection.py
    ├── text_detection.py
    ├── privacy_detection.py
    └── privacy_filter.py
```

### 모듈 설명

| 파일 | 역할 |
|---|---|
| `app.py` | Streamlit 화면, 이미지 입력, 모델 초기화, 처리 흐름 및 다운로드 관리 |
| `modules/face_detection.py` | 얼굴 탐지 모델 초기화 및 얼굴 영역 탐지 |
| `modules/text_detection.py` | PaddleOCR 기반 텍스트 인식 |
| `modules/privacy_detection.py` | OCR 결과에서 개인정보 대상 판별 |
| `modules/privacy_filter.py` | 얼굴 및 개인정보 영역 모자이크 처리 |
| `requirements.txt` | 배포에 필요한 Python 패키지 목록 |

## 실행 환경

- Python 3.11 권장
- Streamlit
- OpenCV
- NumPy
- Pillow
- PaddlePaddle / PaddleOCR
- 얼굴 탐지 모듈에서 사용하는 모델 및 라이브러리

> 정확한 패키지와 버전은 프로젝트의 `requirements.txt`를 기준으로 설치해야 합니다. 얼굴 탐지 모듈의 구현에 따라 추가 패키지가 필요할 수 있습니다.

## 로컬 실행 방법

### 1. 저장소 복제

```bash
git clone https://github.com/mominter/Privacy_Mosaic.git
cd Privacy_Mosaic
```

### 2. 가상환경 생성 및 활성화

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 패키지 설치

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Streamlit 실행

```bash
streamlit run app.py
```

브라우저에서 Streamlit이 출력하는 로컬 주소로 접속합니다.

## Streamlit Cloud 배포

1. 프로젝트를 GitHub 저장소에 Push합니다.
2. [Streamlit Community Cloud](https://share.streamlit.io/)에 접속하고 GitHub 계정으로 로그인합니다.
3. **Create app**을 선택합니다.
4. 저장소를 `mominter/Privacy_Mosaic`로 지정합니다.
5. 브랜치를 `main`으로 지정합니다.
6. Main file path를 `app.py`로 지정합니다.
7. Advanced settings에서 프로젝트와 호환되는 Python 버전을 선택합니다. 현재 개발 기준은 Python 3.11입니다.
8. **Deploy**를 눌러 배포합니다.

GitHub의 `main` 브랜치에 변경 사항을 Push하면 Streamlit Cloud가 앱을 다시 배포합니다.

## OCR 실행 설정

현재 `app.py`의 PaddleOCR 초기화에는 다음 설정이 적용되어 있습니다.

```python
PaddleOCR(
    lang="korean",
    use_textline_orientation=True,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    enable_mkldnn=False,
    cpu_threads=2,
    device=device,
)
```

- `lang="korean"`: 한국어 OCR 모델 사용
- `use_textline_orientation=True`: 텍스트 줄 방향 인식 사용
- `use_doc_orientation_classify=False`: 문서 전체 방향 분류 비활성화
- `use_doc_unwarping=False`: 문서 휘어짐 보정 비활성화
- `enable_mkldnn=False`: oneDNN(MKLDNN) 사용 비활성화
- `cpu_threads=2`: CPU 추론 스레드 수 설정
- `device`: CUDA 사용 가능 여부에 따라 GPU 또는 CPU 선택

이 설정은 현재 앱 코드에 기재된 값입니다. PaddleOCR/PaddlePaddle 버전에 따라 지원 옵션과 실행 동작이 달라질 수 있으므로 배포 환경의 패키지 호환성을 확인해야 합니다.

## 이미지 처리 및 상태 관리

- 업로드된 이미지 바이트를 SHA-256으로 해시해 현재 이미지의 식별 키로 사용합니다.
- 이미지가 바뀌었거나 저장된 분석 결과가 없을 때 얼굴 및 OCR 분석을 수행합니다.
- `st.cache_resource`로 얼굴 탐지 모델과 OCR 모델을 캐싱합니다.
- `st.session_state`에 분석 결과와 이미지 키를 저장해 체크박스를 조작할 때 불필요한 AI 분석이 반복되지 않도록 합니다.
- 모자이크 강도는 `app.py`의 `MOSAIC_STRENGTH` 값으로 설정합니다. 현재 값은 `12`입니다.

## 사용 방법

1. 웹 앱에서 **사진 업로드** 또는 **카메라 촬영**을 선택합니다.
2. 이미지를 업로드하거나 카메라로 촬영합니다.
3. 분석이 자동으로 진행될 때까지 기다립니다.
4. 얼굴 및 개인정보 탐지 개수를 확인합니다.
5. 기본 상태에서는 탐지된 모든 영역에 모자이크가 적용됩니다.
6. 모자이크를 해제할 영역의 체크박스를 선택합니다.
7. 처리 전·후 이미지를 비교합니다.
8. **모자이크 사진 다운로드**를 눌러 JPG 결과물을 저장합니다.

## 배포 및 운영 시 유의사항

- Streamlit Community Cloud의 실행 환경은 로컬 PC와 다를 수 있습니다. 로컬에서 GPU를 사용하더라도 배포 환경에서 GPU가 제공된다고 가정하면 안 됩니다.
- 배포 환경에서는 CPU 실행을 기준으로 패키지와 모델을 검증하는 것이 안전합니다.
- PaddlePaddle, PaddleOCR, PaddleX 및 얼굴 탐지 라이브러리의 버전 호환성을 확인해야 합니다.
- 모델 파일이 로컬 경로에만 존재한다면 배포 환경에서도 접근할 수 있도록 모델 다운로드 또는 별도 배포 방법을 마련해야 합니다.
- 대용량 모델 파일이나 가상환경 폴더를 Git에 무분별하게 포함하지 않도록 `.gitignore`를 설정합니다.
- 사진은 서버에서 처리됩니다. 실제 개인정보가 포함된 사진을 다룰 때는 접근 권한, 보관 여부, 로그 및 임시 파일 관리 정책을 확인하고 테스트에는 가급적 샘플 이미지를 사용합니다.
- 자동 탐지는 누락 또는 오탐이 발생할 수 있으므로 다운로드 전 결과를 확인해야 합니다.

## 문제 해결

### `ModuleNotFoundError`

필요한 라이브러리가 `requirements.txt`에 있는지 확인하고 배포 로그에서 설치 성공 여부를 확인합니다.

### PaddleOCR / PaddlePaddle 추론 오류

PaddleOCR, PaddlePaddle, PaddleX 버전 조합과 실행 장치(CPU/GPU)를 확인합니다. 오류가 발생하면 전체 Traceback과 설치 버전을 함께 확인해야 합니다.

### 모델 파일을 찾을 수 없는 오류

모델 초기화 코드에서 사용하는 경로와 배포 환경의 실제 파일 위치를 확인합니다. 모델이 자동 다운로드되는 구조인지도 확인합니다.

### 배포 후 변경 사항이 반영되지 않는 경우

GitHub의 올바른 저장소와 브랜치(`main`)에 Push했는지 확인하고 Streamlit Cloud의 앱 로그 및 재실행 상태를 확인합니다.

---

## 기술 스택

- **Frontend / Web App**: Streamlit
- **Image Processing**: OpenCV, NumPy, Pillow
- **Face Detection**: 프로젝트의 `modules/face_detection.py` 구현에 따름
- **OCR**: PaddleOCR
- **Privacy Classification**: 프로젝트의 `modules/privacy_detection.py` 구현에 따름
- **Mosaic Processing**: 프로젝트의 `modules/privacy_filter.py` 구현에 따름
- **Deployment**: GitHub, Streamlit Community Cloud
