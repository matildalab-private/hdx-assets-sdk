# hdx-assets-sdk

AssetHub의 데이터셋, 모델, 알고리즘, 파이프라인 에셋을 Python에서 제어하는 SDK입니다.

## 설치

### 개발 중 (소스 수정 즉시 반영)

```bash
pip install -e .
```

### 배포용 설치

```bash
pip install .
```

### 패키지 빌드

```bash
pip install build
python -m build
```

`dist/` 폴더에 배포 파일이 생성됩니다.

```
dist/
  asset_hub-1.0.3-py3-none-any.whl
  asset_hub-1.0.3.tar.gz
```

### 다른 프로젝트에서 설치

```bash
# 빌드된 whl 파일로 설치
pip install dist/asset_hub-1.0.3-py3-none-any.whl

# git 레포에서 직접 설치
pip install git+https://github.com/yourorg/hdx-assets-sdk.git

# 로컬 경로에서 설치
pip install /path/to/hdx-assets-sdk/
```

## 환경 설정

홈 디렉토리에 `~/.asset` 파일을 생성하거나, 프로젝트 루트에 별도 파일을 두고 경로를 지정합니다.

```
HOST=https://your-assethub-host.com
API_KEY=your-api-key
API_USER=your-username
```

```python
api = AssetHubAPI()              # 기본값: ~/.asset
api = AssetHubAPI('.asset_mlops')  # 파일 경로 직접 지정
```

---

## 기본 사용법

### API 초기화 및 에셋 접근

```python
from asset_hub.asset_hub_api import AssetHubAPI, AssetsType

api = AssetHubAPI()

# ID로 접근
dataset = api.assets(assets_id=45)

# 별칭(alias)으로 접근
dataset = api.assets(alias='mnist-dataset')

# 특정 revision 고정
dataset = api.assets(assets_id=45, revision=2)
```

---

### 파일 목록 조회

```python
for f in dataset.ls('/'):
    print(f)
# 출력: 타입(D/F)  이름  수정일  크기

# 하위 경로 조회
for f in dataset.ls('/MNIST_data/MNIST/raw'):
    print(f)
```

---

### 다운로드

```python
# 단일 파일
dataset.download('MNIST_data/MNIST/raw/t10k-images-idx3-ubyte',
                 './t10k-images-idx3-ubyte')

# 폴더
dataset.download('MNIST_data/MNIST/raw/', './raw/')

# 전체 (uname-revision 서브폴더 자동 생성)
dataset.download_all('./')

# 전체 (서브폴더 없이 dst 바로 아래)
dataset.download_all('./', enable_meta=False)
```

---

### 파일을 메모리로 직접 로드 (`io.BytesIO`)

```python
stream = dataset.load('/labels.csv')
import pandas as pd
df = pd.read_csv(stream)
```

---

### 업로드 및 배포

```python
# 기존 에셋에 파일 업로드
model = api.assets(assets_id=100)
model.upload('./model.pt', '/model.pt')

# 폴더 업로드
model.upload('./checkpoints/', '/checkpoints/')

# 배포
model.publish(summary='v1.0 initial release')
```

---

### 새 에셋 생성

```python
dataset = api.assets(assets_id=45)

model = api.new_assets(
    AssetsType.MODEL,         # DATASET / MODEL / ALGORITHM / PIPELINE
    "MNIST_Model",            # 에셋 이름
    "MNIST classification",   # 설명
    used_assets=[dataset],    # 참조 에셋 연결 (선택)
    alias='mnist-model-v1'    # 별칭 (선택)
)

model.upload('./model.pt', '/model.pt')
model.publish()
```

---

## 전체 ML 파이프라인 예시

알고리즘 코드 다운로드 → 환경 설치 → 데이터셋 다운로드 → 학습 → 모델 업로드/배포

```python
import sys
import subprocess
from asset_hub.asset_hub_api import AssetHubAPI, AssetsType

api = AssetHubAPI('.asset_mlops')

# 1. 알고리즘 코드 다운로드
algorithm = api.assets(assets_id=46)
algorithm.download_all('./', enable_meta=False)

# 2. 의존성 설치
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "./requirements.txt"])

# 3. 데이터셋 다운로드
dataset = api.assets(assets_id=45)
dataset.download('MNIST_data', './MNIST_data')

# 4. 학습
from train import train
train(save_to='./model.pt')

# 5. 모델 에셋 생성
model = api.new_assets(
    AssetsType.MODEL,
    "MNIST_Model",
    "MNIST MODEL",
    used_assets=[dataset]
)

# 6. 업로드 & 배포
if model.upload('./model.pt', '/model.pt'):
    model.publish()
```

---

## PyTorch ImageFolder 연동

로컬에 다운로드하지 않고 AssetHub에서 직접 이미지를 로드합니다.

```python
import torch
import torchvision.transforms as transforms
from asset_hub.asset_hub_api import AssetHubAPI
from asset_hub.cache import AssetMemoryCache
from asset_hub.torchbinding.folder import ImageFolder as AssetHubImageFolder

api = AssetHubAPI('.asset_mlops')
api.set_cache_policy(AssetMemoryCache())  # 메모리 캐시로 반복 접근 최적화

vision_dataset = api.assets(assets_id=134)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# torchvision.datasets.ImageFolder과 동일한 인터페이스
train_data = AssetHubImageFolder(vision_dataset, '/OF/train', transform)
val_data   = AssetHubImageFolder(vision_dataset, '/OF/val', transform)

loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
```

---

## 캐시 정책

| 정책 | 클래스 | 설명 |
|------|--------|------|
| 캐시 없음 (기본값) | `AssetNoCache` | 매번 네트워크 요청 |
| 메모리 캐시 | `AssetMemoryCache` | 프로세스 내 메모리에 캐싱 |

```python
from asset_hub.cache import AssetMemoryCache, AssetNoCache

api.set_cache_policy(AssetMemoryCache())
```

---

## API 레퍼런스

### `AssetHubAPI`

| 메서드 | 설명 |
|--------|------|
| `assets(assets_id, alias, revision)` | 에셋 인터페이스 반환 |
| `new_assets(type, name, comment, used_assets, alias)` | 새 에셋 생성 |
| `set_cache_policy(cache)` | 캐시 정책 설정 |

### `Assets`

| 메서드 | 설명 |
|--------|------|
| `ls(path, limit)` | 파일 목록 조회 |
| `download(src, dst)` | 파일 또는 폴더 다운로드 |
| `download_all(dst, enable_meta)` | 전체 다운로드 |
| `load(src)` | 파일을 `io.BytesIO`로 메모리 로드 |
| `get_file_cached(src)` | 캐시를 이용한 파일 로드 |
| `upload(src, dst)` | 파일 또는 폴더 업로드 |
| `publish(summary)` | 에셋 배포 |

### `AssetsType`

| 상수 | 값 |
|------|----|
| `AssetsType.DATASET` | `'dataset'` |
| `AssetsType.MODEL` | `'model'` |
| `AssetsType.ALGORITHM` | `'algorithm'` |
| `AssetsType.PIPELINE` | `'pipeline'` |
