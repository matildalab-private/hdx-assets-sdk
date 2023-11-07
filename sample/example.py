import os

from asset_hub.asset_hub_api import AssetHubAPI, AssetsType

import sys
import subprocess

# MNIST_DATASET_ID = 44
# MNIST_ALGORITM_ID = 45

MNIST_DATASET_ID = 45
MNIST_ALGORITM_ID = 46


def main():
    # 사용자 설정 파일 지정
    api = AssetHubAPI('.asset_mlops')

    algorithm = api.assets(MNIST_ALGORITM_ID)
    if algorithm is None:
        return

    # enable_meta -> uname-revision 하위 디렉토리에 다운로드
    algorithm.download_all('./', enable_meta=False)

    # pip install
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "./requirements.txt"])

    dataset = api.assets(MNIST_DATASET_ID)
    if dataset is None:
        return

    # 배포용 모델 생성
    model = api.new_assets(
        AssetsType.MODEL,
        "MNIST_Model",
        "MNIST MODEL TEST Notebook")
    if model is None:
        return

    print("ls folders & Files")
    for x in dataset.ls('/'):
        print(x)

    # 파일
    # dataset.download('MNIST_data/MNIST/raw/t10k-images-idx3-ubyte',
    #                  'MNIST_data/MNIST/raw/t10k-images-idx3-ubyte')
    #
    # 폴더
    # dataset.download('MNIST_data/MNIST/raw/',
    #                  'MNIST_data/MNIST/raw/')
    #
    # 전체
    # dataset.download_all('./)

    # 데이터셋 다운로드
    if not os.path.exists('MNIST_data'):
        dataset.download('MNIST_data', './MNIST_data')

    print("Begin Train")
    from train import train
    train(save_to='./model.pt')
    print("End Train")

    if model.upload('./model.pt', '/model.pt'):
        model.publish()


if __name__ == '__main__':
    main()
