""" AssetHubAPI Module

AssetHub assets 에 대한 제어를 제공한다.

* 사용 예제

.. code-block:: python

    ALGORITM_ID = ...(user input)
    DATASET_ID = ...(user input)
    DATASET_ALIAS = ...(user input)
    MODEL_ID = ...(user input)
    local_loc = ...(user input)

    try:
        api = AssetHubAPI() # or AssetHubAPI('.asset') # 설정 파일 지정 

        algorithm = api.assets(ALGORITM_ID)
        if algorithm is None:
            return

        # enable_meta -> {uname}-{revision} 하위 디렉토리에 다운로드
        algorithm.download_all('./', enable_meta=False)

        # 알고리즘 실행 환경 재현
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "./requirements.txt"])

        # 에셋 로드
        assets = api.assets(assets_id=DATASET_ID) # or api.assets(alias=DATASET_ALIAS) # 별칭으로 접근
        if assets is not None:
            # 파일 목록 조회
            for x in assets.ls('/'):
                print(x)

            # 에셋 파일 전체 다운로드
            assets.download_all(local_loc)

            # MNIST_data 데이터셋 기준 다음과 같이 다운로드
            # remote 파일 -> local 파일
            # dataset.download('MNIST_data/MNIST/raw/t10k-images-idx3-ubyte',
            #                  './t10k-images-idx3-ubyte')
            #
            # remote 폴더 -> local 폴더
            # dataset.download('MNIST_data/MNIST/raw/',
            #                  './raw/')
            #
            # 전체를 현재 디렉토리에
            # dataset.download_all('./)

            # 사용자 정의 로직 실행
            train(local_loc, ...)

            # 업르드및 배포용 에셋 로드
            model = api.assets(assets_id=MODEL_ID)
            '''
            or
            # 업르드및 배포용 에셋 생성
            model = api.new_assets(AssetsType.MODEL,
                                "model name",
                                "mode description",
                                used_assets=[assets])
            '''
            local_src = ... # ex) './example.py'
            remote_dst = ... # ex) '/example.py'

            # 업로드
            if model.upload(local_src, remote_dst):
                # 배포
                model.publish()

    except Exception as e:
        print(e)

"""
from __future__ import annotations

import io
import os
import time
from math import floor, ceil
from threading import Event, Thread
from typing import Optional

import requests
import tqdm
from colorama import Fore, Style
from urllib3.exceptions import InsecureRequestWarning




class AssetsType:
    """에셋 타입 클래스
    """
    DATASET = 'dataset'
    MODEL = 'model'
    ALGORITHM = 'algorithm'
    PIPELINE = 'pipeline'


class FileItem:
    """파일 정보

    :param str file_type: 파일 종류 'D' Directory 'F' File

    :param name: 파일 이름

    :param update_date: 파일 수정일

    :param size: 파일 크기
    """

    def __init__(self, file_type: str, name: str, update_date: str, size: int):
        """Constructor
        """
        self.file_type = file_type
        self.name = name
        self.update_date = update_date
        self.size = size

    def __repr__(self):
        return f"{self.file_type}\t{self.name : <40}{self.update_date}\t{self.size}"


class APILogger:
    """Logging

    console 로깅 출력
    """

    def error(self, message: str, raise_exception: bool = True):
        """에러 메시지 출력

        :param message: message

        :param raise_exception: 예외 발생 여부
        """
        print(f"{Fore.RED}{message}{Style.RESET_ALL}")
        if raise_exception:
            raise Exception(message)

    def info(self, message: str):
        """정보 메세지 출력.

        :param message: message
        """
        print(f"{Fore.BLUE}{message}{Style.RESET_ALL}")


class APIResponse:
    """API 응답

    :param requests.Response resp: http 응답


    """

    def __init__(self, resp: requests.Response):
        """Constructor
        """
        resp_json = resp.json()
        self.resp = resp
        self.status = resp_json['status']
        self.message = resp_json['message']
        self.errors = resp_json['errors']
        self.data = resp_json['data']

    def success(self) -> bool:
        """성공여부

        :return bool:
        """
        return self.status == 'success'

    def __repr__(self) -> str:
        return str(self.resp.json())


class Assets:
    """Assets 인터페이스

    :param api: api instance

    :param assets: assets dict

    :param revision: revision dict
    """

    def __init__(self, api: AssetHubAPI, assets: dict, revision: dict):
        self.api = api
        self.assets = assets
        self.revision = revision
        self.last_commit_id = self.assets['last_commit_id']

    def ls(self, path: str, limit: int = 1000) -> FileItem:
        """파일 목록 조회

        :param path: 대상 path '/...'

        :param limit: 페이징 처리 한번에 가져올 목록 개수

        :return: :py:class:`FileItem`
        """
        return self.api.ls(self, path, limit)

    def load(self, src: str) -> Optional[bytes]:
        """파일 데이터 로드

        :param src: 대상 파일
        :return: bytes 데이터
        """
        try:
            paths = [x for x in src.split('/') if len(x) > 0]
            if len(paths) == 0:
                return None
            else:
                lpath = paths[-1]
                paths = paths[:-1]
                path = os.path.join(*paths) if len(paths) > 0 else ''
                match = False
                for f in self.ls(path):
                    if f.name == lpath:
                        match = True
                        if f.file_type == 'F':
                            return self._load_file(src)
                        else:
                            return None

                if match is False:
                    self.api.logger.error(f"Load Failed {src} not exist", False)
                    return None

        except Exception as e:
            self.api.logger.error(f"Load Failed {e}", False)

        return None

    def _load_file(self, src: str) -> Optional[bytes]:
        """파일 데이터 로드 통신 구현

        :param src: 대상 파일
        :return: bytes 데이터
        """
        url = AssetHubAPI.URLS["blob"].format(
            self.assets['id'],
            self.revision['commit_id'],
            src)
        with requests.get(
                f'{self.api.host}/{url}',
                headers=self.api.api_headers(),
                stream=True,
                verify=True) as resp:
            if resp.headers.get('content-type') == 'application/json':
                api_resp = APIResponse(resp)
                self.api.logger.error(f"{src} Load Error {api_resp.message} {api_resp.errors}")
            else:
                total_length = int(resp.headers.get('Content-Length'))
                stream = io.BytesIO()
                with tqdm.tqdm(total=total_length) as pbar:
                    for chunk in resp.iter_content(chunk_size=8192):
                        stream.write(chunk)
                        pbar.update(len(chunk))
                        time.sleep(0.0003)
                return stream.getvalue()
        return None

    def download(self, src: str, dst: str) -> bool:
        """ 지정 파일 다운로드

        :param src: 원본 경로 파일, 폴더

        :param dst: 대상 경로 파일, 폴더

        """
        try:
            paths = [x for x in src.split('/') if len(x) > 0]
            if len(paths) == 0:
                self._get_folder('/', dst)
            else:
                lpath = paths[-1]
                paths = paths[:-1]
                path = os.path.join(*paths) if len(paths) > 0 else ''
                match = False
                for f in self.ls(path):
                    if f.name == lpath:
                        match = True
                        if f.file_type == 'F':
                            self._get_file(path, f, dst)
                        else:
                            self._get_folder(src, dst)

                if match is False:
                    self.api.logger.error(f"Download Failed {src} not exist", False)
                    return False

        except Exception as e:
            self.api.logger.error(f"Download Failed {e}", False)
            return False
        return True

    def download_all(self, dst: str, enable_meta: bool = True) -> bool:
        """전체 파일 다운로드

        :param dst: 대상 경로
        :param enable_meta: 활성화시 dst/{uname}-{revision} 하위 디렉토리에 다운로드
        :return:
        """
        if enable_meta:
            uname = self.assets['uname']
            if uname is None or len(uname) <= 0:
                uname = str(self.assets['id'])
            dst = os.path.join(dst, f"{uname}-{self.revision['revision']}")
        return self.download('', dst)

    def upload(self, src: str, dst: str) -> bool:
        """파일 업로드

        :param src: 원본 파일/폴더
        :param dst: 대상 파일/폴더
        :return:
        """
        if not os.path.exists(src):
            self.api.logger.error(f"{src} not exist")
            return False

        src_isdir = os.path.isdir(src)
        dst_isdir = None

        basename, dirname = os.path.basename(dst), os.path.dirname(dst)
        if len(dirname) == 0:
            dirname = '/'
        self.api.logger.info("Check Target")
        for x in self.ls(dirname):
            if x.name == basename:
                if x.file_type == 'D':
                    dst_isdir = True
                else:
                    dst_isdir = False

        if src_isdir and dst_isdir is False:
            self.api.logger.error("cant' push dir to file")
            return False

        if not src_isdir and dst_isdir is True:
            self.api.logger.error("cant' push file to dir - set full dst path")
            return False

        lock_token = None
        try:
            lock_token_data = self.api.lock_acquire(self.assets['id'], self.last_commit_id)
            if lock_token_data is None:
                return False
            lock_token = lock_token_data['lock_token']

            event = Event()
            thread = Thread(target=self.lock_refresh_proc, args=(lock_token, event,))
            thread.start()

            if src_isdir:
                self.send_directory(src, dst, lock_token)
            else:
                self.send_file(src, dst, lock_token)
            event.set()
            thread.join()
            self.api.lock_release(self.assets['id'], lock_token)
            return True
        except Exception as e:
            if lock_token is not None:
                event.set()
                thread.join()
                self.api.lock_release(self.assets['id'], lock_token)
            self.api.logger.error(f"{e}", False)
            return False

    def publish(self, summary: str = 'sdk publish'):
        """에셋 배포

        :param summary: 사용자 코멘트

        :return:
        """
        self.api.logger.info("Publish")
        api_resp = self.api.publish(self.assets['id'], self.last_commit_id, summary)

        if not api_resp.success():
            self.api.logger.error(f"Publish Fail {api_resp.errors}")
            return False

        state_token = api_resp.data['state_token']
        while True:
            time.sleep(5)
            api_resp = self.api.publish_state(
                self.assets['id'], state_token)
            if api_resp.success():
                if api_resp.errors is None:
                    self.api.logger.info("Publish success")
                    return
                else:
                    self.api.logger.info("Wait for publish")
            else:
                self.api.logger.error(f"Publish Failed {api_resp.errors}")
                return False

    def lock_refresh_proc(self, lock_token: str, event: Event):
        """잠금 유지

        :param lock_token: lock_token
        :param event: 종료 여부 체크용 이벤트
        :return:
        """
        count = 0
        while not event.is_set():
            time.sleep(1)
            count += 1
            if count > 60:
                count = 0
                self.api.lock_refresh(self.assets['id'], lock_token)

    def send_file(self, src: str, dst: str, lock_token: str):
        """파일 전송

        :param src: 원본 파일
        :param dst: 대상 파일
        :param lock_token: lock_token
        :return:
        """
        self.api.logger.info(f"Send File : {src} to {dst}")
        basename, dirname = os.path.basename(dst), os.path.dirname(dst)
        if len(dirname) == 0:
            dirname = '/'
        size = os.path.getsize(src)
        if size < AssetHubAPI.CHUNK_SIZE:
            multiple_files = [(dirname, (basename, open(src, 'rb'),)), ]
            api_resp = self.api.add_files(self.assets['id'],
                                          self.last_commit_id,
                                          lock_token,
                                          multiple_files)

            if api_resp.success():
                self.last_commit_id = api_resp.data['commit_id']
            else:
                self.api.logger.error(f"SendFile {src} to {dst} Failed {api_resp.errors}")
        else:
            with open(src, 'rb') as fobj:
                with tqdm.tqdm(total=size) as pbar:
                    total_chunks = ceil(size / AssetHubAPI.CHUNK_SIZE)
                    for i in range(0, total_chunks):
                        data = fobj.read(AssetHubAPI.CHUNK_SIZE)
                        files = [(dirname, (basename, data,)), ]
                        api_resp = self.api.add_chunk(self.assets['id'],
                                                      self.last_commit_id,
                                                      lock_token,
                                                      i,
                                                      total_chunks,
                                                      files)
                        pbar.update(AssetHubAPI.CHUNK_SIZE)
                        time.sleep(0.0003)
                        if api_resp.success():
                            if api_resp.errors is None:
                                self.last_commit_id = api_resp.data['commit_id']
                            else:
                                continue
                        else:
                            self.api.logger.error(f"SendFile {src} to {dst} Failed {data}")

    def send_directory(self, src: str, dst: str, lock_token: str):
        """폴더 전송

        :param src: 원본 폴더
        :param dst: 대상 폴더
        :param lock_token: lock_token
        :return:
        """
        self.api.logger.info(f"Send Driectory : {src} to {dst}")
        chunkfiles = []
        pending_files = []
        pending_files_sizes = 0
        for (root, dirs, files) in os.walk(src):
            if '/.' in root:
                continue
            for file in files:
                if file.startswith('.') or file.endswith('.dvc'):
                    continue
                full_path = os.path.join(root, file)
                size = os.path.getsize(full_path)
                if size > AssetHubAPI.CHUNK_SIZE:
                    chunkfiles.append(full_path)
                else:
                    if pending_files_sizes + size > AssetHubAPI.CHUNK_SIZE or len(pending_files) > 100:
                        self.send_files(pending_files, src, dst, lock_token)
                        pending_files_sizes = 0
                        pending_files = []
                    pending_files.append(full_path)
                    pending_files_sizes += size

        if len(pending_files) > 0:
            self.send_files(pending_files, src, dst, lock_token)

        for x in chunkfiles:
            self.send_file(
                x, x.replace(src, dst), lock_token
            )

    def send_files(self, files: list[str], src: str, dst: str, lock_token: str):
        """파일 목록 전송

        :param files: 전송할 파일 이름 목록
        :param src: 원본 경로
        :param dst: 대상 경로
        :param lock_token: lock_token
        :return:
        """
        multiple_files = []
        for x in files:
            dstname = x.replace(src, dst)
            self.api.logger.info(f"Send Files : {x} to {dstname}")
            basename, dirname = os.path.basename(dstname), os.path.dirname(dstname)
            if len(dirname) == 0:
                dirname = '/'
            multiple_files.append(
                (dirname, (basename, open(x, 'rb'),)),
            )

        api_resp = self.api.add_files(
            self.assets['id'], self.last_commit_id, lock_token, multiple_files
        )

        if api_resp.success():
            self.last_commit_id = api_resp.data['commit_id']
        else:
            self.api.logger.error(f"SendFile {src} to {dst} Failed {api_resp.errors}")

    def _get_folder(self, src: str, dst: str):
        """폴더 다운로드

        :param src: 원본 폴더
        :param dst: 대상 폴더
        """
        self.api.logger.info(f"Download Folder {src} -> {dst}")
        folders = []
        for x in self.ls(src):
            if not os.path.exists(dst):
                os.makedirs(dst, exist_ok=True)
            if x.file_type == "D":
                os.makedirs(os.path.join(dst, x.name), exist_ok=True)
                folders.append((os.path.join(src, x.name), os.path.join(dst, x.name)))
            else:
                self._get_file(src, x, os.path.join(dst, x.name))
        for s, d in folders:
            self._get_folder(s, d)

    def _get_file(self, src_path: str, src_file_info: FileItem, dst: str):
        """파일 다운로드

        :param src_path: 원본 경로
        :param src_file_info: 원본 파일 정보
        :param dst: 대상
        :return:
        """
        src = os.path.join(src_path, src_file_info.name)
        if os.path.exists(dst) and os.path.getsize(dst) == src_file_info.size:
            self.api.logger.info(f"Download file {src} -> {dst} already exist")
            return

        self.api.logger.info(f"Download file {src} -> {dst}")
        url = AssetHubAPI.URLS["blob"].format(
            self.assets['id'],
            self.revision['commit_id'],
            src)
        with requests.get(
                f'{self.api.host}/{url}',
                headers=self.api.api_headers(),
                stream=True,
                verify=True) as resp:
            if resp.headers.get('content-type') == 'application/json':
                api_resp = APIResponse(resp)
                self.api.logger.error(f"{src} Download Error {api_resp.message} {api_resp.errors}")
            else:
                dirname = os.path.dirname(dst)
                if len(dirname) > 0:
                    os.makedirs(dirname, exist_ok=True)

                total_length = int(resp.headers.get('Content-Length'))
                with open(dst, 'wb') as f:
                    with tqdm.tqdm(total=total_length) as pbar:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                            time.sleep(0.0003)

    def __repr__(self):
        return f"{self.assets} {self.revision}"


class AssetHubAPI:
    """API 제어 클래스

    :param str envfile: 환경파일 if envfile is None: "~/.asset" else envfile

    :param APILogger logger: API 메시지 출력 로거


    """
    VERSION = "1.0.1"

    URLS = {
        # "login": "api/asset_hub/v1/auth/login",
        "assets_by_id": "asset_hub/v1/assets/{0}",
        "assets_new": "asset_hub/v1/assets",
        "assets_by_alias": "asset_hub/v1/assets/alias/{0}",
        "assets_revisions": "asset_hub/v1/assets/{0}/revisions?limit={1}&&page={2}",
        "assets_revision": "asset_hub/v1/assets/{0}/revision/{1}",
        "assets_ls": "asset_hub/v1/assets/{0}/{1}/ls/{2}?limit={3}&page={4}",
        "blob": "asset_hub/v1/assets/{0}/{1}/blob/{2}",
        "lock_acquire": "asset_hub/v1/assets/{0}/{1}/lock/acquire",
        "lock_refresh": "asset_hub/v1/assets/{0}/lock/refresh",
        "lock_release": "asset_hub/v1/assets/{0}/lock/release",
        "assets_add_files": "asset_hub/v1/assets/{0}/{1}/add",
        "assets_add_chunk": "asset_hub/v1/assets/{0}/{1}/add_chunk",
        "publish": "asset_hub/v1/assets/{0}/{1}/publish",
        "publish_state": "asset_hub/v1/assets/{0}/publish_state",
        "set_used_assets": "asset_hub/v1/assets/{0}/ext_info/USED_ASSETS",
    }

    CHUNK_SIZE = 10 * 1024 * 1024

    def __init__(self, envfile=None, logger=APILogger()):
        """Constructor
        """
        requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
        if envfile is None:
            envfile = os.path.join(os.path.expanduser('~'), '.asset')
        if not os.path.exists(envfile):
            logger.error(f"{envfile} is not exist")
        self.host = None
        self.api_key = None
        self.api_user = None
        with open(envfile) as f:
            for line in f.readlines():
                if line.startswith('#') or not line.strip():
                    continue
                key, value = line.strip().split('=', 1)
                if key == "HOST":
                    self.host = value.rstrip('/').strip()
                if key == "API_KEY":
                    self.api_key = value.strip()
                if key == "API_USER":
                    self.api_user = value.strip()
            if (
                self.host is None or
                self.api_key is None or
                self.api_user is None
            ):
                logger.error(f"Invalid EnvFiles")
        self.logger = logger

    def api_headers(self) -> dict:
        """api access header
        """
        return {
            "api-key": self.api_key,
            "api-user": self.api_user
        }

    def get(self, url: str) -> APIResponse:
        """HTTP Get Request

        :param url: url string

        """
        resp = requests.get(
            f'{self.host}/{url}',
            headers=self.api_headers(),
            verify=True)
        return APIResponse(resp)

    def post(self, url,
             data: dict = None,
             files: list[tuple[str, tuple[str, int]]] = None,
             is_json: bool = True) -> APIResponse:
        """HTTP Post Request

        :param url: urlstring

        :param data: post data

        :param files: files data

        :param is_json: json 전송 여부 - file 은 multipart/form 으로 처리
        """
        if is_json:
            resp = requests.post(
                f'{self.host}/{url}',
                headers=self.api_headers(),
                json=data, files=files,
                verify=True)
        else:
            resp = requests.post(
                f'{self.host}/{url}',
                headers=self.api_headers(),
                data=data, files=files,
                verify=True)

        return APIResponse(resp)

    def assets(self,
               assets_id: Optional[int] = None,
               alias: Optional[str] = None,
               revision: Optional[int] = None) -> Optional[Assets]:
        """Assets 인터페이스 생성

        :param assets_id: assets_id

        :param alias: unique name

        :param revision: revision

        """
        if assets_id is None and alias is None:
            self.logger.error(f"assets_id or alias must be set")
            return None
        assets_data = None
        if assets_id is not None:
            assets_data = self.get_assets_by_id(assets_id)
        elif alias is not None:
            assets_data = self.get_assets_by_alias(alias)

        if assets_data is None:
            return None

        revision_data = None
        if revision is None:
            revision_data = assets_data['latest_rev']
        else:
            try:
                revision = int(revision)
            except ValueError:
                self.logger.error(f"revision:{revision} 값은 숫자여야 합니다.")
                return None
            revision_data = self.get_revision(assets_data['id'], int(revision))

        if revision_data is None:
            return None
        return Assets(self, assets_data, revision_data)

    def new_assets(self,
                   assets_type: str,
                   name: str,
                   comment: str,
                   used_assets: Optional[list[Assets]] = None
                   ) -> Optional[Assets]:
        """Assets 생성

        :param assets_type: 에셋 타입 :py:class:`AssetsType`

        :param name: 에셋 이릅

        :param comment: 에셋 설명

        :param used_assets: 참조 에셋

        """
        api_resp = self.post(
            AssetHubAPI.URLS['assets_new'],
            data={
                "type": assets_type,
                "name": name,
                "comment": comment,
                "attributes": [],
                "projects": [],
                "permission_read": 0,
                "permission_write": 0,
                "permission_clone": 0
            }
        )
        if not api_resp.success():
            self.logger.error(f"NewModel Failed {api_resp.errors}")
            return None

        if used_assets is not None:
            self.post(
                AssetHubAPI.URLS["set_used_assets"].format(api_resp.data['id']),
                data={
                    "value": [
                        {"id": x.assets['id'],
                         "name": x.assets['name']}
                        for x in used_assets
                    ]})

        return self.assets(assets_id=api_resp.data['id'])

    def get_assets_by_id(self, assets_id: int) -> Optional[dict]:
        """id 로 assets 을 검색

        :param assets_id: assets_id
        """
        api_resp = self.get(
            AssetHubAPI.URLS["assets_by_id"].format(assets_id))
        if api_resp.success():
            return api_resp.data
        self.logger.error(f"{api_resp.message, api_resp.errors}")
        return None

    def get_assets_by_alias(self, alias: str) -> Optional[dict]:
        """별칭으로 assets 을 검색

        :param alias: 별칭
        """
        api_resp = self.get(
            AssetHubAPI.URLS["assets_by_alias"].format(alias))
        if api_resp.success():
            return api_resp.data
        self.logger.error(f"{api_resp.message, api_resp.errors}")
        return None

    def get_revision(self, assets_id: int, revision: int) -> Optional[dict]:
        """에셋의 revision을 검색

        :param assets_id: assets_id

        :param revision: revision_id

        :return dict: "commit_id" : str , "tag" : str, "revision" : int, "differences" : str, "summary" : str
        """
        api_resp = self.get(
            AssetHubAPI.URLS["assets_revision"].format(assets_id, revision))
        if api_resp.success():
            return api_resp.data
        self.logger.error(f"{api_resp.message, api_resp.errors}")
        return None

    def ls(self, assets: Assets, path: str, limit: int) -> FileItem:
        """파일 목록 조회

        :param assets: 대상 Assets Object

        :param path: 대상 path '/...'

        :param limit: 페이징 처리 한번에 가져올 목록 개수

        :return: :py:class:`FileItem`
        """
        paths = [x for x in path.split('/') if len(x) > 0]
        path = os.path.join(*paths) if len(paths) > 0 else ''
        if len(path) > 1:
            path += '/'
        page = 1
        while True:
            api_resp = self.get(
                AssetHubAPI.URLS["assets_ls"].format(assets.assets['id'],
                                                     assets.revision['commit_id'],
                                                     path,
                                                     limit,
                                                     page))
            if not api_resp.success():
                self.logger.error(f"Can't get list {api_resp.errors}")
                return

            data = api_resp.data
            total_items = data['count']
            total_page = floor(total_items / limit) + 1
            dirs = data['dirs']
            for d in dirs:
                yield FileItem('D', d['name'], d['update_date'], 0)
            files = data['files']
            for f in files:
                yield FileItem('F', f['name'], f['update_date'], f['size'])
            page += 1
            if page >= total_page:
                break

    def lock_acquire(self, assets_id: int, last_commit_id: str) -> Optional[dict]:
        """잠금 시도

        :param assets_id: assets_id

        :param last_commit_id: commit_id

        :return dict: "lock_token": str, "expired_at": datetime
        """
        api_resp = self.get(
            AssetHubAPI.URLS["lock_acquire"].format(assets_id,
                                                    last_commit_id))
        if api_resp.success():
            return api_resp.data
        self.logger.error(f"{api_resp.message, api_resp.errors}", False)
        return None

    def lock_refresh(self, assets_id: int, lock_token: str) -> Optional[dict]:
        """잠금 유지

        :param assets_id: assets_id
        :param lock_token:  :py:func:`lock_acquire`  에서 수신한 lock_token
        :return dict: "lock_token": str, "expired_at": datetime
        """
        api_resp = self.post(
            AssetHubAPI.URLS["lock_refresh"].format(assets_id),
            data={
                "lock_token": lock_token
            })
        if api_resp.success():
            return api_resp.data
        self.logger.error(f"{api_resp.message, api_resp.errors}", False)
        return None

    def lock_release(self, assets_id: int, lock_token: str):
        """잠금 해제

        :param assets_id: assets_id
        :param lock_token:  :py:func:`lock_acquire`  에서 수신한 lock_token
        """
        api_resp = self.post(
            AssetHubAPI.URLS["lock_release"].format(assets_id),
            data={
                "lock_token": lock_token
            })
        if api_resp.success():
            return
        self.logger.error(f"{api_resp.message, api_resp.errors}", False)
        return

    def add_files(self, assets_id: int, last_commit_id: str, lock_token: str, files: dict) -> APIResponse:
        """다중 파일 추가

        :param assets_id: assets_id
        :param last_commit_id: 해당 Assets 의 최신 commit_id
        :param lock_token: :py:func:`lock_acquire`  에서 수신한 lock_token
        :param files: 파일 목록
        :return APIResponse:
        """
        return self.post(
            AssetHubAPI.URLS["assets_add_files"].format(assets_id, last_commit_id),
            data={"lock_token": lock_token},
            files=files,
            is_json=False)

    def add_chunk(self, assets_id: int,
                  last_commit_id: str,
                  lock_token: str,
                  chunk_number: int,
                  total_chunks: int,
                  files: dict) -> APIResponse:
        """분할 파일 전송

        :param assets_id: assets_id
        :param last_commit_id: 해당 Assets 의 최신 commit_id
        :param lock_token: :py:func:`lock_acquire`  에서 수신한 lock_token
        :param chunk_number: 분할 데이터 인덱스
        :param total_chunks: 총 분할 데이터 개수
        :param files: 분할 파일 데이터
        :return APIResponse:
        """
        return self.post(
            AssetHubAPI.URLS["assets_add_chunk"].format(assets_id, last_commit_id),
            data={
                "lock_token": lock_token,
                "chunk_number": chunk_number,
                "total_chunks": total_chunks
            },
            files=files,
            is_json=False)

    def publish(self, assets_id: int, commit_id: str, summary: str) -> APIResponse:
        """에셋 배포

        :param assets_id: assets_id
        :param commit_id: commit_id
        :param summary: 사용자 코멘트
        :return APIResponse:
        """
        return self.post(
            AssetHubAPI.URLS["publish"].format(assets_id, commit_id),
            data={
                "summary": summary
            }
        )

    def publish_state(self, assets_id: int, state_token: str) -> APIResponse:
        """에셋 배포 상태 확인

        :param assets_id: assets_id
        :param state_token:  :py:func:`publish`  에서 수신한 state_token
        :return:
        """
        return self.post(
            AssetHubAPI.URLS["publish_state"].format(assets_id),
            data={
                "state_token": state_token
            })
