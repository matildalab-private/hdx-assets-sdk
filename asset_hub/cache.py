from __future__ import annotations

import io
import os
import shutil
import threading
from datetime import datetime, timedelta
from typing import Optional
import hashlib

DEFAULT_FILE_CACHE_LIMIT = 1024 * 1024 * 1024
DEFAULT_MEMORY_CACHE_LIMIT = 1024 * 1024 * 1024


def get_directory_size(start_path):
    """지정 디렉토리의 용량확인

    """
    total_size = 0
    for dirpath, _dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


class AssetFileCache:
    """허브에 파일 요청시 디스크 캐시를 사용

    """

    def __init__(self,
                 limit: int,
                 cache_dir: Optional[str] = None
                 ):
        if cache_dir is None:
            self.cache_dir = os.path.expanduser('~/.asset_cache')
        else:
            self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.limit = abs(limit)
        self.disk_usage_chk_time = datetime.now()

    def clear_cache(self, cur_size: int):
        """캐시 limit 초과시 오래된 파일들 제거

        """
        remove_size = cur_size - (self.limit * 0.7)
        while remove_size > 0:
            files = []
            for dirpath, _dirnames, filenames in os.walk(self.cache_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if not os.path.islink(fp):
                        size = os.path.getsize(fp)
                        date = datetime.fromtimestamp(os.path.getmtime(fp))
                        files.append((fp, size, date))
                        if len(files) > 8192:
                            files.sort(key=lambda x: x[2])
                            files = files[:4096]

            for f in files:
                try:
                    os.remove(f[0])
                except:
                    ...
                remove_size -= f[1]
                if remove_size <= 0:
                    return

    def get_file(self, assets, path: str) -> Optional[io.BytesIO]:
        """지정한 파일이 캐시에 존재하면 해당 데이터를 리턴, 없을시 캐시로 파일을 다운로드후 리턴

        """
        now = datetime.now()
        # 멀티 프로세스 로더는 파일간 경합이 발생할 수 잇음.
        # 파일 삭제는 사용자 레벨에서 처리해야 할지 확인이 필요
        # if now - self.disk_usage_chk_time > timedelta(seconds=5):
        #     self.disk_usage_chk_time = datetime.now()
        #     size = get_directory_size(self.cache_dir)
        #     if size > self.limit:
        #         self.clear_cache(size)

        filename = hashlib.md5(path.encode('utf-8')).hexdigest()
        path_file = os.path.join(self.cache_dir,
                                 str(assets.assets.get('id')),
                                 filename[:2],
                                 filename[2:])
        if os.path.exists(path_file):
            ts = now.timestamp()
            os.utime(path_file, (ts, ts))
            with open(path_file, "rb") as fd:
                return io.BytesIO(fd.read())
        else:
            temp_path_file = path_file + "_"
            if assets.download_file(path, temp_path_file, with_info=False):
                shutil.move(temp_path_file, path_file)
                with open(path_file, "rb") as fd:
                    return io.BytesIO(fd.read())
        return None


class AssetMemoryCache:
    """허브에 파일 요청시 메모리 캐시를 사용

    멀티 프로세스 로더에서 메모리 캐시가 적용이 되는가..?
    """

    def __init__(self, bytes_limit: int = DEFAULT_MEMORY_CACHE_LIMIT):
        self.bytes_limit = abs(bytes_limit)
        self.cache = {}

    def get_file(self, assets, path: str) -> Optional[io.BytesIO]:
        """지정한 파일이 캐시에 존재하면 해당 데이터를 리턴, 없을시 캐시로 파일을 다운로드후 리턴

        """
        file_data = self.cache.get(path, None)
        if file_data is None:
            total_bytes = sum(
                [v.get('size') for v in self.cache.values()]
            )
            if total_bytes > self.bytes_limit:
                # 정렬
                items = sorted(self.cache.items(), key=lambda x: x[1].get('last_access'))
                # 기존 rc 헤재
                self.cache = {}
                # limit 대비 70% 보존
                cutoff = self.bytes_limit * 0.7
                remain_bytes = total_bytes
                for (k, v) in items:
                    if remain_bytes < cutoff:
                        self.cache[k] = v
                    else:
                        remain_bytes -= v.get('size')

            byteio = assets.load(path, with_info=False)
            file_data = {
                'byteio': byteio,
                'size': byteio.getbuffer().nbytes,
                'last_access': datetime.now()
            }
            self.cache[path] = file_data

        file_data['last_access'] = datetime.now()
        byteio = file_data.get('byteio')
        byteio.seek(0, os.SEEK_SET)
        return byteio


class AssetNoCache:
    def get_file(self, assets, path: str) -> Optional[io.BytesIO]:
        """요청시마다 다운로드

        """
        return assets.load(path, with_info=False)