from __future__ import annotations

import os
import shutil
from datetime import datetime, timedelta
from typing import Optional
import hashlib

DEFAULT_CACHE_LIMIT = 1024 * 1024 * 1024


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

    def get_file(self, assets, path: str) -> Optional[str]:
        """지정한 파일이 캐시에 존재하면 해당 파일 이름을 리턴, 없을시 캐시로 파일을 다운로드후 해당 이름을 리턴

        """
        now = datetime.now()
        if now - self.disk_usage_chk_time > timedelta(seconds=5):
            self.disk_usage_chk_time = datetime.now()
            size = get_directory_size(self.cache_dir)
            if size > self.limit:
                self.clear_cache(size)

        filename = hashlib.md5(path.encode('utf-8')).hexdigest()
        path_file = os.path.join(self.cache_dir,
                                 str(assets.assets.get('id')),
                                 filename[:2],
                                 filename[2:])
        if os.path.exists(path_file):
            ts = now.timestamp()
            os.utime(path_file, (ts, ts))
            return path_file
        else:
            temp_path_file = path_file + "_"
            if assets.download_file(path, temp_path_file, with_info=False):
                shutil.move(temp_path_file, path_file)
                return path_file
        return None
