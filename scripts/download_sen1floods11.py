#!/usr/bin/env python3
"""Sen1Floods11 下载器：从 GCS 公开桶拉取所需子集到 ~/datasets/sen1floods11/

只下载 S1Hand（SAR 影像）+ LabelHand（人工标注）+ 官方划分 CSV，
跳过 S2 光学影像等不需要的大层。支持断点续传（已存在且非空的文件自动跳过）。
仅用标准库，无需虚拟环境。
"""
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

BUCKET = "https://storage.googleapis.com/sen1floods11"
PREFIXES = [
    "v1.1/data/flood_events/HandLabeled/S1Hand/",
    "v1.1/data/flood_events/HandLabeled/LabelHand/",
    "v1.1/splits/flood_handlabeled/",
]
DEST = os.path.expanduser("~/datasets/sen1floods11")


def localname(tag):
    return tag.split("}")[-1]


def list_keys(prefix):
    """翻页列出桶内指定前缀下的全部对象 key"""
    keys, marker = [], ""
    while True:
        url = f"{BUCKET}?prefix={urllib.parse.quote(prefix)}&max-keys=1000"
        if marker:
            url += f"&marker={urllib.parse.quote(marker)}"
        with urllib.request.urlopen(url, timeout=60) as r:
            tree = ET.fromstring(r.read())
        batch = [el.text for el in tree.iter() if localname(el.tag) == "Key" and el.text]
        keys += batch
        truncated = [el.text for el in tree.iter() if localname(el.tag) == "IsTruncated"]
        if truncated and truncated[0] == "true" and batch:
            marker = batch[-1]
        else:
            break
    return keys


def fetch(key, tries=3):
    """下载单个对象到本地，.part 临时文件 + 原子改名，失败重试"""
    rel = key[len("v1.1/"):]
    dest = os.path.join(DEST, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "skip", key, 0
    url = f"{BUCKET}/{urllib.parse.quote(key)}"
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r, open(dest + ".part", "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(dest + ".part", dest)
            return "ok", key, os.path.getsize(dest)
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                return "fail", key, str(e)
            time.sleep(2 * (i + 1))


def main():
    all_keys = []
    for p in PREFIXES:
        ks = list_keys(p)
        print(f"[list] {p} -> {len(ks)} 个对象", flush=True)
        all_keys += ks
    print(f"[total] 共 {len(all_keys)} 个文件", flush=True)

    t0, done, new_bytes = time.time(), 0, 0
    fails = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for status, key, info in ex.map(fetch, all_keys):
            done += 1
            if status == "ok":
                new_bytes += info
            elif status == "fail":
                fails += 1
                print(f"[FAIL] {key}: {info}", flush=True)
            if done % 50 == 0 or done == len(all_keys):
                dt = time.time() - t0
                print(f"[progress] {done}/{len(all_keys)} 处理完, "
                      f"新下载 {new_bytes / 1e6:.0f}MB, 耗时 {dt:.0f}s", flush=True)
    print(f"[done] 结束：失败 {fails} 个", flush=True)
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
