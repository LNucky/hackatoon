# -*- coding: utf-8 -*-
# dgis_matrix_client.py — 2ГИС Distance Matrix (с пробками, чанк 10×10, ретраи, ясные ошибки)
import time
import json
import logging
import requests
from typing import List, Tuple, Dict, Any

class DGISMatrixError(RuntimeError):
    pass

class DGISMatrixClient:
    """
    POST https://routing.api.2gis.com/get_dist_matrix?key=...&version=2.0
    body: {
      "points":[{"lat":..,"lon":..}, ...],
      "sources":[...],
      "targets":[...],
      "type":"jam" | "statistics"
    }
    Ответ: {
      "generation_time": ...,
      "routes":[{source_id,target_id,distance(m),duration(s),status,...}, ...]
    }
    """
    BASE_URL = "https://routing.api.2gis.com/get_dist_matrix"

    def __init__(self, api_key: str, version: str = "2.0",
                 chunk_size: int = 10,      # <= лимит синхронного API на демо
                 max_retries: int = 3,
                 sleep_between: float = 0.2):
        if not api_key:
            raise DGISMatrixError("DGIS_API_KEY is empty")
        self.api_key = api_key
        self.version = version
        self.chunk = int(chunk_size)
        self.max_retries = int(max_retries)
        self.sleep_between = float(sleep_between)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "centinvest-hack/1.0 (+route-optimizer)",
            "Accept": "application/json",
            "Content-Type": "application/json"
        })

    # ---------- низкоуровневый вызов ----------
    def _one_call(self, points: List[Dict[str, float]], src_ids: List[int], tgt_ids: List[int]) -> Dict[str, Any]:
        payload = {
            "points": points,
            "sources": src_ids,
            "targets": tgt_ids,
            "type": "jam"  # текущие пробки
        }
        params = {"key": self.api_key, "version": self.version}
        r = self.session.post(self.BASE_URL, params=params, data=json.dumps(payload), timeout=90)
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            body = r.text[:500]
            raise DGISMatrixError(f"{e}; body={body}") from e
        return r.json()

    # ---------- быстрая проверка ключа ----------
    def warmup(self) -> None:
        payload = {
            "points": [{"lat": 47.222078, "lon": 39.720349},
                       {"lat": 47.223000, "lon": 39.721000}],
            "sources": [0],
            "targets": [1],
            "type": "jam"
        }
        params = {"key": self.api_key, "version": self.version}
        r = self.session.post(self.BASE_URL, params=params, data=json.dumps(payload), timeout=30)
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            raise DGISMatrixError(f"Warmup failed: {e}; body={r.text[:500]}") from e

    # ---------- публичный метод получения матриц ----------
    def get_time_distance_matrices(self, coords_latlon: List[Tuple[float, float]]) -> Tuple[List[List[float]], List[List[float]]]:
        n = len(coords_latlon)
        points = [{"lat": lat, "lon": lon} for lat, lon in coords_latlon]
        time_min = [[0.0]*n for _ in range(n)]
        dist_km  = [[0.0]*n for _ in range(n)]

        # Проверяем ключ мини-запросом (и ловим явные 403/ключевые ошибки заранее)
        self.warmup()

        # Режем всю матрицу пачками по 10×10 (требование демо-ключа)
        self._fill_by_chunks(points, time_min, dist_km, self.chunk)
        return time_min, dist_km

    def _fill_by_chunks(self, points, time_min, dist_km, chunk):
        n = len(points)
        for si in range(0, n, chunk):
            for tj in range(0, n, chunk):
                src_ids = list(range(si, min(si+chunk, n)))
                tgt_ids = list(range(tj, min(tj+chunk, n)))

                retries = 0
                while True:
                    try:
                        data = self._one_call(points, src_ids, tgt_ids)
                        break
                    except DGISMatrixError as e:
                        msg = str(e).lower()
                        # если прилетит «permissible dimension ... 10x10» — это нормально, мы уже на 10×10
                        if "permissible dimension" in msg and "10x10" in msg:
                            raise DGISMatrixError("2GIS demo key allows only 10x10 chunks; already using 10x10 but got refusal") from e
                        # квоты и временные ошибки — мягкий ретрай
                        if any(s in msg for s in ("quota", "limit", "too many", "429", "timeout")) and retries < self.max_retries:
                            retries += 1
                            time.sleep(0.8 * retries)
                            continue
                        # неключевые 5xx — тоже попробуем ретрай
                        if (" 5" in msg) and retries < self.max_retries:
                            retries += 1
                            time.sleep(0.8 * retries)
                            continue
                        # всё остальное — пробрасываем
                        raise

                routes = data.get("routes", [])
                for item in routes:
                    i = item["source_id"]; j = item["target_id"]
                    if item.get("status") == "OK":
                        dist_km[i][j]  = float(item.get("distance", 0.0)) / 1000.0
                        time_min[i][j] = float(item.get("duration", 0.0)) / 60.0
                    else:
                        dist_km[i][j]  = float("inf")
                        time_min[i][j] = float("inf")

                time.sleep(self.sleep_between)
