# -*- coding: utf-8 -*-
# dgis_geocoder.py — геокодирование "адрес -> (lat, lon)" с кэшем.
# 1) Пытаемся 2ГИС Catalog Geocode (fields=items.point)
# 2) Если не получилось — Nominatim (OSM) как фоллбек
# 3) Результаты кешируем в geocode_cache_2gis.json
import os
import json
import time
import logging
import requests
from typing import Optional, Tuple, Dict, Any, List

class GeocodeError(RuntimeError):
    pass

class DGISGeocoder:
    def __init__(self, api_key: Optional[str], cache_path: str = "geocode_cache_2gis.json",
                 city_hint: str = "Ростов-на-Дону", country_hint: str = "Россия"):
        self.api_key = api_key or ""
        self.cache_path = cache_path
        self.city_hint = city_hint
        self.country_hint = country_hint
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "centinvest-hack/1.0 (+geocoder)",
            "Accept": "application/json"
        })

    # ---------- cache ----------
    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    def _save_cache(self):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.warning("Не удалось сохранить кэш геокодера: %s", e)

    # ---------- 2GIS ----------
    def _geocode_2gis(self, address: str) -> Optional[Tuple[float, float]]:
        # Док: https://catalog.api.2gis.com/3.0/items/geocode?q=...&fields=items.point&key=...
        if not self.api_key:
            return None
        url = "https://catalog.api.2gis.com/3.0/items/geocode"
        q = address
        # Если в адресе нет города, добавим подсказки
        if self.city_hint and self.city_hint not in address:
            q = f"{address}, {self.city_hint}"
        if self.country_hint and self.country_hint not in address:
            q = f"{q}, {self.country_hint}"
        params = {
            "q": q,
            "key": self.api_key,
            "fields": "items.point"
        }
        r = self.session.get(url, params=params, timeout=15)
        if r.status_code == 403:
            # может не быть прав на catalog — пробуем фоллбек
            return None
        r.raise_for_status()
        data = r.json()
        items = data.get("result", {}).get("items", [])
        if not items:
            return None
        pt = items[0].get("point") or {}
        lat = pt.get("lat"); lon = pt.get("lon")
        if lat is None or lon is None:
            return None
        return float(lat), float(lon)

    # ---------- Nominatim fallback ----------
    def _geocode_nominatim(self, address: str) -> Optional[Tuple[float, float]]:
        url = "https://nominatim.openstreetmap.org/search"
        q = address
        if self.city_hint and self.city_hint not in address:
            q = f"{address}, {self.city_hint}"
        if self.country_hint and self.country_hint not in address:
            q = f"{q}, {self.country_hint}"
        params = {"q": q, "format": "json", "limit": 1}
        headers = {"User-Agent": "centinvest-hack/1.0 (+geocoder)"}
        r = self.session.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        arr = r.json()
        if not arr:
            return None
        lat = arr[0].get("lat"); lon = arr[0].get("lon")
        if lat is None or lon is None:
            return None
        return float(lat), float(lon)

    # ---------- public ----------
    def geocode_one(self, address: str) -> Optional[Tuple[float, float]]:
        """
        Геокодирует один адрес -> (lat, lon) с кэшем.
        ВАЖНО: ключ кэша включает city_hint/country_hint, чтобы не ловить старые
        промахи из других городов. Возвращает только валидные float.
        """
        raw = (address or "").strip()
        if not raw:
            return None

        # Новый ключ кэша: адрес + город + страна (избавляет от "битых" ранних записей)
        cache_key = f"{raw} | {self.city_hint} | {self.country_hint}"

        # чтение из кэша
        ent = self._cache.get(cache_key)
        if ent is not None:
            lat = ent.get("lat"); lon = ent.get("lon")
            try:
                if lat is None or lon is None:
                    return None
                return float(lat), float(lon)  # (lat, lon)
            except (TypeError, ValueError):
                return None

        # 1) 2ГИС
        try:
            res = self._geocode_2gis(raw)
            if res:
                lat, lon = float(res[0]), float(res[1])  # гарантируем (lat, lon)
                self._cache[cache_key] = {"lat": lat, "lon": lon, "src": "2gis"}
                self._save_cache()
                time.sleep(0.15)  # вежливая пауза
                return lat, lon
        except Exception as e:
            logging.warning("2ГИС geocode fail: %s", e)

        # 2) Nominatim (OSM)
        try:
            res = self._geocode_osm(raw)
            if res:
                lat, lon = float(res[0]), float(res[1])  # (lat, lon)
                self._cache[cache_key] = {"lat": lat, "lon": lon, "src": "osm"}
                self._save_cache()
                time.sleep(0.15)
                return lat, lon
        except Exception as e:
            logging.warning("Nominatim geocode fail: %s", e)

        # не нашли
        self._cache[cache_key] = {"lat": None, "lon": None, "src": "none"}
        self._save_cache()
        return None


    def batch_geocode(self, addresses: List[str]) -> List[Optional[Tuple[float, float]]]:
        out: List[Optional[Tuple[float, float]]] = []
        for adr in addresses:
            res = self.geocode_one(adr)
            out.append(res)
        return out
