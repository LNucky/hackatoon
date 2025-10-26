# -*- coding: utf-8 -*-
# route_optimizer_2gis.py — построение маршрута С УЧЁТОМ ПРОБОК,
# адреса -> геокодинг -> 2ГИС матрица (10×10 чанки) -> окна/обед/приоритет -> 2-opt.
# Полностью игнорируем lat/lon из CSV. Координаты берём только из геокодера.

import os
import csv
import json
import math
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional

from dgis_matrix_client import DGISMatrixClient, DGISMatrixError
from dgis_geocoder import DGISGeocoder

logging.basicConfig(level=logging.ERROR)

# ---------- утилиты времени/форматов ----------

def parse_time_hhmm(s: str) -> datetime:
    s = (s or "").strip()
    return datetime.strptime(s, "%H:%M")

def to_hhmm(dt: datetime) -> str:
    return dt.strftime("%H:%M")

def minutes_between(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 60.0

def safe_float(x) -> Optional[float]:
    try:
        return float(str(x).replace(",", "."))
    except:
        return None

# ======================= ОСНОВНОЙ КЛАСС =======================

class RouteOptimizer2GIS:
    def __init__(self, dgis_api_key: str):
        self.dgis_api_key = dgis_api_key
        self._provider = "2GIS"   # или "OFFLINE"
        # оффлайн-параметры
        self.city_speed_kmh = 28.0
        self.congestion = 1.25

    # ---------- оффлайн-хелперы ----------
    @staticmethod
    def _haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        return 2 * R * math.asin(min(1.0, math.sqrt(a)))

    def _approximate_distance_matrix(self, coordinates: List[Tuple[float, float]]) -> List[List[float]]:
        n = len(coordinates)
        d = [[0.0]*n for _ in range(n)]
        for i in range(n):
            lat1, lon1 = coordinates[i]
            for j in range(n):
                if i == j:
                    continue
                lat2, lon2 = coordinates[j]
                d[i][j] = self._haversine_km(lat1, lon1, lat2, lon2)
        return d

    def _approximate_time_matrix(self, coordinates: List[Tuple[float, float]]) -> List[List[float]]:
        d = self._approximate_distance_matrix(coordinates)
        n = len(d); factor = (60.0 / max(self.city_speed_kmh, 1e-6)) * self.congestion
        t = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                t[i][j] = d[i][j] * factor
        return t

    # ---------- приоритет ----------
    def _parse_priority(self, v) -> int:
        if v is None: return 1
        s = str(v).strip().lower()
        if s.isdigit():
            try: return int(s)
            except: pass
        mapping = {
            "vip": 3, "vip+": 4, "platinum": 4, "gold": 3, "silver": 2,
            "standart": 1, "standard": 1, "std": 1, "base": 1, "basic": 1,
            "a": 3, "b": 2, "c": 1,
            "high": 3, "medium": 2, "low": 1,
        }
        return mapping.get(s, 1)

    # ---------- чтение CSV (Только адреса! Координаты из CSV игнорируем) ----------
    def _read_csv(self, csv_path: str) -> Dict[str, Any]:
        rows = []
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                address = (r.get("Адрес объекта") or r.get("Адрес") or "").strip()
                if not address:
                    # пустой адрес — пропустим
                    continue
                rows.append({
                    "id":           r.get("Номер объекта") or r.get("Номер Объекта") or r.get("ID") or str(len(rows)+1),
                    "address":      address,
                    # координаты из CSV игнорируем: lat=None, lon=None
                    "lat":          None,
                    "lon":          None,
                    "prio":         self._parse_priority(r.get("Уровень клиента")),
                    "win_start":    (r.get("Время начала рабочего дня") or "09:00").strip(),
                    "win_end":      (r.get("Время окончания рабочего дня") or "18:00").strip(),
                    "lunch_start":  (r.get("Время начала обеда") or "13:00").strip(),
                    "lunch_end":    (r.get("Время окончания обеда") or "14:00").strip(),
                })
        if not rows:
            raise RuntimeError("CSV пустой или нет валидных адресов")
        return {"rows": rows}

    # ---------- геокодинг адресов ----------
    def _geocode_rows(self, rows: List[Dict[str, Any]]) -> Tuple[List[Tuple[float, float]], List[str], List[str]]:
        geocoder = DGISGeocoder(api_key=os.getenv("DGIS_API_KEY", ""), city_hint="Ростов-на-Дону", country_hint="Россия")
        coords: List[Tuple[float, float]] = []
        addresses: List[str] = []
        ids: List[str] = []
        dropped_idx = []

        for idx, r in enumerate(rows):
            adr = r["address"]
            res = geocoder.geocode_one(adr)
            if not res:
                # не смогли геокодить — выкидываем точку
                dropped_idx.append(idx)
                logging.warning("Не удалось геокодировать адрес: %s", adr)
                continue
            lat, lon = res
            coords.append((lat, lon))
            addresses.append(adr)
            ids.append(r["id"])

        if not coords:
            raise RuntimeError("После геокодинга не осталось валидных координат")
        return coords, addresses, ids

    # ---------- матрицы времени/дистанции ----------
    def _get_matrices(self, coords) -> Tuple[List[List[float]], List[List[float]]]:
        try:
            client = DGISMatrixClient(self.dgis_api_key)  # chunk_size=10 по умолчанию
            tm, dm = client.get_time_distance_matrices(coords)
            self._provider = "2GIS"
            return tm, dm
        except DGISMatrixError as e:
            logging.exception("2ГИС недоступен/ошибка ключа или лимиты: %s", e)
        except Exception as e:
            logging.exception("Неожиданная ошибка 2ГИС: %s", e)
        # оффлайн
        self._provider = "OFFLINE"
        dm = self._approximate_distance_matrix(coords)
        tm = self._approximate_time_matrix(coords)
        return tm, dm

    # ---------- обед / окна ----------
    @staticmethod
    def _fits_lunch(service_start: datetime, service_end: datetime, lunch_start: datetime, lunch_end: datetime) -> Tuple[datetime, datetime, bool]:
        if service_start < lunch_end and service_end > lunch_start:
            shift = (service_end - service_start)
            ss = lunch_end
            ee = lunch_end + shift
            return ss, ee, True
        return service_start, service_end, False

    # ---------- жадный выбор ----------
    def _choose_next(self, cur_idx: int, now: datetime, unvis: set, time_mat, meta, meet_minutes, day_end):
        W_WAIT = 1.15; VIP_BOOST = 0.15; LUNCH_HIT = 15.0
        best = None; best_score = None; details = None
        for j in list(unvis):
            travel_min = time_mat[cur_idx][j]
            if math.isinf(travel_min): continue
            arrive = now + timedelta(minutes=travel_min)
            w_start = meta[j]["win_start_abs"]; w_end = meta[j]["win_end_abs"]
            start_srv = max(arrive, w_start); end_srv = start_srv + timedelta(minutes=meet_minutes)
            l_start = meta[j]["lunch_start_abs"]; l_end = meta[j]["lunch_end_abs"]
            moved_by_lunch = False
            if start_srv < l_end and end_srv > l_start:
                moved_by_lunch = True
                start_srv, end_srv, _ = self._fits_lunch(start_srv, end_srv, l_start, l_end)
            if end_srv > day_end: continue
            wait_min = max(0.0, minutes_between(arrive, start_srv))
            prio = meta[j]["prio"]
            score = (travel_min + W_WAIT*wait_min) / (1.0 + VIP_BOOST*prio)
            if moved_by_lunch: score += LUNCH_HIT
            if best_score is None or score < best_score - 1e-6:
                best, best_score = j, score
                details = {"arrive": arrive, "start": start_srv, "end": end_srv,
                           "travel_min": travel_min, "wait_min": wait_min, "moved_by_lunch": moved_by_lunch}
        return best, details

    # ---------- оценка и 2-opt ----------
    def _eval_route(self, route, time_mat, meta, meeting_minutes, day_start, day_end):
        W_WAIT = 1.10
        cur = route[0]
        now = max(day_start, meta[cur]["win_start_abs"])
        end_first = now + timedelta(minutes=meeting_minutes)
        l_s = meta[cur]["lunch_start_abs"]; l_e = meta[cur]["lunch_end_abs"]
        if now < l_e and end_first > l_s:
            now, end_first, _ = self._fits_lunch(now, end_first, l_s, l_e)
        if end_first > day_end: return None
        schedule = [{"idx": cur, "arrive": now, "start": now, "end": end_first,
                     "travel_min": 0.0, "wait_min": 0.0, "moved_by_lunch": False}]
        total_drive = 0.0; total_wait = 0.0
        cur_time = end_first; cur_idx = cur
        for j in route[1:]:
            travel = time_mat[cur_idx][j]
            if math.isinf(travel): return None
            arrive = cur_time + timedelta(minutes=travel)
            w_start = meta[j]["win_start_abs"]; w_end = meta[j]["win_end_abs"]
            start_srv = max(arrive, w_start); end_srv = start_srv + timedelta(minutes=meeting_minutes)
            l_s = meta[j]["lunch_start_abs"]; l_e = meta[j]["lunch_end_abs"]
            moved = False
            if start_srv < l_e and end_srv > l_s:
                moved = True
                start_srv, end_srv, _ = self._fits_lunch(start_srv, end_srv, l_s, l_e)
            if end_srv > day_end: return None
            wait = max(0.0, minutes_between(arrive, start_srv))
            total_drive += travel; total_wait += wait
            schedule.append({"idx": j, "arrive": arrive, "start": start_srv, "end": end_srv,
                             "travel_min": round(travel,1), "wait_min": round(wait,1), "moved_by_lunch": moved})
            cur_time = end_srv; cur_idx = j
        cost = total_drive + W_WAIT*total_wait
        return cost, schedule, total_drive, total_wait

    def _local_search_2opt(self, route, time_mat, meta, meeting_minutes, day_start, day_end, max_iters=150):
        best_eval = self._eval_route(route, time_mat, meta, meeting_minutes, day_start, day_end)
        if best_eval is None: return route, None
        best_cost, best_schedule, best_drive, best_wait = best_eval
        best_route = route[:]
        improved = True; iters = 0
        while improved and iters < max_iters:
            improved = False; iters += 1
            for i in range(1, len(best_route)-2):
                for j in range(i+1, len(best_route)-1):
                    cand = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                    ev = self._eval_route(cand, time_mat, meta, meeting_minutes, day_start, day_end)
                    if ev is None: continue
                    cost, _, _, _ = ev
                    if cost + 1e-6 < best_cost:
                        best_cost = cost; best_route = cand; best_eval = ev; improved = True; break
                if improved: break
        return best_route, best_eval

    # ---------- метрики плеч ----------
    def _attach_leg_metrics(self, route, schedule, dist_mat):
        for k in range(len(schedule)):
            if k == 0:
                schedule[k]["leg_km"] = 0.0
                schedule[k]["avg_kmh"] = 0.0
            else:
                a = route[k-1]; b = route[k]
                km = float(dist_mat[a][b])
                schedule[k]["leg_km"] = round(km, 2)
                tm = float(schedule[k]["travel_min"])
                schedule[k]["avg_kmh"] = round(km / (tm / 60.0), 1) if tm > 0 else 0.0

    # ---------- стартовая точка ----------
    def _select_start(self, meta: List[Dict[str, Any]]) -> int:
        earliest = None; idx = 0
        for i, m in enumerate(meta):
            if earliest is None or m["win_start_abs"] < earliest:
                earliest = m["win_start_abs"]; idx = i
        return idx

    # ---------- основной метод ----------
    def optimize_from_csv(self, csv_path: str, work_start_str="09:00", work_end_str="18:00", meeting_minutes: int = 30) -> Dict[str, Any]:
        raw = self._read_csv(csv_path)
        rows = raw["rows"]

        # 1) Геокодирование адресов (полностью игнорируем lat/lon из CSV)
        coords, addresses, ids = self._geocode_rows(rows)

        # 2) Матрицы времени/дистанции
        print("🔄 Получение данных о времени пути (2ГИС/пробки)...")
        time_mat, dist_mat = self._get_matrices(coords)

        # 3) Временные параметры
        today = datetime.today().replace(second=0, microsecond=0)
        day_start = today.replace(hour=parse_time_hhmm(work_start_str).hour,
                                  minute=parse_time_hhmm(work_start_str).minute)
        day_end   = today.replace(hour=parse_time_hhmm(work_end_str).hour,
                                  minute=parse_time_hhmm(work_end_str).minute)
        if day_end <= day_start:
            day_end = day_start + timedelta(hours=9)

        # 4) Метаданные точек
        meta = []
        # синхронизация окон/обеда с исходными строками: берём только те, которые прошли геокодинг
        kept_indices = []
        k = 0
        for idx, r in enumerate(rows):
            # если адрес этого r попал в addresses (в том же порядке), добавляем мету
            if k < len(addresses) and r["address"] == addresses[k]:
                w_s = parse_time_hhmm(r["win_start"]); w_e = parse_time_hhmm(r["win_end"])
                l_s = parse_time_hhmm(r["lunch_start"]); l_e = parse_time_hhmm(r["lunch_end"])
                meta.append({
                    "win_start_abs": today.replace(hour=w_s.hour, minute=w_s.minute),
                    "win_end_abs":   today.replace(hour=w_e.hour, minute=w_e.minute),
                    "lunch_start_abs": today.replace(hour=l_s.hour, minute=l_s.minute),
                    "lunch_end_abs":   today.replace(hour=l_e.hour, minute=l_e.minute),
                    "prio": r["prio"]
                })
                kept_indices.append(idx)
                k += 1
            else:
                # этот адрес был отброшен на этапе геокодинга
                pass

        n = len(addresses)

        # 5) Жадная конструкция
        route: List[int] = []
        visited = set(); unvis = set(range(n))

        cur = self._select_start(meta)
        route.append(cur); visited.add(cur); unvis.remove(cur)

        now = max(day_start, meta[cur]["win_start_abs"])
        end_first = now + timedelta(minutes=meeting_minutes)
        ns, ne, moved = self._fits_lunch(now, end_first, meta[cur]["lunch_start_abs"], meta[cur]["lunch_end_abs"])
        if moved: now, end_first = ns, ne

        schedule: List[Dict[str, Any]] = [{
            "idx": cur, "id": ids[cur], "address": addresses[cur],
            "arrive": to_hhmm(now), "start": to_hhmm(now),
            "end": to_hhmm(end_first), "travel_min": 0.0, "wait_min": 0.0
        }]
        now = end_first

        while unvis:
            j, det = self._choose_next(cur, now, unvis, time_mat, meta, meeting_minutes, day_end)
            if j is None: break
            route.append(j); unvis.remove(j); visited.add(j)
            cur = j; now = det["end"]
            schedule.append({
                "idx": j, "id": ids[j], "address": addresses[j],
                "arrive": to_hhmm(det["arrive"]), "start": to_hhmm(det["start"]),
                "end": to_hhmm(det["end"]), "travel_min": round(det["travel_min"],1),
                "wait_min": round(det["wait_min"],1), "moved_by_lunch": det["moved_by_lunch"]
            })

        dropped_idx = []  # отброшенные на этапе геокодинга мы не включали изначально

        # 6) 2-opt локальный поиск
        improved_route, ev = self._local_search_2opt(route, time_mat, meta, meeting_minutes, day_start, day_end)
        if ev is not None and improved_route != route:
            cost, sched2, total_drive2, total_wait2 = ev
            schedule = []
            for s in sched2:
                idx = s["idx"]
                schedule.append({
                    "idx": idx, "id": ids[idx], "address": addresses[idx],
                    "arrive": to_hhmm(s["arrive"]), "start": to_hhmm(s["start"]),
                    "end": to_hhmm(s["end"]), "travel_min": s["travel_min"],
                    "wait_min": s["wait_min"], "moved_by_lunch": s["moved_by_lunch"]
                })
            route = improved_route

        # 7) Метрики плеч (км и средняя скорость)
        self._attach_leg_metrics(route, schedule, dist_mat)

        # 8) Агрегаты
        total_drive_min = 0.0
        for i in range(len(route)-1):
            a, b = route[i], route[i+1]
            total_drive_min += time_mat[a][b]
        total_drive_min = round(total_drive_min, 1)

        total_km = round(sum(s.get("leg_km", 0.0) for s in schedule), 2)

        result = {
            "external_provider": self._provider,            # "2GIS" или "OFFLINE"
            "uses_external_api": (self._provider == "2GIS"),
            "meeting_minutes": meeting_minutes,
            "work_window": {"start": to_hhmm(day_start), "end": to_hhmm(day_end)},
            "n_points": n,
            "visited_order_idx": route,
            "visited_order_ids": [ids[k] for k in route],
            "schedule": schedule,
            "dropped_idx": dropped_idx,
            "dropped_ids": [],  # уже отброшены до построения
            "stats": {
                "total_drive_min": total_drive_min,
                "total_distance_km": total_km,
                "visited_count": len(route),
                "dropped_count": len(dropped_idx)
            }
        }
        return result

# ---------- сохранение JSON ----------
def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
