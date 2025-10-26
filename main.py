# main.py
# FastAPI backend for route optimization (2GIS + fallback).
# Endpoints:
#   GET  /api/health
#   GET  /api/ping
#   POST /api/validate_dgis_key?dgis_api_key=...
#   POST /api/optimize_2gis?dgis_api_key=...&work_start=HH:MM&work_end=HH:MM&meeting_minutes=NN[&city_hint=...&country_hint=...]

import os
import re
import shutil
import tempfile
from typing import List, Dict, Any, Optional
from math import radians, sin, cos, sqrt, atan2

import requests
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from route_optimizer_2gis import RouteOptimizer2GIS
from dgis_geocoder import DGISGeocoder

# ------------------------ App & CORS ------------------------

app = FastAPI(title="Hackathon Route Optimizer API", version="1.2")

# Разрешаем фронту доступ откуда угодно. Если хочешь — зафиксируй список доменов.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # при необходимости перечисли конкретные origin'ы
    allow_credentials=False,      # не ставь True вместе с "*" — браузеры ругаются
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ------------------------ Utils ------------------------

UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

def _save_upload_to_tmp(upload: UploadFile, expected_exts=(".csv",)) -> str:
    """Persist UploadFile to a temp .csv and return its path."""
    name = (upload.filename or "").lower()
    if not any(name.endswith(ext) for ext in expected_exts):
        raise HTTPException(status_code=400, detail="Ожидается CSV (.csv)")
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        with tmp as f:
            shutil.copyfileobj(upload.file, f)
        return tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Не удалось сохранить файл: {e}")

def _fmt_hhmm(val: Optional[str]) -> str:
    return val or ""

def _format_text_report(core: Dict[str, Any]) -> str:
    """Тот же читаемый отчёт, что выводил CLI."""
    provider = core.get("external_provider", "OFFLINE")
    uses_api = core.get("uses_external_api", False)
    w = core.get("work_window") or {}
    stats = core.get("stats") or {}
    schedule: List[Dict[str, Any]] = core.get("schedule", [])
    n_points = core.get("n_points", len(schedule))
    visited = stats.get("visited_count", len(schedule))
    dropped = stats.get("dropped_count", 0)
    total_km = stats.get("total_distance_km", 0.0)
    total_drive_min = stats.get("total_drive_min", 0.0)

    lines = []
    lines.append("=== РЕЗУЛЬТАТЫ ===")
    lines.append(f"Провайдер матрицы: {provider} (uses_external_api={uses_api})")
    lines.append(f"Рабочее окно: {_fmt_hhmm(w.get('start'))} - {_fmt_hhmm(w.get('end'))}")
    lines.append(f"Точек в CSV: {n_points}, посещено: {visited}, пропущено: {dropped}")
    lines.append(f"Итог: {round(total_km, 2)} км, {round(total_drive_min, 1)} мин в движении")
    lines.append("")

    # Список шагов
    for i, s in enumerate(schedule, start=1):
        idx = s.get("id", s.get("idx", "?"))
        addr = s.get("address", "")
        arr = s.get("arrive", "")
        st = s.get("start", "")
        en = s.get("end", "")
        tr = s.get("travel_min", 0.0)
        wt = s.get("wait_min", 0.0)
        moved = s.get("moved_by_lunch", False)
        lines.append(f"{i:02d}. [{idx}] {addr}")
        if moved:
            lines.append(f"    прибытие {arr}, старт {st}, конец {en}, дорога {tr} мин, ожидание {wt} мин (сдвинул из-за обеда)")
        else:
            lines.append(f"    прибытие {arr}, старт {st}, конец {en}, дорога {tr} мин, ожидание {wt} мин")
    return "\n".join(lines)

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

def _geocode_route_points(schedule, city_hint: str, country_hint: str):
    geocoder = DGISGeocoder(
        api_key=os.getenv("DGIS_API_KEY", ""),
        city_hint=city_hint or "",
        country_hint=country_hint or "",
    )

    # Центр города (для Ростова — ~47.2357, 39.7015 как фоллбек)
    center = geocoder.geocode_one(f"{city_hint}, {country_hint}") or \
             geocoder.geocode_one(city_hint) or (47.2357, 39.7015)
    clat, clon = float(center[0]), float(center[1])

    out = []
    for leg in schedule:
        addr = leg.get("address", "")
        lat, lon = None, None
        try:
            res = geocoder.geocode_one(addr)  # (lat, lon)
            if res:
                lat, lon = float(res[0]), float(res[1])
        except Exception:
            pass

        # sanity: диапазоны
        if lat is not None and not (-90 <= lat <= 90):
            lat, lon = lon, lat
        if lon is not None and not (-180 <= lon <= 180):
            lat, lon = lon, lat

        # anti-mirror: если свап заметно ближе к центру города — меняем
        if lat is not None and lon is not None:
            d_ok = _haversine_km(lat, lon, clat, clon)
            d_sw = _haversine_km(lon, lat, clat, clon)
            if d_sw + 100 < d_ok:  # запас 100 км
                lat, lon = lon, lat

        out.append({
            "address": addr,
            "lat": float(lat) if lat is not None else 0.0,
            "lon": float(lon) if lon is not None else 0.0,
            "eta": leg.get("arrive"),
        })
    return out

def _to_frontend_payload(core: Dict[str, Any], meeting_minutes: int,
                         city_hint: str, country_hint: str) -> Dict[str, Any]:
    schedule = core.get("schedule", [])
    stats = core.get("stats", {})

    route_points = _geocode_route_points(schedule, city_hint, country_hint)
    # проставляем длительность визита одинаковую для всех точек (как фронт ожидает)
    for p in route_points:
        p["service_min"] = int(meeting_minutes)

    summary = {
        "drive_min": stats.get("total_drive_min", 0.0),
        "visits": stats.get("visited_count", len(schedule)),
        "late": 0,
        "late_penalty": 0,
    }

    return {
        "route": route_points,
        "summary": summary,
        "text_report": _format_text_report(core),
        "external_provider": core.get("external_provider"),
        "uses_external_api": core.get("uses_external_api"),
        "work_window": core.get("work_window"),
    }

# ------------------------ Endpoints ------------------------

@app.get("/api/health")
def health():
    return {"ok": True, "service": "route-optimizer", "version": "1.2"}

@app.get("/api/ping", response_class=PlainTextResponse)
def ping():
    return "pong"

@app.post("/api/validate_dgis_key")
def validate_dgis_key(dgis_api_key: str = Query(..., description="API ключ 2ГИС")):
    """
    Лёгкая онлайн-проверка ключа 2ГИС.
    1) Проверяем формат UUID (чтобы сразу дать фидбек).
    2) Пингуем лёгкий эндпоинт каталога 2ГИС — 200 ⇒ ок, 401/403 ⇒ ключ отклонён.
    Всегда возвращаем 200 с JSON {"valid": bool, "error": "..."}.
    """
    if not UUID_RE.match(dgis_api_key):
        return {"valid": False, "error": "Неверный формат ключа"}

    try:
        r = requests.get(
            "https://catalog.api.2gis.com/3.0/items",
            params={"q": "test", "page": 1, "page_size": 1, "key": dgis_api_key},
            timeout=3.0,
        )
        if r.status_code == 200:
            return {"valid": True}
        if r.status_code in (401, 403):
            try:
                msg = r.json().get("message") or r.json().get("error_message")
            except Exception:
                msg = None
            return {"valid": False, "error": f"Ключ отклонён 2ГИС ({r.status_code})" + (f": {msg}" if msg else "")}

        try:
            body = r.json()
        except Exception:
            body = r.text[:200]
        return {"valid": False, "error": f"Неожиданный ответ 2ГИС: HTTP {r.status_code} {body}"}

    except requests.RequestException as e:
        return {"valid": False, "error": f"Не удалось связаться с 2ГИС: {e}"}

@app.post("/api/optimize_2gis")
async def optimize_2gis(
    file: UploadFile = File(..., description="CSV с адресами"),
    dgis_api_key: str = Query(..., description="API ключ 2ГИС"),
    work_start: str = Query("09:00", description="Начало рабочего окна, HH:MM"),
    work_end: str = Query("18:00", description="Конец рабочего окна, HH:MM"),
    meeting_minutes: int = Query(30, ge=5, le=240, description="Длительность визита, мин"),
    city_hint: str = Query("Ростов-на-Дону", description="Подсказка геокодеру — город"),
    country_hint: str = Query("Россия", description="Подсказка геокодеру — страна"),
):
    """
    Получаем CSV, запускаем оптимизацию, возвращаем маршрут + summary + текстовый отчёт.
    ВАЖНО: мы возвращаем lat/lon в правильном порядке; на фронте Яндекс ожидает [lat, lon].
    """
    # Прокидываем ключ в окружение — его подхватят клиент матрицы и геокодер
    os.environ["DGIS_API_KEY"] = dgis_api_key

    # 1) Сохраняем файл
    csv_path = _save_upload_to_tmp(file)

    # 2) Оптимизируем
    try:
        optimizer = RouteOptimizer2GIS(dgis_api_key)
        core_result = optimizer.optimize_from_csv(
            csv_path,
            work_start_str=work_start,
            work_end_str=work_end,
            meeting_minutes=meeting_minutes
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка оптимизации маршрута: {e}")
    finally:
        try:
            os.remove(csv_path)
        except Exception:
            pass

    # 3) Сборка ответа в формат фронта
    payload = _to_frontend_payload(
        core_result,
        meeting_minutes=meeting_minutes,
        city_hint=city_hint,
        country_hint=country_hint,
    )
    return JSONResponse(payload)

# ------------------------ Main ------------------------

# Запуск локально:
# uvicorn main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True
    )
