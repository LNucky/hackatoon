# main.py
# FastAPI backend that pairs with the provided frontend (index.html/app.js).
# Endpoints:
#   POST /api/optimize_2gis?dgis_api_key=...&work_start=HH:MM&work_end=HH:MM&meeting_minutes=NN[&city_hint=...&country_hint=...]
# Returns JSON: { route: [...], summary: {...}, text_report: "..." }

import os
import shutil
import tempfile
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from route_optimizer_2gis import RouteOptimizer2GIS
from dgis_geocoder import DGISGeocoder

app = FastAPI(title="Hackathon Route Optimizer API", version="1.1")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------- helpers -----------------------

def _save_upload_to_tmp(upload: UploadFile, expected_exts=(".csv",)) -> str:
    """Persist UploadFile to a temp file and return its path."""
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
    """Собираем тот же читаемый отчёт, что печатал example_usage_2gis.py."""
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

def _build_api_response(
    core: Dict[str, Any],
    meeting_minutes: int,
    city_hint: str,
    country_hint: str,
) -> Dict[str, Any]:
    """
    Возвращаем формат, который ждёт фронт.
    ВАЖНО: Яндекс рисует [lon, lat], но в JSON оставляем поля 'lat' и 'lon' как есть.
    """
    schedule = core.get("schedule", [])

    # Геокодим детерминированно (подсказки помогают избегать «не того» города)
    geocoder = DGISGeocoder(
        api_key=os.getenv("DGIS_API_KEY", ""),
        city_hint=city_hint or "",
        country_hint=country_hint or "",
    )

    route_out = []
    for leg in schedule:
        addr = leg.get("address", "")
        lat, lon = None, None
        try:
            res = geocoder.geocode_one(addr)   # res = (lat, lon)
            if res:
                lat, lon = float(res[0]), float(res[1])
        except Exception:
            lat, lon = None, None

        # Страхуемся: если вдруг перепутали — проверим допустимые диапазоны
        if lat is not None and not (-90.0 <= lat <= 90.0):
            lat, lon = lon, lat
        if lon is not None and not (-180.0 <= lon <= 180.0):
            lat, lon = lon, lat

        route_out.append({
            "address": addr,
            "lat": float(lat) if lat is not None else 0.0,
            "lon": float(lon) if lon is not None else 0.0,
            "eta": leg.get("arrive"),
            "service_min": int(meeting_minutes),
        })

    stats = core.get("stats", {})
    summary = {
        "drive_min": stats.get("total_drive_min", 0.0),
        "visits": stats.get("visited_count", len(schedule)),
        "late": 0,
        "late_penalty": 0,
    }

    return {
        "route": route_out,
        "summary": summary,
        "text_report": _format_text_report(core),
        "external_provider": core.get("external_provider"),
        "uses_external_api": core.get("uses_external_api"),
        "work_window": core.get("work_window"),
    }

# ----------------------- endpoints -----------------------

@app.get("/api/health")
def health():
    return {"ok": True, "service": "route-optimizer", "version": "1.1"}

@app.get("/api/ping", response_class=PlainTextResponse)
def ping():
    return "pong"

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
    Получаем CSV, запускаем оптимизацию, возвращаем маршрут + сводку + текстовый отчёт.
    """
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

    # 3) Готовим ответ
    payload = _build_api_response(
        core_result,
        meeting_minutes=meeting_minutes,
        city_hint=city_hint,
        country_hint=country_hint,
    )
    return JSONResponse(payload)

# Локальный запуск:
# uvicorn main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
