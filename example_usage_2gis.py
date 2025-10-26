# -*- coding: utf-8 -*-
# example_usage_2gis.py — запуск оптимизатора на CSV с 2ГИС API
import os
import sys
from route_optimizer_2gis import RouteOptimizer2GIS, save_json

def main():
    print("🚀 ТЕСТ: 2ГИС матрица (пробки) + окна/обед/приоритет")

    # 1) Путь к CSV: берём из аргумента командной строки или используем файл пользователя
    if len(sys.argv) > 1:
        csv_path = os.path.abspath(sys.argv[1])
    else:
        # Файл от пользователя (в этой же папке) — имя без изменения
        base_dir = os.path.dirname(__file__)
        candidate = os.path.join(base_dir, "Data Set - Лист1.csv")
        if os.path.exists(candidate):
            csv_path = candidate
        else:
            # прежний fallback
            csv_path = os.path.abspath(os.path.join(base_dir, "test_list - Лист1.csv"))

    if not os.path.exists(csv_path):
        print("❌ Не найден CSV: ", csv_path)
        print("👉 Запустите так: python example_usage_2gis.py \"/путь/к/Data Set - Лист1.csv\"")
        return

    work_start = (input("Введите время начала работы (HH:MM, напр. 09:00): ").strip() or "09:00")
    work_end   = (input("Введите время окончания работы (HH:MM, напр. 18:00): ").strip() or "18:00")
    meet_str   = (input("Длительность визита, мин (по умолчанию 30): ").strip() or "30")
    try:
        meet_min = int(meet_str)
    except:
        meet_min = 30

    key = os.getenv("DGIS_API_KEY", "")
    if not key:
        print("⚠️  Переменная окружения DGIS_API_KEY пустая — будет оффлайн-оценка (без реального трафика).")

    opt = RouteOptimizer2GIS(key)
    result = opt.optimize_from_csv(csv_path, work_start, work_end, meet_min)

    print("\n=== РЕЗУЛЬТАТЫ ===")
    print(f"Провайдер матрицы: {result['external_provider']} (uses_external_api={result['uses_external_api']})")
    print(f"Рабочее окно: {result['work_window']['start']} - {result['work_window']['end']}")
    print(f"Точек в CSV: {result['n_points']}, посещено: {result['stats']['visited_count']}, пропущено: {result['stats']['dropped_count']}")
    print(f"Итог: {result['stats']['total_distance_km']} км, {result['stats']['total_drive_min']} мин в движении\n")

    for i, leg in enumerate(result["schedule"], 1):
        mv = " (сдвинул из-за обеда)" if leg.get("moved_by_lunch") else ""
        print(f"{i:02d}. [{leg['id']}] {leg['address']}\n    прибытие {leg['arrive']}, старт {leg['start']}, конец {leg['end']}, дорога {leg['travel_min']} мин, ожидание {leg['wait_min']} мин{mv}")

    out_name = "optimization_results_2gis.json"
    out_path = os.path.abspath(os.path.join(os.path.dirname(csv_path), out_name))
    save_json(result, out_path)
    print(f"\n💾 JSON-отчёт сохранён рядом с CSV: {out_path}")

if __name__ == "__main__":
    main()
