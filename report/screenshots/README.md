# Screenshots taken by the team

`report/build_report.py` picks these files up automatically when they exist.

| File | What to capture |
|---|---|
| `api_laptop1.png` | **Required (Deliverables §7).** The terminal/CMD on Laptop 1 running `uvicorn api.main:app --host 0.0.0.0 --port 8000`, showing a `POST /predict` from Laptop 2's IP address (not 127.0.0.1) and the logged accuracy line. |
| `ui_laptop2.png` | **Required.** The browser on Laptop 2 showing the evaluator UI after selecting a trip, with the API URL set to Laptop 1's IP address. |
| `eda_trip_00000041.png`, `student_trip_<id>.png` | Optional. Re-renders of the Folium maps with the basemap visible. Open `results/maps/<name>.html` on a machine with internet access and screenshot it, or run `node report/capture.js` (Node and Playwright are needed). |

After adding the files, rebuild the report with `python report/build_report.py`.
