import streamlit as st
import folium
from streamlit.components.v1 import html
import os
import time
import threading
import json
from pipeline2 import parse_kml_points, generate_points, process_point, process_route, STATUS_FILE

st.set_page_config(layout='wide', page_title='360 Extractor UI')

st.title('360 Street View Extractor — UI')

kml_file = st.text_input('KML file', value='route.kml')
step_m = st.number_input('Interpolation step (meters)', value=5, min_value=1)

# Auto-refresh every 2s when enabled
auto = st.sidebar.checkbox('Live updates (auto-refresh every 2s)', value=True)
if auto:
    # Simple built-in auto-refresh: rerun the Streamlit script every interval seconds
    interval = 2.0
    now = time.time()
    last = st.session_state.get('last_refresh', 0)
    if now - last > interval:
        st.session_state['last_refresh'] = now
        # Trigger a rerun by updating a query parameter (works without extra deps)
        st.experimental_set_query_params(_r=int(now))

if 'raw_coords' not in st.session_state:
    st.session_state.raw_coords = parse_kml_points(kml_file)
    st.session_state.points = generate_points(st.session_state.raw_coords, step_m)
    st.session_state.thread = None

st.sidebar.header('Controls')
start = st.sidebar.button('Start / Resume')
stop = st.sidebar.button('Stop')
step = st.sidebar.button('Process next point')
refresh = st.sidebar.button('Refresh Status')

# Camera object-preservation toggles
st.sidebar.markdown('**Keep objects visible on specific cams (no masking)**')
keep_list = []
for j in range(8):
    key = f'keep_cam{j}'
    default = False
    # Load existing config if present
    if os.path.exists('ui_config.json'):
        try:
            cfg = json.load(open('ui_config.json', 'r'))
            default = j in cfg.get('keep_objects_cams', [])
        except Exception:
            default = False
    val = st.sidebar.checkbox(f'Keep objects on cam{j}', value=default, key=key)
    if val:
        keep_list.append(j)

# Save UI config
cfg = {'keep_objects_cams': keep_list}
try:
    with open('ui_config.json', 'w') as f:
        json.dump(cfg, f)
except Exception:
    pass

# Read status early (so we can display processed panos on the map)
status = {}
if os.path.exists(STATUS_FILE):
    try:
        with open(STATUS_FILE,'r') as f:
            status = json.load(f)
    except Exception:
        status = {}

# Map
center = st.session_state.raw_coords[0]
m = folium.Map(location=[center[0], center[1]], zoom_start=17)
# KML polyline
folium.PolyLine([[lat,lon] for lat,lon in st.session_state.raw_coords], color='orange', weight=4).add_to(m)

# Points markers (interpolated)
for i, p in enumerate(st.session_state.points):
    color = 'red' if p['type']=='kml' else 'blue'
    folium.CircleMarker(location=[p['lat'], p['lon']], radius=3, color=color, fill=True, fill_color=color, popup=f"{i}: {p['type']}").add_to(m)

# Processed panos markers
processed = status.get('processed', [])
for proc in processed:
    try:
        lat = proc.get('lat')
        lon = proc.get('lon')
        idx = proc.get('pano_index')
        stt = proc.get('status')
        color = 'green' if stt == 'done' else 'red'
        folium.CircleMarker(location=[lat, lon], radius=5, color=color, fill=True, fill_color=color, popup=f"pano_{idx}: {stt}").add_to(m)
    except Exception:
        pass

# Status panel (auto-updating)
if status:
    st.sidebar.subheader('Status')
    st.sidebar.json(status)
    st.sidebar.markdown('---')
    st.sidebar.write(f"Processing index: {status.get('interp_index', status.get('index', 'n/a'))}")
    st.sidebar.write(f"Status: {status.get('status')}")
    st.sidebar.write(f"Pano: {status.get('pano_id', 'n/a')}")

# Progress bar
total = status.get('total_interp_points', len(st.session_state.points))
processed_count = len(processed)
progress_pct = int(processed_count / max(1, total) * 100)
st.sidebar.progress(progress_pct)
st.sidebar.write(f"Processed: {processed_count}/{total} ({progress_pct}%)")
# Auto-refresh info
if auto:
    st.sidebar.write('Auto-refresh is ON — UI updates every 2s')
else:
    st.sidebar.write('Auto-refresh is OFF — use Refresh Status')

# Start/Stop/Step actions
if start:
    st.sidebar.write('Starting...')
    def background_run():
        process_route(st.session_state.raw_coords, step_m)
    if st.session_state.thread is None or not st.session_state.thread.is_alive():
        st.session_state.thread = threading.Thread(target=background_run, daemon=True)
        st.session_state.thread.start()

if stop:
    st.sidebar.write('Stopping...')
    # write stop command to status
    with open(STATUS_FILE, 'w') as f:
        json.dump({'cmd': 'stop'}, f)

if step:
    st.sidebar.write('Processing single step...')
    # read index
    s = {}
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE,'r') as f:
            try: s = json.load(f)
            except: s = {}
    idx = s.get('index', -1) + 1
    points = st.session_state.points
    if idx < len(points):
        process_point(points[idx], idx, st.session_state.raw_coords)
        # trigger a lightweight rerun by changing query params
        st.experimental_set_query_params(step=int(time.time()))
    else:
        st.sidebar.write('No more points')

# Embed map
map_html = m._repr_html_()
html(map_html, height=600)

# Show current point thumbnails if available (auto-refresh picks up new files)
current_idx = status.get('interp_index', status.get('index')) if status else None
if current_idx is not None:
    idx = current_idx
    base = f"pano_{idx:04d}"
    st.subheader(f'Point {idx} thumbnails')
    cols = st.columns(4)
    for j in range(8):
        img_path = f"output/images/{base}_cam{j}.jpg"
        mask_path = f"output/masks/{base}_cam{j}_mask.png"
        caption = f'cam{j}'
        # Indicate if this cam keeps objects
        try:
            cfg = json.load(open('ui_config.json','r'))
            if j in cfg.get('keep_objects_cams',[]):
                caption += ' (keep objects)'
        except Exception:
            pass
        if os.path.exists(img_path):
            cols[j%4].image(img_path, caption=caption, use_column_width=True)
            if os.path.exists(mask_path):
                cols[j%4].write('mask: available')
        else:
            cols[j%4].write(f'cam{j} missing')

st.markdown('---')
st.markdown('**Guide**: Start the pipeline with the Start button. Use Live updates to see progress and thumbnails. Use the cam toggles to keep objects visible on specific cams (no removal).')

st.text('Tip: use "Process next point" to step through points interactively.')