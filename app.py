"""User interface (Streamlit).  Run:  streamlit run app.py
Meets the handout's UI list: load/save dataset, process, view markers, view trajectory+map, diagnostics."""
import json
import os
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="VI-SLAM with ArUco", layout="wide")
st.title("Visual–Inertial SLAM with ArUco Anchors (IE415)")
mode = st.sidebar.radio("Data source", ["Simulated demo", "Real dataset folder"])
res = None

if mode == "Simulated demo":
    from run_pipeline import analyse
    from src.simulate import simulate
    seed = st.sidebar.number_input("Random seed", 0, 99, 0)
    p_out = st.sidebar.slider("False-detection rate", 0.0, 0.3, 0.03)
    lo = st.sidebar.slider("Marker-loss window start (keyframe)", 0, 120, 55)
    robust = st.sidebar.checkbox("Robust (Huber) marker factors", True)
    if st.sidebar.button("Process", type="primary"):
        sim = simulate(int(seed), loss=(lo, lo + 20), p_out=p_out)
        m, x = analyse(sim, tag="app", robust=robust, plot=False)
        st.session_state["res"] = dict(base=x["base"], fused=x["fused"], gt=x["gt"], markers=x["markers"],
                                       metrics=m, e=(x["e_base"], x["e_fused"]), loss=(lo, lo + 20))
else:
    d = st.sidebar.text_input("Dataset folder", "data/run1")
    step = st.sidebar.number_input("Frame step", 1, 10, 3)
    map_path = st.sidebar.text_input("Saved map for relocalization (optional)", "")
    closed = st.sidebar.checkbox("Closed-loop walk", True)
    st.sidebar.write("Files found:", [f for f in ["video.mp4", "calib.npz", "meta.json", "markers.json", "odom.npz"]
                                      if os.path.exists(f"{d}/{f}")])
    if st.sidebar.button("1. Detect markers"):
        import extract_markers
        st.session_state["det"] = extract_markers.main(d, int(step))
    if st.sidebar.button("2. Run SLAM", type="primary"):
        import run_real
        r = run_real.run(d, int(step), map_path or None, closed)
        st.session_state["res"] = dict(base=r["base"], fused=r["fused"], gt=None, markers=r["markers"],
                                       metrics=r["metrics"], e=None, loss=None)
    if "det" in st.session_state:
        st.sidebar.json(st.session_state["det"])
    if os.path.exists(f"{d}/preview.jpg"):
        st.sidebar.image(f"{d}/preview.jpg", caption="Detected markers (axes drawn)")

res = st.session_state.get("res")
if res:
    c1, c2 = st.columns([3, 2])
    fig = go.Figure()
    if res["gt"] is not None:
        fig.add_trace(go.Scatter3d(x=res["gt"][:, 0], y=res["gt"][:, 1], z=res["gt"][:, 2], 
                                   mode="lines", name="ground truth", line=dict(color="black", width=5)))
    fig.add_trace(go.Scatter3d(x=res["base"][:, 0], y=res["base"][:, 1], z=res["base"][:, 2], 
                               mode="lines", name="odometry only", line=dict(color="red", dash="dash")))
    fig.add_trace(go.Scatter3d(x=res["fused"][:, 0], y=res["fused"][:, 1], z=res["fused"][:, 2], 
                               mode="lines", name="with markers", line=dict(color="green", width=5)))
    mk = np.array(list(res["markers"].values()))
    if len(mk) > 0:
        fig.add_trace(go.Scatter3d(x=mk[:, 0], y=mk[:, 1], z=mk[:, 2], mode="markers+text", 
                                   text=[f"M{i}" for i in res["markers"]],
                                   name="markers", marker=dict(size=6, symbol="square", color="blue")))
    fig.update_layout(scene=dict(aspectmode="data"), height=600, margin=dict(l=0, r=0, t=0, b=0))
    c1.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Diagnostics")
        st.json(res["metrics"], expanded=1)
        if res["e"] is not None:
            f2 = go.Figure()
            f2.add_trace(go.Scatter(y=res["e"][0], name="baseline", line=dict(color="red")))
            f2.add_trace(go.Scatter(y=res["e"][1], name="with markers", line=dict(color="green")))
            f2.add_vrect(x0=res["loss"][0], x1=res["loss"][1], fillcolor="gray", opacity=0.25, annotation_text="no markers")
            f2.update_layout(yaxis_title="position error (m)", xaxis_title="keyframe", height=300)
            st.plotly_chart(f2, use_container_width=True)
else:
    st.info("Choose a data source in the sidebar and press the button.")