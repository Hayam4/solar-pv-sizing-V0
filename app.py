"""PV String / MPPT Allocation — web app (Streamlit).  Run:  streamlit run app.py"""
import pandas as pd
import streamlit as st

import pv_core

st.set_page_config(page_title="PV String / MPPT Allocation", page_icon="☀️", layout="wide")
st.title("☀️ PV String / MPPT Allocation")
st.caption("Enter the project, press **Run**, then copy the PVsyst sub-arrays or download the Excel file.")

# ----------------------------------------------------------------------------- settings (sidebar)
with st.sidebar:
    st.header("Project settings")
    module_wp = st.number_input("Module power (Wp)", 100, 1000, 630, 5)
    c1, c2 = st.columns(2)
    min_len = c1.number_input("Min modules / string", 1, 40, 17)
    max_len = c2.number_input("Max modules / string", 1, 40, 18)
    inv_ratio = st.number_input("Max inverter DC/AC", 0.5, 2.0, 1.30, 0.01, format="%.2f")
    mppt_ratio = st.number_input("Max DC/AC per MPPT / sub-array", 0.5, 3.0, 1.33, 0.01, format="%.2f")
    max_strings = st.number_input("Max strings per MPPT (global cap)", 1, 6, 3)
    check_mppt = st.checkbox("Check DC/AC per MPPT strictly", value=False,
                             help="Off = judge by inverter total and by the PVsyst sub-array average. "
                                  "On = every MPPT must stay within the limit (falls back automatically if impossible).")
    allow_pairs = st.checkbox("Allow 2 longest strings on 1 MPPT (blocks)", value=True,
                              help="Only used when no inverter takes 3+ strings per MPPT.")
    st.subheader("Orientation sharing")
    share_pool = st.text_input("Orientations that may share an MPPT", "O1,O2",
                               help="Comma separated names from the orientation table. Empty = no sharing.")
    share_lengths = st.text_input("...only for these string lengths", "17,18")
    with st.expander("Solver"):
        solver_seconds = st.number_input("Solver seconds", 5, 600, 120, 5)
        solver_workers = st.number_input("Solver workers", 1, 32, 8)

# ----------------------------------------------------------------------------- input tables
left, right = st.columns(2)
with left:
    st.subheader("Orientations")
    orient_default = pd.DataFrame([
        {"Orientation": "O1", "Tilt": 5, "Azimuth": -80, "Panels": 590},
        {"Orientation": "O2", "Tilt": 5, "Azimuth": 101, "Panels": 415},
    ])
    orient_df = st.data_editor(orient_default, num_rows="dynamic", key="orient",
                               column_config={"Panels": st.column_config.NumberColumn(min_value=0, step=1)})
with right:
    st.subheader("Inverters (one row per model)")
    inv_default = pd.DataFrame([
        {"Model": "SUN2000-150K-MG0", "AC_kW": 150.0, "MPPTs_per_Inverter": 7, "Strings_per_MPPT": 3, "Quantity": 3},
        {"Model": "SUN2000-50KTL-M3", "AC_kW": 50.0, "MPPTs_per_Inverter": 4, "Strings_per_MPPT": 2, "Quantity": 1},
    ])
    inv_df = st.data_editor(inv_default, num_rows="dynamic", key="inv")

run = st.button("▶ Run allocation", type="primary")

settings = {
    "module_power_wp": int(module_wp), "min_modules_per_string": int(min_len), "max_modules_per_string": int(max_len),
    "max_inverter_dc_ac": float(inv_ratio), "max_mppt_dc_ac": float(mppt_ratio),
    "max_strings_per_mppt": int(max_strings), "solver_time_seconds": int(solver_seconds),
    "solver_workers": int(solver_workers), "check_mppt_ratio": bool(check_mppt),
    "max_mppt_dc_ac_maxlen": float(mppt_ratio), "allow_maxlen_pairs": bool(allow_pairs),
}

if run:
    with st.spinner("Solving… (can take up to a minute for big projects)"):
        st.session_state["result"] = pv_core.run_pipeline(settings, orient_df, inv_df, share_pool, share_lengths)

# ----------------------------------------------------------------------------- results
res = st.session_state.get("result")
if res:
    if res["ok"]:
        st.success("Validation passed")
    elif res.get("errors"):
        st.error("Validation failed — do not use these results")
        for e in res["errors"]:
            st.write("• " + e)
    else:
        st.error("Run failed")

    with st.expander("Run log", expanded=not res["ok"]):
        for line in res["log"]:
            st.write(line)

    if res["ok"]:
        inv_sum = res["inverter_summary_df"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Modules", int(res["validation"]["input_modules"]))
        m2.metric("DC capacity (kWp)", f"{res['validation']['allocated_dc_kWp']:.2f}")
        m3.metric("Strings", len(res["strings_df"]))
        m4.metric("MPPTs used", f"{int(inv_sum['MPPTs_Used'].sum())} / {int(inv_sum['MPPTs_Available'].sum())}")

        t1, t2, t3, t4, t5 = st.tabs(["PVsyst sub-arrays", "Per inverter", "MPPT detail", "Inverter summary", "Strings"])
        with t1:
            st.caption("One sub-array per inverter model + string length (orientations together).")
            st.dataframe(res["pvsyst_df"], hide_index=True)
        with t2:
            st.dataframe(res["pvsyst_inv_df"], hide_index=True)
        with t3:
            st.dataframe(res["detailed_df"], hide_index=True)
        with t4:
            st.dataframe(inv_sum.round(3), hide_index=True)
        with t5:
            st.dataframe(res["string_counts_df"], hide_index=True)
            st.dataframe(res["strings_df"], hide_index=True)

        st.download_button("⬇ Download Excel workbook", data=pv_core.build_excel(res),
                           file_name="PV_Allocation.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
