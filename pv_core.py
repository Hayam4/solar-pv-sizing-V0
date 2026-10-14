"""
PV String / MPPT Allocation — core engine (same logic as the Colab notebook V1).
Used by app.py (Streamlit). No notebook widgets in here.
"""
import io
import pandas as pd
from ortools.sat.python import cp_model


def display(*_args, **_kwargs):   # the notebook helpers call display() only when show=True
    pass


# ---------------------------------------------------------------- cell8
SHAREABLE_ORIENTATION_POOL = globals().get("SHAREABLE_ORIENTATION_POOL", set())
SHARE_LENGTHS = globals().get("SHARE_LENGTHS", {12, 13})
MAX_STRINGS_PER_MPPT = globals().get("MAX_STRINGS_PER_MPPT", 3)
MODULE_KWP = globals().get("MODULE_KWP", 0.630)
ALLOWED_LENGTHS = globals().get("ALLOWED_LENGTHS", {11, 12, 13})
ALLOW_MAXLEN_PAIRS = globals().get("ALLOW_MAXLEN_PAIRS", True)
PAIR_OK_LENGTHS = globals().get("PAIR_OK_LENGTHS", None)          # None = pairs allowed for every length
MAXLEN_BLOCK_OK = globals().get("MAXLEN_BLOCK_OK", {3: True, 4: True, 5: True})
DENSE_MODE = globals().get("DENSE_MODE", False)      # True when some inverter takes 3+ strings per MPPT
S_BY_LENGTH = globals().get("S_BY_LENGTH", {})        # strings per MPPT planned for each string length


def orientations_can_share(orientations):
    orientations = set(orientations)
    if len(orientations) <= 1:
        return True
    return orientations.issubset(SHAREABLE_ORIENTATION_POOL)


def pool_of(orientation, length):
    """Orientations mix only if they are in the sharing list AND the length allows it."""
    if orientation in SHAREABLE_ORIENTATION_POOL and length in SHARE_LENGTHS:
        return "SHARED_POOL"
    return orientation


def get_max_length_blocks(n):
    """
    Blocks for the longest strings (manufacturer pattern):
      5 strings -> 2+2+1 on 3 MPPTs     4 strings -> 2+1+1 on 3 MPPTs     3 strings -> 2+1 on 2 MPPTs
    A block type is only used if its average DC/AC per MPPT is within the limit (MAXLEN_BLOCK_OK);
    otherwise those strings get one MPPT each. If pairs are not allowed at all, every string is alone.
    """
    if n <= 0:
        return []
    if MAX_STRINGS_PER_MPPT < 2 or not ALLOW_MAXLEN_PAIRS:
        return [[1] for _ in range(n)]
    patterns = {5: [2, 2, 1], 4: [2, 1, 1], 3: [2, 1]}
    blocks = []
    while n > 0:
        for size in (5, 4, 3):
            if n >= size and MAXLEN_BLOCK_OK.get(size, True):
                blocks.append(patterns[size])
                n -= size
                break
        else:
            blocks.append([1])
            n -= 1
    return blocks


def get_max_length_pattern(n):
    return [s for block in get_max_length_blocks(n) for s in block]


def _pair_size(length):
    """Strings planned per MPPT for a string length."""
    if DENSE_MODE:
        return max(1, S_BY_LENGTH.get(length, 1))
    if MAX_STRINGS_PER_MPPT < 2:
        return 1
    if PAIR_OK_LENGTHS is not None and length not in PAIR_OK_LENGTHS:
        return 1
    return 2


def set_pattern_limits(inverters_df, settings, check_ratio):
    """Tell Stage 1 and the grouping how many strings per MPPT the limits allow."""
    global PAIR_OK_LENGTHS, MAXLEN_BLOCK_OK, DENSE_MODE, S_BY_LENGTH
    cap_n = float(settings["max_mppt_dc_ac"])
    cap_b = float(settings.get("max_mppt_dc_ac_maxlen", 1.33))
    best_kw = float((inverters_df["AC_kW"] / inverters_df["MPPTs_per_Inverter"]).max())  # most generous MPPT
    if "Strings_per_MPPT" in inverters_df.columns:
        s_models = int(inverters_df["Strings_per_MPPT"].max())
    else:
        s_models = 2
    s_plan = max(1, min(s_models, int(settings.get("max_strings_per_mppt", 3))))

    DENSE_MODE = s_plan >= 3
    S_BY_LENGTH = {}
    if DENSE_MODE:
        for L in ALLOWED_LENGTHS:
            if check_ratio:
                fit = int((cap_n * best_kw + 1e-9) // (L * MODULE_KWP))
                S_BY_LENGTH[L] = max(1, min(s_plan, fit))
            else:
                S_BY_LENGTH[L] = s_plan
        PAIR_OK_LENGTHS = None
        MAXLEN_BLOCK_OK = {3: True, 4: True, 5: True}
        return

    if not check_ratio:
        PAIR_OK_LENGTHS = None
        MAXLEN_BLOCK_OK = {3: True, 4: True, 5: True}
        return
    max_len = max(ALLOWED_LENGTHS)
    PAIR_OK_LENGTHS = {L for L in ALLOWED_LENGTHS
                       if L != max_len and 2 * L * MODULE_KWP <= cap_n * best_kw + 1e-9}
    k = max_len * MODULE_KWP
    MAXLEN_BLOCK_OK = {5: 5 * k / (3 * best_kw) <= cap_b + 1e-9,
                       4: 4 * k / (3 * best_kw) <= cap_b + 1e-9,
                       3: 3 * k / (2 * best_kw) <= cap_b + 1e-9}


def capacity_under_mppt_limit(inverters_df, settings):
    """Upper bound of DC (kWp) the fleet can host if every MPPT must respect the per-MPPT limits."""
    cap_n = float(settings["max_mppt_dc_ac"])
    cap_b = float(settings.get("max_mppt_dc_ac_maxlen", 1.33))
    max_len = max(ALLOWED_LENGTHS)
    total = 0.0
    for _, r in inverters_df.iterrows():
        kw = float(r["AC_kW"]) / int(r["MPPTs_per_Inverter"])
        opts = []
        for L in sorted(ALLOWED_LENGTHS):
            single = L * MODULE_KWP
            lim = cap_b if L == max_len else cap_n
            if single <= lim * kw + 1e-9:
                opts.append(single)
            if L != max_len and MAX_STRINGS_PER_MPPT >= 2 and 2 * single <= cap_n * kw + 1e-9:
                opts.append(2 * single)
        if MAX_STRINGS_PER_MPPT >= 2 and ALLOW_MAXLEN_PAIRS:
            for n_str, n_mppt in ((5, 3), (4, 3), (3, 2)):
                avg = n_str * max_len * MODULE_KWP / n_mppt
                if avg <= cap_b * kw + 1e-9:
                    opts.append(avg)
        best = max(opts) if opts else 0.0
        total += int(r["Quantity"]) * min(float(r["AC_kW"]) * float(settings["max_inverter_dc_ac"]),
                                          int(r["MPPTs_per_Inverter"]) * best)
    return total


def mppts_for_length(k, length):
    """MPPTs needed for k strings of one length in one pool. Used by Stage 1 AND grouping."""
    if k <= 0:
        return 0
    if length == max(ALLOWED_LENGTHS) and not DENSE_MODE:
        return len(get_max_length_pattern(k))
    return -(-k // _pair_size(length))


def build_mppt_groups(strings_df):
    required = {"String_ID", "Orientation", "Modules_in_String"}
    missing = required - set(strings_df.columns)
    if missing:
        raise ValueError(f"strings_df is missing required columns: {sorted(missing)}")

    strings = strings_df.copy()
    strings["Orientation"] = strings["Orientation"].astype(str)
    strings["Modules_in_String"] = pd.to_numeric(
        strings["Modules_in_String"], errors="raise"
    ).astype(int)

    if strings["String_ID"].duplicated().any():
        raise ValueError("String_ID values must be unique.")
    if not strings["Modules_in_String"].isin(ALLOWED_LENGTHS).all():
        raise ValueError(f"Only string lengths {sorted(ALLOWED_LENGTHS)} are allowed.")

    max_len = max(ALLOWED_LENGTHS)

    strings["_Pool"] = [
        pool_of(o, n) for o, n in zip(strings["Orientation"], strings["Modules_in_String"])
    ]

    groups, counter, block_counter = [], 0, 0
    for (pool, length), subset in strings.groupby(["_Pool", "Modules_in_String"], sort=True):
        records = subset.sort_values(["Orientation", "String_ID"]).to_dict("records")

        # Mix orientations on an MPPT ONLY when it saves an MPPT; otherwise keep each orientation apart.
        by_orient = {}
        for r in records:
            by_orient.setdefault(r["Orientation"], []).append(r)
        if pool == "SHARED_POOL" and len(by_orient) > 1:
            separate = sum(mppts_for_length(len(v), length) for v in by_orient.values())
            pooled = mppts_for_length(len(records), length)
            parts = [by_orient[o] for o in sorted(by_orient)] if separate == pooled else [records]
        else:
            parts = [records]

        for part in parts:
            if length == max_len and not DENSE_MODE:
                blocks = get_max_length_blocks(len(part))
            else:
                pair_size = _pair_size(length)
                sizes = [pair_size] * (len(part) // pair_size)
                if len(part) % pair_size:
                    sizes.append(len(part) % pair_size)
                blocks = [[s] for s in sizes]

            cursor = 0
            for block in blocks:
                block_counter += 1
                block_id = f"B{block_counter:03d}"
                for size in block:
                    selected = part[cursor:cursor + size]
                    cursor += size

                    orientations = sorted({r["Orientation"] for r in selected})
                    if not orientations_can_share(orientations):
                        raise ValueError(f"Invalid orientation mix for an MPPT: {orientations}")
                    if len(selected) > MAX_STRINGS_PER_MPPT:
                        raise ValueError(f"An MPPT group exceeds {MAX_STRINGS_PER_MPPT} strings.")

                    counter += 1
                    groups.append({
                        "MPPT_Group_ID": f"MPPT-G{counter:03d}",
                        "Block_ID": block_id,
                        "String_IDs": [r["String_ID"] for r in selected],
                        "Orientations": orientations,
                        "String_Length": int(length),
                        "String_Count": len(selected),
                        "Panel_Count": sum(int(r["Modules_in_String"]) for r in selected),
                        "DC_kWp": sum(
                            float(r.get("DC_kWp", int(r["Modules_in_String"]) * MODULE_KWP))
                            for r in selected
                        ),
                    })

            if cursor != len(part):
                raise RuntimeError(f"Not all strings were grouped for pool={pool}, length={length}.")

    groups_df = pd.DataFrame(groups)

    grouped_ids = [sid for ids in groups_df["String_IDs"] for sid in ids]
    input_ids = strings_df["String_ID"].tolist()
    if len(grouped_ids) != len(input_ids):
        raise AssertionError("String count changed during MPPT grouping.")
    if set(grouped_ids) != set(input_ids):
        raise AssertionError("Missing or unexpected strings in MPPT groups.")
    if len(grouped_ids) != len(set(grouped_ids)):
        raise AssertionError("A string was assigned to more than one MPPT.")
    return groups_df


# ---------------------------------------------------------------- cell6
def solve_string_counts(orientations, settings, available_mppts, quiet=False):
    lengths = list(range(settings["min_modules_per_string"],
                         settings["max_modules_per_string"] + 1))
    min_len, max_len = lengths[0], lengths[-1]
    panels = {row["Orientation"]: int(row["Panels"])
              for _, row in orientations.iterrows()}

    def build(cap):
        model = cp_model.CpModel()
        x = {(o, n): model.NewIntVar(0, panels[o] // n, f"s_{o}_{n}")
             for o in panels for n in lengths}
        for o, count in panels.items():
            model.Add(sum(n * x[o, n] for n in lengths) == count)
        terms = []
        for n in lengths:
            pools = {}
            for o in panels:
                key = "SHARED" if (o in SHAREABLE_ORIENTATION_POOL and n in SHARE_LENGTHS) else o
                pools.setdefault(key, []).append(o)
            for key, members in pools.items():
                max_k = sum(panels[o] // n for o in members)
                table = [[k, mppts_for_length(k, n)] for k in range(max_k + 1)]
                cnt = model.NewIntVar(0, max_k, f"cnt_{key}_{n}")
                model.Add(cnt == sum(x[o, n] for o in members))
                m = model.NewIntVar(0, max(t[1] for t in table), f"m_{key}_{n}")
                model.AddAllowedAssignments([cnt, m], table)
                terms.append(m)
        if cap is not None:
            model.Add(sum(terms) <= cap)
        return model, x, terms

    def run(model):
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = settings["solver_time_seconds"]
        solver.parameters.num_search_workers = settings["solver_workers"]
        return solver, solver.Solve(model)

    model, x, terms = build(available_mppts)
    total_max = sum(x[o, max_len] for o in panels)
    total_min = sum(x[o, min_len] for o in panels)
    model.Maximize(1_000_000 * total_max - 1_000 * total_min - sum(terms))
    solver, status = run(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        m2, _, t2 = build(None)
        m2.Minimize(sum(t2))
        s2, st2 = run(m2)
        if st2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            need = int(s2.ObjectiveValue())
            raise ValueError(f"The project needs at least {need} MPPTs with the current rules, "
                             f"but only {available_mppts} are available.")
        raise ValueError("No exact string-count solution. Check panel counts and string lengths.")
    rows = []
    for o in panels:
        for n in lengths:
            qty = solver.Value(x[o, n])
            if qty:
                rows.append({"Orientation": o, "Modules_per_String": n, "Strings": qty,
                             "Modules": qty * n, "DC_kWp": qty * n * settings["module_power_wp"] / 1000})
    return pd.DataFrame(rows)

def create_individual_strings(string_counts, settings, quiet=False):
    module_kwp = settings["module_power_wp"] / 1000
    strings, k = [], 1
    for _, row in string_counts.iterrows():
        for _ in range(int(row["Strings"])):
            n = int(row["Modules_per_String"])
            strings.append({"String_ID": f"S{k:03d}", "Orientation": row["Orientation"],
                            "Modules_in_String": n, "Modules": n, "DC_kWp": n * module_kwp})
            k += 1
    return pd.DataFrame(strings)


# ---------------------------------------------------------------- cell9

def allocate_mppt_groups(groups_df, inverters_df, settings, time_limit_seconds=120):
    required_inv = {"Model", "AC_kW", "MPPTs_per_Inverter", "Quantity"}
    missing_inv = required_inv - set(inverters_df.columns)
    if missing_inv:
        raise ValueError(f"inverters_df is missing columns: {sorted(missing_inv)}")

    required_groups = {"MPPT_Group_ID", "Block_ID", "String_IDs", "Orientations", "String_Length",
                       "String_Count", "Panel_Count", "DC_kWp"}
    missing_groups = required_groups - set(groups_df.columns)
    if missing_groups:
        raise ValueError(f"groups_df is missing columns: {sorted(missing_groups)}")
    if groups_df.empty:
        raise ValueError("No MPPT groups were generated.")

    max_ratio = float(settings["MAX_INVERTER_DC_AC"])
    cap_normal = float(settings["max_mppt_dc_ac"])                    # 11/12-module strings, per MPPT
    cap_block = float(settings.get("max_mppt_dc_ac_maxlen", 1.33))    # longest strings, per block average
    check_mppt = bool(settings.get("check_mppt_ratio", True))
    max_len = max(ALLOWED_LENGTHS)

    # Expand quantities into physical inverters
    rows = []
    for _, row in inverters_df.iterrows():
        ac_kw = float(row["AC_kW"]); n_mppt = int(row["MPPTs_per_Inverter"]); qty = int(row["Quantity"])
        for unit in range(1, qty + 1):
            rows.append({
                "Inverter_ID": f"{row['Model']}-{unit:02d}",
                "Model": str(row["Model"]),
                "AC_kW": ac_kw,
                "MPPT_Capacity": n_mppt,
                "MPPT_kW": ac_kw / n_mppt,
                "Max_Strings": int(row["Strings_per_MPPT"]) if "Strings_per_MPPT" in inverters_df.columns
                               else int(settings.get("max_strings_per_mppt", 3)),
                "Max_DC_kWp": ac_kw * max_ratio,
            })
    inv_df = pd.DataFrame(rows).reset_index(drop=True)
    grp_df = groups_df.reset_index(drop=True).copy()
    if grp_df["MPPT_Group_ID"].duplicated().any():
        raise ValueError("MPPT_Group_ID values must be unique.")

    I = len(inv_df)
    SCALE = 1000
    group_dc = [int(round(float(v) * SCALE)) for v in grp_df["DC_kWp"]]
    inv_max_dc = [int(round(float(v) * SCALE)) for v in inv_df["Max_DC_kWp"]]

    # The unit placed on an inverter is a BLOCK (e.g. 3 strings of 13 on 2 MPPTs), never a lone pair.
    block_ids = list(dict.fromkeys(grp_df["Block_ID"]))
    members = {b: [int(g) for g in grp_df.index[grp_df["Block_ID"] == b]] for b in block_ids}
    B = len(block_ids)
    b_slots = [len(members[b]) for b in block_ids]
    b_dc = [sum(group_dc[g] for g in members[b]) for b in block_ids]
    b_len = [int(grp_df.loc[members[b][0], "String_Length"]) for b in block_ids]

    allowed, bad = {}, []
    for k in range(B):
        ok = []
        for i in range(I):
            fits = (b_slots[k] <= int(inv_df.loc[i, "MPPT_Capacity"])
                    and b_dc[k] <= inv_max_dc[i]
                    and max(int(grp_df.loc[g, "String_Count"]) for g in members[block_ids[k]])
                        <= int(inv_df.loc[i, "Max_Strings"]))
            if fits and check_mppt:
                mppt_kw = float(inv_df.loc[i, "MPPT_kW"])
                if b_len[k] == max_len and not DENSE_MODE:
                    fits = (b_dc[k] / SCALE) / b_slots[k] <= cap_block * mppt_kw + 1e-9
                else:
                    fits = all(group_dc[g] / SCALE <= cap_normal * mppt_kw + 1e-9
                               for g in members[block_ids[k]])
            if fits:
                ok.append(i)
        allowed[k] = ok
        if not ok:
            bad.append(f"{block_ids[k]} ({b_slots[k]} MPPT, {b_dc[k] / SCALE:.2f} kWp, "
                       f"{b_len[k]}-module strings)")
    if bad:
        raise ValueError("These MPPT blocks fit on NO inverter (MPPT count, DC limit or per-MPPT "
                         "ratio): " + "; ".join(bad[:6]) + (" ..." if len(bad) > 6 else ""))

    model = cp_model.CpModel()
    x = {(k, i): model.NewBoolVar(f"x_b{k}_i{i}") for k in range(B) for i in allowed[k]}
    for k in range(B):
        model.Add(sum(x[k, i] for i in allowed[k]) == 1)

    loads = []
    for i in range(I):
        ks = [k for k in range(B) if (k, i) in x]
        model.Add(sum(b_slots[k] * x[k, i] for k in ks) <= int(inv_df.loc[i, "MPPT_Capacity"]))
        load_i = sum(b_dc[k] * x[k, i] for k in ks)
        model.Add(load_i <= inv_max_dc[i])
        loads.append(load_i)

    # Minimize the highest inverter DC/AC ratio (permille)
    t = model.NewIntVar(0, 5000, "max_ratio_permille")
    for i in range(I):
        ac_x100 = int(round(float(inv_df.loc[i, "AC_kW"]) * 100))
        model.Add(loads[i] * 100 <= t * ac_x100)
    model.Minimize(t)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_search_workers = int(settings.get("solver_workers", 8))
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(
            "No feasible inverter allocation. Check MPPT capacity per inverter, "
            "the inverter DC/AC limit, and the per-MPPT ratio limits."
        )

    block_inv = {}
    for k in range(B):
        block_inv[block_ids[k]] = next(i for i in allowed[k] if solver.Value(x[k, i]) == 1)

    assignments = []
    for g in range(len(grp_df)):
        i = block_inv[grp_df.loc[g, "Block_ID"]]
        group = grp_df.loc[g].to_dict()
        inv = inv_df.loc[i].to_dict()
        assignments.append({
            **group,
            "Inverter_ID": inv["Inverter_ID"],
            "Inverter_Model": inv["Model"],
            "Inverter_AC_kW": inv["AC_kW"],
            "Inverter_MPPT_Count": int(inv["MPPT_Capacity"]),
            "Inverter_Max_DC_kWp": inv["Max_DC_kWp"],
        })
    allocation_df = pd.DataFrame(assignments)

    summary_rows = []
    for _, inv in inv_df.iterrows():
        assigned = allocation_df[allocation_df["Inverter_ID"] == inv["Inverter_ID"]]
        dc_kwp = float(assigned["DC_kWp"].sum())
        ac_kw = float(inv["AC_kW"])
        summary_rows.append({
            "Inverter_ID": inv["Inverter_ID"],
            "Model": inv["Model"],
            "AC_kW": ac_kw,
            "MPPTs_Available": int(inv["MPPT_Capacity"]),
            "MPPTs_Used": int(len(assigned)),
            "DC_kWp": dc_kwp,
            "DC_AC_Ratio": dc_kwp / ac_kw,
            "Max_DC_kWp": float(inv["Max_DC_kWp"]),
        })
    return allocation_df, pd.DataFrame(summary_rows), solver.StatusName(status)


def use_spare_mppts(strings_df, groups_df, allocation_df, inverter_summary_df, settings):
    """
    Use MPPT inputs that would stay empty: split a 2-string MPPT that is above the per-MPPT limit
    into two single-string MPPTs on the SAME inverter (inverter DC/AC does not change).
    The worst MPPTs are split first. Returns the updated tables and the number of splits.
    """
    cap_normal = float(settings["max_mppt_dc_ac"])
    cap_block = float(settings.get("max_mppt_dc_ac_maxlen", 1.33))
    max_len = max(ALLOWED_LENGTHS)

    alloc = allocation_df.copy().reset_index(drop=True)
    inv = inverter_summary_df.set_index("Inverter_ID")
    kw = (inv["AC_kW"] / inv["MPPTs_Available"]).to_dict()
    used = alloc.groupby("Inverter_ID").size().to_dict()
    spare = {i: int(inv.loc[i, "MPPTs_Available"]) - int(used.get(i, 0)) for i in inv.index}
    s_lookup = strings_df.set_index("String_ID")

    splits = 0
    while True:
        blk = alloc.groupby("Block_ID").agg(dc=("DC_kWp", "sum"), n=("MPPT_Group_ID", "count"))
        blk_dc, blk_n = blk["dc"].to_dict(), blk["n"].to_dict()
        best = None
        for idx, row in alloc.iterrows():
            if int(row["String_Count"]) < 2 or spare[row["Inverter_ID"]] <= 0:
                continue
            mkw = kw[row["Inverter_ID"]]
            if int(row["String_Length"]) == max_len and not DENSE_MODE:   # longest strings: block average
                over = blk_dc[row["Block_ID"]] / (blk_n[row["Block_ID"]] * mkw) - cap_block
            else:
                over = row["DC_kWp"] / mkw - cap_normal
            if over > 1e-9 and (best is None or over > best[0]):
                best = (over, idx)
        if best is None:
            break

        row = alloc.loc[best[1]]
        ids = list(row["String_IDs"])
        parts = []
        for suffix, chunk in zip("ab", (ids[:-1], ids[-1:])):      # peel ONE string onto the spare MPPT
            sub = s_lookup.loc[chunk]
            new = row.to_dict()
            new.update({
                "MPPT_Group_ID": f"{row['MPPT_Group_ID']}{suffix}",
                "String_IDs": chunk,
                "Orientations": sorted(set(sub["Orientation"])),
                "String_Count": len(chunk),
                "Panel_Count": int(sub["Modules_in_String"].sum()),
                "DC_kWp": float(sub["DC_kWp"].sum()),
            })
            parts.append(new)
        alloc = pd.concat([alloc.drop(index=best[1]), pd.DataFrame(parts)], ignore_index=True)
        spare[row["Inverter_ID"]] -= 1
        splits += 1

    new_groups = alloc[list(groups_df.columns)].sort_values("MPPT_Group_ID").reset_index(drop=True)
    new_summary = inverter_summary_df.copy()
    counts = alloc.groupby("Inverter_ID").size()
    new_summary["MPPTs_Used"] = new_summary["Inverter_ID"].map(counts).fillna(0).astype(int)
    return new_groups, alloc, new_summary, splits


def allocate_joint(strings_df, inverters_df, settings, time_limit_seconds=60):
    """
    DENSE MODE (some inverter takes 3+ strings per MPPT).
    Grouping and inverter allocation are solved TOGETHER: for every inverter the solver chooses how
    many of its MPPTs carry 1, 2 or 3 strings of each string type, so that
      - every string is placed, inverter DC/AC and MPPT counts are respected,
      - no MPPT carries more strings than its inverter model allows,
      - the load is spread over ALL MPPT inputs (lowest possible DC/AC per MPPT).
    """
    check_mppt = bool(settings.get("check_mppt_ratio", True))
    cap_normal = float(settings["max_mppt_dc_ac"])
    max_ratio = float(settings["MAX_INVERTER_DC_AC"])
    s_glob = int(settings.get("max_strings_per_mppt", 3))

    # --- string types: same orientation pool + same number of modules in series -------------
    strings = strings_df.copy()
    strings["_Type"] = [(pool_of(o, int(n)), int(n))
                        for o, n in zip(strings["Orientation"], strings["Modules_in_String"])]
    types = {}
    for key, sub in strings.groupby("_Type", sort=True):
        types[key] = sub.sort_values(["Orientation", "String_ID"]).to_dict("records")
    tkeys = list(types)

    # --- physical inverters ------------------------------------------------------------------
    rows = []
    for _, row in inverters_df.iterrows():
        ac_kw = float(row["AC_kW"]); n_mppt = int(row["MPPTs_per_Inverter"])
        s_model = int(row["Strings_per_MPPT"]) if "Strings_per_MPPT" in inverters_df.columns else 2
        for unit in range(1, int(row["Quantity"]) + 1):
            rows.append({"Inverter_ID": f"{row['Model']}-{unit:02d}", "Model": str(row["Model"]),
                         "AC_kW": ac_kw, "MPPT_Capacity": n_mppt, "MPPT_kW": ac_kw / n_mppt,
                         "Max_Strings": max(1, min(s_model, s_glob)), "Max_DC_kWp": ac_kw * max_ratio})
    inv_df = pd.DataFrame(rows).reset_index(drop=True)
    I = len(inv_df)
    SCALE = 1000

    model = cp_model.CpModel()
    n_var, cost_terms = {}, []
    for i in range(I):
        kw = float(inv_df.loc[i, "MPPT_kW"])
        for s in range(1, int(inv_df.loc[i, "Max_Strings"]) + 1):
            for t in tkeys:
                length = t[1]
                dc_group = s * length * MODULE_KWP
                if check_mppt and dc_group > cap_normal * kw + 1e-9:
                    continue
                ub = min(int(inv_df.loc[i, "MPPT_Capacity"]), len(types[t]) // s)
                if ub <= 0:
                    continue
                v = model.NewIntVar(0, ub, f"n_{i}_{s}_{t[0]}_{length}")
                n_var[i, s, t] = v
                ratio_pm = int(round(1000 * dc_group / kw))
                cost_terms.append(v * (ratio_pm * ratio_pm // 1000))

    for t in tkeys:
        model.Add(sum(s * v for (i, s, tt), v in n_var.items() if tt == t) == len(types[t]))

    # PVsyst view: one sub-array = one inverter + one string length (all orientations together).
    # Its DC/AC per MPPT is the AVERAGE over the MPPTs it uses. Overload is penalised (soft limit),
    # and every extra sub-array (inverter + length) costs a little, so lengths are consolidated.
    lengths_used = sorted({t[1] for t in tkeys})
    for i in range(I):
        kw = float(inv_df.loc[i, "MPPT_kW"])
        cap_w = int(round(cap_normal * kw * SCALE))
        for L in lengths_used:
            mine = [(s, v) for (ii, s, t), v in n_var.items() if ii == i and t[1] == L]
            if not mine:
                continue
            count = sum(v for _, v in mine)
            dc_w = sum(int(round(s * L * MODULE_KWP * SCALE)) * v for s, v in mine)
            excess = model.NewIntVar(0, 10**7, f"excess_{i}_{L}")
            model.Add(excess >= dc_w - cap_w * count)
            cost_terms.append(60 * excess)
            used = model.NewBoolVar(f"used_{i}_{L}")
            model.Add(count <= int(inv_df.loc[i, "MPPT_Capacity"]) * used)
            cost_terms.append(3000 * used)

    loads = []
    for i in range(I):
        mine = [(s, t, v) for (ii, s, t), v in n_var.items() if ii == i]
        model.Add(sum(v for _, _, v in mine) <= int(inv_df.loc[i, "MPPT_Capacity"]))
        load = sum(int(round(s * t[1] * MODULE_KWP * SCALE)) * v for s, t, v in mine)
        model.Add(load <= int(round(float(inv_df.loc[i, "Max_DC_kWp"]) * SCALE)))
        loads.append(load)

    t_max = model.NewIntVar(0, 5000, "max_ratio_permille")
    for i in range(I):
        model.Add(loads[i] * 100 <= t_max * int(round(float(inv_df.loc[i, "AC_kW"]) * 100)))
    model.Minimize(sum(cost_terms) + t_max)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_search_workers = int(settings.get("solver_workers", 8))
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible grouping + inverter allocation (MPPT count, strings per MPPT, "
                           "inverter DC limit or per-MPPT ratio).")

    # --- build groups / allocation from the solution --------------------------------------------
    pointer = {t: 0 for t in tkeys}
    group_rows, alloc_rows, counter = [], [], 0
    for i in range(I):
        for (ii, s, t), v in sorted(n_var.items(), key=lambda kv: (kv[0][0], -kv[0][1], str(kv[0][2]))):
            if ii != i:
                continue
            for _ in range(solver.Value(v)):
                recs = types[t][pointer[t]:pointer[t] + s]
                pointer[t] += s
                counter += 1
                g = {
                    "MPPT_Group_ID": f"MPPT-G{counter:03d}",
                    "Block_ID": f"B{counter:03d}",
                    "String_IDs": [r["String_ID"] for r in recs],
                    "Orientations": sorted({r["Orientation"] for r in recs}),
                    "String_Length": int(t[1]),
                    "String_Count": len(recs),
                    "Panel_Count": sum(int(r["Modules_in_String"]) for r in recs),
                    "DC_kWp": sum(float(r["DC_kWp"]) for r in recs),
                }
                group_rows.append(g)
                alloc_rows.append({**g,
                                   "Inverter_ID": inv_df.loc[i, "Inverter_ID"],
                                   "Inverter_Model": inv_df.loc[i, "Model"],
                                   "Inverter_AC_kW": float(inv_df.loc[i, "AC_kW"]),
                                   "Inverter_MPPT_Count": int(inv_df.loc[i, "MPPT_Capacity"]),
                                   "Inverter_Max_DC_kWp": float(inv_df.loc[i, "Max_DC_kWp"])})
    groups_out = pd.DataFrame(group_rows)
    allocation_out = pd.DataFrame(alloc_rows)

    summary_rows = []
    for _, inv in inv_df.iterrows():
        assigned = allocation_out[allocation_out["Inverter_ID"] == inv["Inverter_ID"]]
        dc_kwp = float(assigned["DC_kWp"].sum())
        summary_rows.append({"Inverter_ID": inv["Inverter_ID"], "Model": inv["Model"],
                             "AC_kW": float(inv["AC_kW"]), "MPPTs_Available": int(inv["MPPT_Capacity"]),
                             "MPPTs_Used": int(len(assigned)), "DC_kWp": dc_kwp,
                             "DC_AC_Ratio": dc_kwp / float(inv["AC_kW"]),
                             "Max_DC_kWp": float(inv["Max_DC_kWp"])})
    return groups_out, allocation_out, pd.DataFrame(summary_rows), solver.StatusName(status)


# ---------------------------------------------------------------- cell10
def validate_v1(strings_df, groups_df, allocation_df, inverter_summary_df, expected_modules, max_dc_ac_ratio):
    errors = []
    required_strings = {"String_ID", "Orientation", "Modules_in_String"}
    if not required_strings.issubset(strings_df.columns):
        raise ValueError(f"strings_df must contain {sorted(required_strings)}")
    if strings_df["String_ID"].duplicated().any():
        errors.append("Input string IDs are not unique.")
    actual_modules = int(strings_df["Modules_in_String"].sum())
    if actual_modules != expected_modules:
        errors.append(f"Expected {expected_modules} modules, but string formation produced {actual_modules}.")
    invalid_lengths = sorted(set(strings_df["Modules_in_String"].astype(int)) - ALLOWED_LENGTHS)
    if invalid_lengths:
        errors.append(f"Invalid string lengths found: {invalid_lengths}.")
    expected_ids = strings_df["String_ID"].tolist()
    grouped_ids = [sid for ids in groups_df["String_IDs"] for sid in ids]
    if len(grouped_ids) != len(expected_ids):
        errors.append("String count changed during MPPT grouping.")
    if set(grouped_ids) != set(expected_ids):
        errors.append("MPPT groups do not contain exactly the input strings.")
    if len(grouped_ids) != len(set(grouped_ids)):
        errors.append("At least one string is assigned to multiple MPPT groups.")
    if allocation_df["MPPT_Group_ID"].duplicated().any():
        errors.append("An MPPT group appears more than once in the allocation.")
    if set(allocation_df["MPPT_Group_ID"]) != set(groups_df["MPPT_Group_ID"]):
        errors.append("Some MPPT groups are missing or unexpected in the allocation.")
    string_lookup = strings_df.set_index("String_ID").to_dict("index")
    for _, row in allocation_df.iterrows():
        group_id = row["MPPT_Group_ID"]; ids = row["String_IDs"]
        if not ids:
            errors.append(f"{group_id} contains no strings."); continue
        if len(ids) > MAX_STRINGS_PER_MPPT:
            errors.append(f"{group_id} exceeds the {MAX_STRINGS_PER_MPPT}-string MPPT limit.")
        try:
            lengths = {int(string_lookup[sid]["Modules_in_String"]) for sid in ids}
            orientations = {str(string_lookup[sid]["Orientation"]) for sid in ids}
        except KeyError as exc:
            errors.append(f"{group_id} references an unknown string: {exc}."); continue
        if len(lengths) != 1:
            errors.append(f"{group_id} contains different string lengths.")
        if not orientations_can_share(orientations):
            errors.append(f"{group_id} has a prohibited orientation combination: {sorted(orientations)}.")
        expected_group_dc = sum(int(string_lookup[sid]["Modules_in_String"]) * MODULE_KWP for sid in ids)
        if abs(float(row["DC_kWp"]) - expected_group_dc) > 0.002:
            errors.append(f"{group_id} has an inconsistent DC power value.")
    for _, row in inverter_summary_df.iterrows():
        if row["MPPTs_Used"] > row["MPPTs_Available"]:
            errors.append(f"{row['Inverter_ID']} exceeds its available MPPT count.")
        if row["DC_AC_Ratio"] > max_dc_ac_ratio + 1e-9:
            errors.append(f"{row['Inverter_ID']} DC/AC ratio {row['DC_AC_Ratio']:.4f} exceeds {max_dc_ac_ratio:.2f}.")
        if row["DC_kWp"] > row["Max_DC_kWp"] + 0.002:
            errors.append(f"{row['Inverter_ID']} exceeds its DC capacity limit.")
    input_dc_kwp = actual_modules * MODULE_KWP
    allocated_dc_kwp = float(allocation_df["DC_kWp"].sum())
    if abs(input_dc_kwp - allocated_dc_kwp) > 0.002:
        errors.append(f"DC power mismatch: input={input_dc_kwp:.3f} kWp, allocated={allocated_dc_kwp:.3f} kWp.")
    return {"passed": len(errors) == 0, "errors": errors, "input_modules": actual_modules,
            "expected_modules": expected_modules, "input_dc_kWp": input_dc_kwp, "allocated_dc_kWp": allocated_dc_kwp}


# ---------------------------------------------------------------- cell11
def _numbered(allocation):
    """Sorted allocation with MPPT_No per inverter (blocks stay on adjacent MPPTs)."""
    alloc = allocation.sort_values(["Inverter_ID", "Block_ID", "MPPT_Group_ID"]).reset_index(drop=True)
    alloc["MPPT_No"] = alloc.groupby("Inverter_ID").cumcount() + 1
    return alloc


def make_detailed_report(allocation, show=True):
    report = _numbered(allocation)

    report["MPPT_kW"] = report["Inverter_AC_kW"] / report["Inverter_MPPT_Count"]
    report["MPPT_DC_AC"] = (report["DC_kWp"] / report["MPPT_kW"]).round(3)

    blk = report.groupby("Block_ID").agg(
        B_Strings=("String_Count", "sum"), B_MPPTs=("MPPT_Group_ID", "count"),
        B_DC=("DC_kWp", "sum"), B_kW=("MPPT_kW", "first"), B_Len=("String_Length", "first"))
    blk["Block_DC_AC"] = (blk["B_DC"] / (blk["B_MPPTs"] * blk["B_kW"])).round(3)
    blk["Block_Config"] = blk.apply(
        lambda b: f"{int(b['B_Strings'])}×{int(b['B_Len'])} on {int(b['B_MPPTs'])} MPPT", axis=1)
    report = report.join(blk[["Block_DC_AC", "Block_Config"]], on="Block_ID")

    report["Orientation"] = report["Orientations"].apply(lambda x: " + ".join(x))
    report["Strings_Display"] = report["String_IDs"].apply(lambda x: " + ".join(x))
    report["Modules_Display"] = report.apply(
        lambda r: f'{r["String_Count"]} × {r["String_Length"]}', axis=1
    )
    report["Grouping_Rule"] = report["Orientations"].apply(
        lambda x: "Shared orientations" if len(x) > 1 else "Single orientation"
    )

    detailed = report.rename(columns={
        "Inverter_ID": "Inverter",
        "Inverter_Model": "Model",
        "MPPT_Group_ID": "MPPT_Group",
        "Panel_Count": "Modules",
        "DC_kWp": "MPPT_DC_kWp",
    })[[
        "Inverter", "Model", "MPPT_No", "MPPT_Group", "Block_ID", "Block_Config",
        "Orientation", "Modules_Display", "Strings_Display", "Modules",
        "MPPT_DC_kWp", "MPPT_DC_AC", "Block_DC_AC", "Grouping_Rule",
    ]].reset_index(drop=True)

    if show:
        display(detailed)
    return detailed


# ---------------------------------------------------------------- cell12
def _mppt_rows(strings, allocation, orientations):
    """One row per physical MPPT. A block that contains a mixed MPPT is reported as one unit."""
    lookup = strings.set_index("String_ID")["Orientation"].to_dict()
    names = list(orientations["Orientation"])
    alloc = _numbered(allocation)

    rows = []
    for _, r in alloc.iterrows():
        per = [lookup[s] for s in r["String_IDs"]]
        present = [o for o in names if o in set(per)]
        rec = {
            "Inverter": r["Inverter_ID"],
            "Model": r["Inverter_Model"],
            "AC_kW": float(r["Inverter_AC_kW"]),
            "MPPT_kW": float(r["Inverter_AC_kW"]) / int(r["Inverter_MPPT_Count"]),
            "MPPT_No": int(r["MPPT_No"]),
            "MPPT_Group": r["MPPT_Group_ID"],
            "Block_ID": r["Block_ID"],
            "Present": present,
            "Modules_per_String": int(r["String_Length"]),
            "Strings": len(per),
            "Modules": int(r["Panel_Count"]),
            "DC_kWp": float(r["DC_kWp"]),
        }
        for o in names:
            rec[f"{o}_Strings"] = per.count(o)
        rows.append(rec)
    df = pd.DataFrame(rows)
    inv_tot = df.groupby("Inverter")["DC_kWp"].sum()
    df["Inv_DC_AC"] = df["Inverter"].map(inv_tot) / df["AC_kW"]

    return df, names


def _tilt_az_text(members, orientations):
    t = orientations.set_index("Orientation")
    return " ; ".join(f"{int(t.loc[o, 'Tilt'])}° / {int(t.loc[o, 'Azimuth'])}°" for o in members)


def _order_key(label, names):
    return names.index(label) if label in names else len(names)


def create_pvsyst_inputs(strings, allocation, orientations, show=True, limit=1.33):
    """Project-wide PVsyst sub-arrays, per inverter model. MPPT counts add up to the real total."""
    mppt, names = _mppt_rows(strings, allocation, orientations)

    agg = {
        "Inverters": ("Inverter", "nunique"),
        "Number_of_MPPTs": ("MPPT_Group", "nunique"),
        "Number_of_Strings": ("Strings", "sum"),
        "Total_Modules": ("Modules", "sum"),
        "DC_kWp": ("DC_kWp", "sum"),
        "MPPT_kW": ("MPPT_kW", "first"),
        "AC_kW": ("AC_kW", "first"),
        "Inverter_DC_AC": ("Inv_DC_AC", "max"),
    }
    for o in names:
        agg[f"{o}_Strings"] = (f"{o}_Strings", "sum")

    summary = mppt.groupby(["Model", "Modules_per_String"], as_index=False).agg(**agg)

    # One PVsyst sub-array per inverter model + string length; orientation(s) from the string counts.
    summary["Members"] = summary.apply(lambda r: [o for o in names if r[f"{o}_Strings"] > 0], axis=1)
    summary["Category"] = summary["Members"].apply(lambda m: m[0] if len(m) == 1 else "Mixed " + " + ".join(m))
    summary["Tilt_Azimuth"] = summary["Members"].apply(lambda m: _tilt_az_text(m, orientations))
    summary["Strings_per_MPPT"] = (summary["Number_of_Strings"] / summary["Number_of_MPPTs"]).round(2)
    summary["DC_AC_per_MPPT"] = (summary["DC_kWp"] / (summary["Number_of_MPPTs"] * summary["MPPT_kW"])).round(3)
    summary["Inverter_DC_AC"] = summary["Inverter_DC_AC"].round(3)
    summary["Check"] = summary["DC_AC_per_MPPT"].apply(
        lambda v: "OK" if v <= limit + 1e-9 else f"ABOVE {limit:.2f}")

    summary["_m"] = -summary["AC_kW"]
    summary = (summary.sort_values(["_m", "Model", "Modules_per_String"])
               .drop(columns=["_m", "Members"]).reset_index(drop=True))
    summary.insert(0, "PVsyst_Subarray", [f"Sub-array {i:02d}" for i in range(1, len(summary) + 1)])
    summary["DC_kWp"] = summary["DC_kWp"].round(3)

    cols = (["PVsyst_Subarray", "Model", "Inverters", "Category", "Tilt_Azimuth", "Modules_per_String",
             "Number_of_MPPTs", "Number_of_Strings", "Strings_per_MPPT", "Total_Modules", "DC_kWp",
             "DC_AC_per_MPPT", "Check", "Inverter_DC_AC"] + [f"{o}_Strings" for o in names])
    summary = summary[cols]

    physical_mppt_count = int(len(allocation))
    assert int(summary["Number_of_MPPTs"].sum()) == physical_mppt_count, "MPPT total mismatch."
    assert int(summary["Number_of_Strings"].sum()) == len(strings), "String total mismatch."
    assert int(summary["Total_Modules"].sum()) == int(strings["Modules"].sum()), "Module total mismatch."

    if show:
        print("PVsyst sub-array inputs (project-wide, per inverter model):")
        display(summary)
        print("Physical MPPTs used:", int(summary["Number_of_MPPTs"].sum()))
        print("Total strings:", int(summary["Number_of_Strings"].sum()))
        print("Total modules:", int(summary["Total_Modules"].sum()))
        print("Total DC capacity (kWp):", round(summary["DC_kWp"].sum(), 3))
        high = summary[summary["Check"] != "OK"]
        if len(high):
            print(f"NOTE: {len(high)} sub-array(s) above {limit:.2f} DC/AC per MPPT: "
                  + ", ".join(high["PVsyst_Subarray"]))

    return summary, physical_mppt_count


def create_pvsyst_by_inverter(strings, allocation, orientations, show=True, limit=1.33):
    """Same sub-arrays, split per physical inverter, with the MPPT numbers used."""
    mppt, names = _mppt_rows(strings, allocation, orientations)

    agg = {
        "Model": ("Model", "first"),
        "Number_of_MPPTs": ("MPPT_Group", "nunique"),
        "MPPT_Numbers": ("MPPT_No", lambda s: ", ".join(str(x) for x in sorted(s))),
        "Number_of_Strings": ("Strings", "sum"),
        "Total_Modules": ("Modules", "sum"),
        "DC_kWp": ("DC_kWp", "sum"),
        "MPPT_kW": ("MPPT_kW", "first"),
        "Inverter_DC_AC": ("Inv_DC_AC", "first"),
    }
    for o in names:
        agg[f"{o}_Strings"] = (f"{o}_Strings", "sum")

    out = mppt.groupby(["Inverter", "Modules_per_String"], as_index=False).agg(**agg)
    out["Category"] = out.apply(
        lambda r: (lambda m: m[0] if len(m) == 1 else "Mixed " + " + ".join(m))(
            [o for o in names if r[f"{o}_Strings"] > 0]), axis=1)
    out["DC_AC_per_MPPT"] = (out["DC_kWp"] / (out["Number_of_MPPTs"] * out["MPPT_kW"])).round(3)
    out["Check"] = out["DC_AC_per_MPPT"].apply(lambda v: "OK" if v <= limit + 1e-9 else f"ABOVE {limit:.2f}")
    out = out.sort_values(["Inverter", "Modules_per_String"]).drop(columns=["MPPT_kW"]).reset_index(drop=True)
    out["DC_kWp"] = out["DC_kWp"].round(3)
    out["Inverter_DC_AC"] = out["Inverter_DC_AC"].round(3)

    first = ["Inverter", "Model", "Category", "Modules_per_String", "Number_of_MPPTs", "MPPT_Numbers",
             "Number_of_Strings", "Total_Modules", "DC_kWp", "DC_AC_per_MPPT", "Check", "Inverter_DC_AC"]
    out = out[first + [f"{o}_Strings" for o in names]]

    if show:
        print("PVsyst sub-arrays per inverter (MPPT numbers follow the Detailed Report):")
        display(out)
    return out



# ======================================================================================
# INPUT CLEANING
# ======================================================================================
def clean_orientations(df):
    df = df.copy()
    df["Orientation"] = df["Orientation"].astype(str).str.strip()
    df = df[df["Orientation"].notna() & (df["Orientation"] != "") & (df["Orientation"] != "None")]
    for c in ("Tilt", "Azimuth"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["Panels"] = pd.to_numeric(df["Panels"], errors="coerce").fillna(0).astype(int)
    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError("Enter at least one orientation.")
    if df["Orientation"].duplicated().any():
        raise ValueError("Orientation names must be unique.")
    if (df["Panels"] < 0).any():
        raise ValueError("Panel counts cannot be negative.")
    if (df["Panels"] == 0).all():
        raise ValueError("Enter a panel count for at least one orientation.")
    return df


def clean_inverters(df):
    df = df.copy()
    df["Model"] = df["Model"].astype(str).str.strip()
    df = df[df["Model"].notna() & (df["Model"] != "") & (df["Model"] != "None")]
    df["AC_kW"] = pd.to_numeric(df["AC_kW"], errors="coerce").fillna(0.0)
    for c in ("MPPTs_per_Inverter", "Strings_per_MPPT", "Quantity"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError("Enter at least one inverter model.")
    if df["Model"].duplicated().any():
        raise ValueError("Inverter model names must be unique (use Quantity for several units).")
    if (df["AC_kW"] <= 0).any() or (df["MPPTs_per_Inverter"] <= 0).any() or (df["Quantity"] <= 0).any():
        raise ValueError("Inverter AC kW, MPPTs per inverter and Quantity must all be positive.")
    if (df["Strings_per_MPPT"] < 1).any():
        raise ValueError("Strings per MPPT must be at least 1.")
    return df


def check_settings(v):
    if v["module_power_wp"] <= 0:
        raise ValueError("Module power must be greater than zero.")
    if v["min_modules_per_string"] <= 0 or v["max_modules_per_string"] < v["min_modules_per_string"]:
        raise ValueError("Check the minimum / maximum modules per string.")
    if v["max_inverter_dc_ac"] <= 0 or v["max_mppt_dc_ac"] <= 0:
        raise ValueError("DC/AC ratios must be positive.")
    if v["max_strings_per_mppt"] < 1:
        raise ValueError("Max strings per MPPT must be at least 1.")
    v = dict(v)
    v["MAX_INVERTER_DC_AC"] = v["max_inverter_dc_ac"]
    return v


# ======================================================================================
# THE RUN (same logic as notebook Cell 13, returning data instead of printing)
# ======================================================================================
def run_pipeline(settings, orientation_df, inverter_df, share_pool_text, share_lengths_text):
    global SHAREABLE_ORIENTATION_POOL, SHARE_LENGTHS, MAX_STRINGS_PER_MPPT
    global MODULE_KWP, ALLOWED_LENGTHS, ALLOW_MAXLEN_PAIRS

    log = []
    say = log.append
    res = {"ok": False, "log": log, "errors": []}
    try:
        settings = check_settings(settings)
        orientation_df = clean_orientations(orientation_df)
        inverter_df = clean_inverters(inverter_df)
        names = list(orientation_df["Orientation"])

        strict_pool = {s.strip() for s in share_pool_text.split(",") if s.strip()}
        unknown = strict_pool - set(names)
        if unknown:
            raise ValueError(f"Sharing list has unknown orientations: {sorted(unknown)}")
        strict_lengths = {int(s) for s in share_lengths_text.split(",") if s.strip()}

        MAX_STRINGS_PER_MPPT = settings["max_strings_per_mppt"]
        MODULE_KWP = settings["module_power_wp"] / 1000
        ALLOW_MAXLEN_PAIRS = bool(settings.get("allow_maxlen_pairs", True))
        ALLOWED_LENGTHS = set(range(settings["min_modules_per_string"], settings["max_modules_per_string"] + 1))
        expected_modules = int(orientation_df["Panels"].sum())
        check_user = bool(settings.get("check_mppt_ratio", True))

        say(f"Mode: {'per-MPPT ratio checked' if check_user else 'inverter DC/AC only (per-MPPT ratio ignored)'}")
        say("Strings per MPPT allowed: " + ", ".join(
            f"{r['Model']} = {int(r['Strings_per_MPPT'])}" for _, r in inverter_df.iterrows()))

        available_mppts = int((inverter_df["MPPTs_per_Inverter"] * inverter_df["Quantity"]).sum())
        fleet_ac = float((inverter_df["AC_kW"] * inverter_df["Quantity"]).sum())
        total_dc = expected_modules * MODULE_KWP
        say(f"Pre-check: {total_dc:.2f} kWp DC | fleet {fleet_ac:.0f} kW AC "
            f"(limit {fleet_ac * settings['max_inverter_dc_ac']:.2f} kWp) | {available_mppts} MPPT inputs")
        if total_dc > fleet_ac * settings["max_inverter_dc_ac"]:
            raise ValueError("Total DC exceeds fleet AC x max DC/AC. Add inverters or reduce panels.")

        fast = {**settings, "solver_time_seconds": min(int(settings["solver_time_seconds"]), 30)}
        flags = [check_user] + ([False] if check_user else [])
        pools = ([("no orientation mixing", set(), set())] if strict_pool else [])
        pools.append(("sharing as configured", strict_pool, strict_lengths))
        tiers = [(f"{'Per-MPPT ratio checked' if flag else 'Per-MPPT ratio ignored'} | {pname}", pool, lens, flag)
                 for flag in flags for pname, pool, lens in pools]
        tiers.append(("FALLBACK: orientations mixed on MPPTs (same modules in series)",
                      set(names), set(ALLOWED_LENGTHS), False))

        found, reasons = None, []
        for label, pool, lens, flag in tiers:
            SHAREABLE_ORIENTATION_POOL = pool
            SHARE_LENGTHS = lens
            set_pattern_limits(inverter_df, fast, flag)
            alloc_settings = {**fast, "check_mppt_ratio": flag}

            if flag:
                cap_kwp = capacity_under_mppt_limit(inverter_df, fast)
                if total_dc > cap_kwp + 1e-6:
                    reasons.append(f"'{label}' is impossible: with every MPPT within its limit the fleet can host "
                                   f"at most {cap_kwp:.2f} kWp, but the project is {total_dc:.2f} kWp.")
                    continue

            first_reason = None
            n_try = 4 if DENSE_MODE else 12
            for budget in range(available_mppts, max(available_mppts - n_try, 0), -1):
                try:
                    sc = solve_string_counts(orientation_df, fast, budget, quiet=True)
                except ValueError as exc:
                    first_reason = first_reason or str(exc)
                    break
                st = create_individual_strings(sc, fast, quiet=True)
                try:
                    if DENSE_MODE:
                        gr, al, sm, status = allocate_joint(st, inverter_df, alloc_settings, time_limit_seconds=30)
                    else:
                        gr = build_mppt_groups(st)
                        al, sm, status = allocate_mppt_groups(gr, inverter_df, alloc_settings, time_limit_seconds=30)
                except (RuntimeError, ValueError) as exc:
                    first_reason = first_reason or str(exc)
                    continue
                found = (label, flag, budget, sc, st, gr, al, sm, status)
                break
            if found:
                break
            reasons.append(f"'{label}' failed: {first_reason}")

        if not found:
            raise RuntimeError("No feasible allocation, even with the fallbacks. " + " | ".join(reasons)
                               + " Add inverter capacity, reduce panels, or allow shorter strings.")

        (label, flag, budget, string_counts_df, strings_df, groups_df,
         allocation_df, inverter_summary_df, solver_status) = found

        groups_df, allocation_df, inverter_summary_df, n_split = use_spare_mppts(
            strings_df, groups_df, allocation_df, inverter_summary_df, settings)

        for why in reasons:
            say("Why an earlier option was not used: " + why)
        say(f"Result found with: {label}")
        if n_split:
            say(f"Spare MPPT inputs used: {n_split} string(s) moved onto empty MPPTs.")
        if budget < available_mppts:
            say(f"Note: Stage 1 was limited to {budget} of {available_mppts} MPPTs (DC limits).")
        say(f"Strings: {len(strings_df)} | MPPT groups: {len(groups_df)} of {available_mppts}")

        if label.startswith("FALLBACK"):
            is_relaxed = allocation_df.apply(
                lambda r: len(r["Orientations"]) > 1 and not (
                    set(r["Orientations"]).issubset(strict_pool) and r["String_Length"] in strict_lengths), axis=1)
            for _, r in allocation_df[is_relaxed].iterrows():
                say(f"WARNING: {r['MPPT_Group_ID']} on {r['Inverter_ID']} mixes orientations "
                    f"({' + '.join(r['Orientations'])}, {r['String_Count']} x {r['String_Length']}).")

        validation = validate_v1(strings_df, groups_df, allocation_df, inverter_summary_df,
                                 expected_modules=expected_modules, max_dc_ac_ratio=settings["max_inverter_dc_ac"])
        errors = list(validation["errors"])
        if (allocation_df.groupby("Block_ID")["Inverter_ID"].nunique() > 1).any():
            errors.append("A block is split across two inverters.")

        max_len = max(ALLOWED_LENGTHS)
        cap_block = settings.get("max_mppt_dc_ac_maxlen", 1.33)
        inv_idx = inverter_summary_df.set_index("Inverter_ID")
        mppt_kw = inv_idx["AC_kW"] / inv_idx["MPPTs_Available"]
        lim_by_model = inverter_df.set_index("Model")["Strings_per_MPPT"]
        for _, r in allocation_df.iterrows():
            if r["String_Count"] > int(lim_by_model[r["Inverter_Model"]]):
                errors.append(f"{r['MPPT_Group_ID']}: {r['String_Count']} strings on one MPPT, but "
                              f"{r['Inverter_Model']} allows {int(lim_by_model[r['Inverter_Model']])}.")
            if flag and (DENSE_MODE or r["String_Length"] != max_len):
                ratio = r["DC_kWp"] / mppt_kw[r["Inverter_ID"]]
                if ratio > settings["max_mppt_dc_ac"] + 1e-9:
                    errors.append(f"{r['MPPT_Group_ID']}: MPPT DC/AC {ratio:.3f} > {settings['max_mppt_dc_ac']:.2f}")
            if len(r["Orientations"]) > 1 and r["String_Length"] not in SHARE_LENGTHS:
                errors.append(f"{r['MPPT_Group_ID']}: orientations mixed on a {r['String_Length']}-module MPPT.")
        if flag and not DENSE_MODE:
            blk = allocation_df[allocation_df["String_Length"] == max_len].groupby("Block_ID").agg(
                dc=("DC_kWp", "sum"), n=("MPPT_Group_ID", "count"), inv=("Inverter_ID", "first"))
            for b, r in blk.iterrows():
                ratio = r["dc"] / (r["n"] * mppt_kw[r["inv"]])
                if ratio > cap_block + 1e-9:
                    errors.append(f"Block {b}: DC/AC {ratio:.3f} per MPPT > {cap_block:.2f}")

        res.update(errors=errors, settings=settings, orientation_df=orientation_df, inverter_df=inverter_df,
                   string_counts_df=string_counts_df, strings_df=strings_df, groups_df=groups_df,
                   allocation_df=allocation_df, inverter_summary_df=inverter_summary_df,
                   label=label, solver_status=solver_status, validation=validation)
        if errors:
            say("VALIDATION FAILED — do not use these results.")
            return res

        limit = settings["max_mppt_dc_ac"]
        res["detailed_df"] = make_detailed_report(allocation_df, show=False)
        res["pvsyst_df"], _ = create_pvsyst_inputs(strings_df, allocation_df, orientation_df, show=False, limit=limit)
        res["pvsyst_inv_df"] = create_pvsyst_by_inverter(strings_df, allocation_df, orientation_df,
                                                         show=False, limit=limit)
        high = res["pvsyst_df"][res["pvsyst_df"]["Check"] != "OK"]
        if len(high):
            say(f"PVsyst note: {len(high)} sub-array(s) above {limit:.2f} DC/AC per MPPT: "
                + ", ".join(high["PVsyst_Subarray"]))
        else:
            say("PVsyst check: every sub-array is within the per-MPPT limit.")
        say("VALIDATION PASSED")
        res["ok"] = True
        return res

    except Exception as exc:        # shown to the user, never crashes the page
        say(f"RUN FAILED: {type(exc).__name__}: {exc}")
        return res


def build_excel(res):
    """Excel workbook (bytes) with every table of a successful run."""
    mppt_export = res["allocation_df"].copy()
    mppt_export["Orientations"] = mppt_export["Orientations"].apply(" + ".join)
    mppt_export["String_IDs"] = mppt_export["String_IDs"].apply(" + ".join)
    groups_export = res["groups_df"].copy()
    groups_export["Orientations"] = groups_export["Orientations"].apply(" + ".join)
    groups_export["String_IDs"] = groups_export["String_IDs"].apply(" + ".join)
    sheets = {
        "Inverter Summary": res["inverter_summary_df"],
        "MPPT Allocation": mppt_export,
        "Detailed Report": res["detailed_df"],
        "PVsyst Inputs": res["pvsyst_df"],
        "PVsyst by Inverter": res["pvsyst_inv_df"],
        "String Counts": res["string_counts_df"],
        "Individual Strings": res["strings_df"],
        "MPPT Groups": groups_export,
        "Inputs - Orientations": res["orientation_df"],
        "Inputs - Inverters": res["inverter_df"],
        "Inputs - Settings": pd.DataFrame([{"Setting": k, "Value": str(v)} for k, v in res["settings"].items()]),
    }
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                width = max(len(str(c.value or "")) for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(width + 2, 45)
    return buf.getvalue()
