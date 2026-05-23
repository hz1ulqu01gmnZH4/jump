"""Analyse run log: pipeline health, condition checks, hg summary, cost."""
import json, math, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, '/home/ak/projects/jump')


def analyse(log_path: str, label: str = ""):
    records = [json.loads(l) for l in Path(log_path).read_text().splitlines() if l.strip()]
    print(f"\n{'='*60}")
    print(f"ANALYSE{' — '+label if label else ''}: {len(records)} runs")
    print(f"{'='*60}\n")

    # (1) Adapter health
    print("=== (1) ADAPTER HEALTH ===")
    by_model = defaultdict(list)
    for r in records:
        by_model[r['model']].append(r)
    for model, runs in sorted(by_model.items()):
        errors = [r for r in runs if r.get('error')]
        with_acc = [r for r in runs if r.get('accuracy') is not None]
        avg_acc = (sum(r['accuracy'] for r in with_acc) / len(with_acc)) if with_acc else None
        acc_str = f"{avg_acc:.3f}" if avg_acc is not None else "N/A"
        print(f"  {model}: {len(runs)} runs | {len(errors)} errors | avg_acc={acc_str}")
        if errors:
            print(f"    errors: {[r['error'][:60] for r in errors[:3]]}")

    # (2) Log schema completeness
    print("\n=== (2) LOG SCHEMA ===")
    required = ['world_id','pair_id','family','model','condition','seed',
                'accuracy','kappa','cost_usd','prompt_tokens','completion_tokens',
                'G','H','delta','kappa_robust']
    missing_fields = defaultdict(list)
    for r in records:
        for f in required:
            if f not in r:
                missing_fields[f].append(r.get('world_id','?'))
    if missing_fields:
        for f, worlds in missing_fields.items():
            print(f"  MISSING field '{f}' in {len(worlds)} records")
    else:
        print(f"  All {len(required)} required fields present in all {len(records)} records ✓")

    # (3) hg pipeline
    print("\n=== (3) HG PIPELINE ===")
    with_hg = [r for r in records if r.get('G') is not None]
    print(f"  hg computed: {len(with_hg)}/{len(records)}")
    if with_hg:
        avg_G = sum(r['G'] for r in with_hg) / len(with_hg)
        avg_H = sum(r['H'] for r in with_hg) / len(with_hg)
        avg_d = sum(r['delta'] for r in with_hg) / len(with_hg)
        n_robust = sum(1 for r in with_hg if r.get('kappa_robust'))
        print(f"  avg G={avg_G:.3f} H={avg_H:.3f} δ={avg_d:.3f} | kappa_robust: {n_robust}/{len(with_hg)}")

    # (4) Condition discipline
    print("\n=== (4) CONDITION DISCIPLINE ===")
    passive = [r for r in records if r['condition'] == 'passive']
    passive_viol = [r for r in passive if r.get('condition_intervene_called')]
    on_runs = [r for r in records if r['condition'] == 'ON']
    on_intervened = [r for r in on_runs if r.get('n_interventions', 0) > 0]
    print(f"  passive: {len(passive)} runs | violations (intervene called): {len(passive_viol)}")
    print(f"  ON:      {len(on_runs)} runs | with interventions: {len(on_intervened)}")
    if passive_viol:
        print(f"  VIOLATION DETAIL: {[r['world_id']+'/'+r['model'] for r in passive_viol[:3]]}")

    # (5) Cost summary
    print("\n=== (5) COST SUMMARY ===")
    with_cost = [r for r in records if r.get('cost_usd') is not None]
    if with_cost:
        total = sum(r['cost_usd'] for r in with_cost)
        avg = total / len(with_cost)
        print(f"  Runs with cost: {len(with_cost)}/{len(records)}")
        print(f"  Total: ${total:.4f} | avg/run: ${avg:.5f}")
    else:
        print(f"  No cost_usd data (local: cost=0.0 expected)")

    # (6) Δδ preview
    print("\n=== (6) Δδ PREVIEW (ON vs passive) ===")
    worlds_set = set(r['world_id'] for r in records)
    models_set = set(r['model'] for r in records)
    for world in sorted(worlds_set)[:3]:
        for model in sorted(models_set):
            on_r = [r for r in records if r['world_id']==world and r['model']==model
                    and r['condition']=='ON' and r.get('delta') is not None]
            pa_r = [r for r in records if r['world_id']==world and r['model']==model
                    and r['condition']=='passive' and r.get('delta') is not None]
            if on_r and pa_r:
                d_on = on_r[0]['delta']
                d_pa = pa_r[0]['delta']
                dd = d_on - d_pa
                print(f"  {world[:20]} | {model}: δ_ON={d_on:.3f} δ_passive={d_pa:.3f} Δδ={dd:+.3f}")

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}\n")


def _classify_cell(G: float, H: float, delta: float, kappa_robust: bool) -> str:
    """Apply §4 classification rubric."""
    if G < 0.15 and H < 0.15:
        return 'RANDOM'
    if not kappa_robust:
        return 'AMBIGUOUS'
    if delta < 0 and H >= 0.30:
        return 'HOMOGENIZED'
    if delta > 0 and G >= 0.30:
        return 'GROUNDED'
    return 'WEAK'


def _bootstrap_mean_delta(deltas: list, B: int = 2000, seed: int = 0) -> tuple:
    """Paired bootstrap CI on mean Δδ by resampling worlds."""
    import random
    rng = random.Random(seed)
    n = len(deltas)
    boot_means = []
    for _ in range(B):
        sample = [rng.choice(deltas) for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo = int(math.floor(0.025 * B))
    hi = min(int(math.ceil(0.975 * B)) - 1, B - 1)
    return boot_means[lo], boot_means[hi]


def _effect_sizes(deltas: list) -> tuple:
    """Return (d_z, r_rb) for paired signed differences."""
    n = len(deltas)
    if n < 2:
        return float('nan'), float('nan')
    mean_d = sum(deltas) / n
    var_d = sum((x - mean_d) ** 2 for x in deltas) / (n - 1)
    d_z = mean_d / math.sqrt(var_d) if var_d > 0 else float('nan')

    # Matched rank-biserial: r_rb = 1 - 2*W_neg / (n*(n+1)/2)
    # where W_neg = sum of ranks of negative differences
    abs_deltas = [(abs(d), i) for i, d in enumerate(deltas) if d != 0]
    abs_deltas.sort(key=lambda x: x[0])
    W_pos = W_neg = 0.0
    for rank, (val, idx) in enumerate(abs_deltas, 1):
        if deltas[idx] > 0:
            W_pos += rank
        else:
            W_neg += rank
    T = n * (n + 1) / 2
    r_rb = 1.0 - 2.0 * W_neg / T if T > 0 else float('nan')
    return d_z, r_rb


def within_model_report(log_path: str, primary_temp: float = 1.0):
    """Generate full W3 within-model homogenization report per §3–§5 of design doc."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False
        print("[WARN] matplotlib not available — figures will not be generated", file=sys.stderr)

    try:
        from scipy import stats as scipy_stats
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False
        print("[WARN] scipy not available — Wilcoxon test unavailable", file=sys.stderr)

    records = [json.loads(l) for l in Path(log_path).read_text().splitlines() if l.strip()]
    print(f"\n[within_model_report] {len(records)} records from {log_path}\n")

    # ── Build per-cell summary ──────────────────────────────────────────────
    # Cell = (world_id, condition, temp) — take first record for hg metrics (all seeds share same)
    cell_data = {}
    for r in records:
        key = (r['world_id'], r['condition'], r.get('temp', primary_temp))
        if key not in cell_data:
            cell_data[key] = r
        # Accumulate g_r values for mean_g_r
        if 'g_r_list' not in cell_data[key]:
            cell_data[key]['g_r_list'] = []
        if r.get('accuracy') is not None:
            cell_data[key]['g_r_list'].append(r['accuracy'])

    # ── Table 1 ─────────────────────────────────────────────────────────────
    print("=" * 90)
    print("TABLE 1 — Per-cell results (temp={:.1f})".format(primary_temp))
    print("=" * 90)
    header = f"{'world_id':<22} {'fam':<18} {'cond':<8} {'N':>3} {'G':>6} {'H':>6} {'δ':>7} {'κ':>6} {'robust':<7} {'class':<14}"
    print(header)
    print("-" * 90)

    worlds_set = sorted(set(r['world_id'] for r in records))
    families = {}
    for r in records:
        families[r['world_id']] = r['family']

    table1_rows = []
    for world_id in sorted(worlds_set):
        for cond in ['passive', 'ON']:
            key = (world_id, cond, primary_temp)
            if key not in cell_data:
                continue
            c = cell_data[key]
            G = c.get('G')
            H = c.get('H')
            delta = c.get('delta')
            kappa = c.get('kappa', 0.0)
            kappa_robust = c.get('kappa_robust', False)
            n_seeds = len([r for r in records if r['world_id'] == world_id
                           and r['condition'] == cond
                           and abs(r.get('temp', primary_temp) - primary_temp) < 1e-6])

            if G is not None and H is not None and delta is not None and kappa_robust is not None:
                cls = _classify_cell(G, H, delta, kappa_robust)
            else:
                cls = 'NO_DATA'

            table1_rows.append({
                'world_id': world_id, 'family': families.get(world_id, '?'),
                'condition': cond, 'N': n_seeds,
                'G': G, 'H': H, 'delta': delta, 'kappa': kappa,
                'kappa_robust': kappa_robust, 'class': cls,
            })

            G_s = f"{G:.4f}" if G is not None else "N/A"
            H_s = f"{H:.4f}" if H is not None else "N/A"
            d_s = f"{delta:+.4f}" if delta is not None else "N/A"
            k_s = f"{kappa:.4f}" if kappa is not None else "N/A"
            r_s = "yes" if kappa_robust else "no"
            print(f"{world_id:<22} {families.get(world_id,'?'):<18} {cond:<8} {n_seeds:>3} "
                  f"{G_s:>6} {H_s:>6} {d_s:>7} {k_s:>6} {r_s:<7} {cls:<14}")

    # ── Table 2 — per-world Δδ ───────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("TABLE 2 — Per-world Δδ (primary temp={:.1f})".format(primary_temp))
    print("=" * 90)
    print(f"{'world_id':<22} {'family':<18} {'δ_pass':>7} {'δ_ON':>6} {'Δδ_w':>7} {'eligible':<9} {'call'}")
    print("-" * 90)

    table2_rows = []
    all_delta_w = []
    eligible_delta_w = []
    excluded_worlds = []

    for world_id in sorted(worlds_set):
        pa_key = (world_id, 'passive', primary_temp)
        on_key = (world_id, 'ON', primary_temp)
        pa_c = cell_data.get(pa_key)
        on_c = cell_data.get(on_key)

        if pa_c is None or on_c is None:
            excluded_worlds.append((world_id, 'missing_condition'))
            continue

        pa_rows = [r for r in table1_rows if r['world_id'] == world_id and r['condition'] == 'passive']
        on_rows = [r for r in table1_rows if r['world_id'] == world_id and r['condition'] == 'ON']
        pa_cls = pa_rows[0]['class'] if pa_rows else 'NO_DATA'
        on_cls = on_rows[0]['class'] if on_rows else 'NO_DATA'

        d_pa = pa_c.get('delta')
        d_on = on_c.get('delta')
        pa_robust = pa_c.get('kappa_robust', False)
        on_robust = on_c.get('kappa_robust', False)
        pa_G = pa_c.get('G', 0.0) or 0.0
        on_G = on_c.get('G', 0.0) or 0.0

        if d_pa is None or d_on is None:
            excluded_worlds.append((world_id, 'no_delta'))
            continue

        delta_w = d_on - d_pa
        all_delta_w.append(delta_w)

        # Eligibility: both non-RANDOM and both kappa_robust
        eligible = (pa_cls not in ('RANDOM', 'NO_DATA') and
                    on_cls not in ('RANDOM', 'NO_DATA') and
                    pa_robust and on_robust)

        # WEAK flag: max(G_ON, G_passive) < 0.3
        is_weak = max(pa_G, on_G) < 0.3

        if eligible and not is_weak:
            eligible_delta_w.append(delta_w)
            per_world_call = "ELIGIBLE"
        elif eligible and is_weak:
            excluded_worlds.append((world_id, 'WEAK'))
            per_world_call = "WEAK-excluded"
        elif pa_cls == 'RANDOM' or on_cls == 'RANDOM':
            excluded_worlds.append((world_id, 'RANDOM'))
            per_world_call = "RANDOM-excluded"
        elif not (pa_robust and on_robust):
            excluded_worlds.append((world_id, 'AMBIGUOUS'))
            per_world_call = "AMBIGUOUS-excluded"
        else:
            excluded_worlds.append((world_id, 'other'))
            per_world_call = "excluded"

        table2_rows.append({
            'world_id': world_id, 'family': families.get(world_id, '?'),
            'd_passive': d_pa, 'd_ON': d_on, 'delta_w': delta_w,
            'eligible': eligible and not is_weak, 'call': per_world_call,
        })

        d_pa_s = f"{d_pa:+.4f}"
        d_on_s = f"{d_on:+.4f}"
        dw_s = f"{delta_w:+.4f}"
        print(f"{world_id:<22} {families.get(world_id,'?'):<18} {d_pa_s:>7} {d_on_s:>6} "
              f"{dw_s:>7} {'yes' if (eligible and not is_weak) else 'no':<9} {per_world_call}")

    # ── Statistical tests ────────────────────────────────────────────────────
    n_eligible = len(eligible_delta_w)
    n_total = len(all_delta_w)
    excluded_list = [f"{w}({r})" for w, r in excluded_worlds]

    wilcoxon_p_eligible = float('nan')
    wilcoxon_p_full = float('nan')

    if HAS_SCIPY and n_eligible >= 2:
        try:
            res = scipy_stats.wilcoxon(eligible_delta_w, alternative='greater')
            wilcoxon_p_eligible = res.pvalue
        except Exception as e:
            print(f"[WARN] Wilcoxon on eligible set failed: {e}", file=sys.stderr)

    if HAS_SCIPY and n_total >= 2:
        try:
            res_full = scipy_stats.wilcoxon(all_delta_w, alternative='greater')
            wilcoxon_p_full = res_full.pvalue
        except Exception as e:
            print(f"[WARN] Wilcoxon on full set failed: {e}", file=sys.stderr)

    mean_delta_eligible = sum(eligible_delta_w) / n_eligible if n_eligible > 0 else float('nan')
    mean_delta_full = sum(all_delta_w) / n_total if n_total > 0 else float('nan')

    ci_lo, ci_hi = (float('nan'), float('nan'))
    if n_eligible >= 2:
        ci_lo, ci_hi = _bootstrap_mean_delta(eligible_delta_w)

    d_z, r_rb = _effect_sizes(eligible_delta_w) if n_eligible >= 2 else (float('nan'), float('nan'))

    # Model-ceiling check
    all_G_vals = []
    for world_id in sorted(worlds_set):
        pa_key = (world_id, 'passive', primary_temp)
        on_key = (world_id, 'ON', primary_temp)
        pa_c = cell_data.get(pa_key)
        on_c = cell_data.get(on_key)
        if pa_c and on_c:
            pa_G = pa_c.get('G') or 0.0
            on_G = on_c.get('G') or 0.0
            all_G_vals.append(max(pa_G, on_G))

    ceiling = all_G_vals and all(g < 0.3 for g in all_G_vals)
    ceiling_str = "CEILING: G low in all worlds" if ceiling else "PASS: grounding present"

    # Verdict
    if ceiling:
        verdict = "INCONCLUSIVE-CEILING"
    elif (not math.isnan(wilcoxon_p_eligible) and wilcoxon_p_eligible < 0.05
          and not math.isnan(ci_lo) and ci_lo > 0):
        verdict = "SUPPORTS"
    else:
        verdict = "NULL"

    print("\n" + "=" * 90)
    print("TABLE 2 FOOTER")
    print("-" * 90)
    print(f"  n_eligible={n_eligible}/9   mean_Δδ(eligible)={mean_delta_eligible:+.4f}   "
          f"CI=[{ci_lo:+.4f},{ci_hi:+.4f}]")
    print(f"  Wilcoxon p(eligible, one-sided)={wilcoxon_p_eligible:.4f}   "
          f"Wilcoxon p(full-9, one-sided)={wilcoxon_p_full:.4f}")
    print(f"  d_z={d_z:.3f}   r_rb={r_rb:.3f}")
    print(f"  mean_Δδ(full-9)={mean_delta_full:+.4f}")

    # ── Key-number block ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("KEY-NUMBER BLOCK")
    print("=" * 80)
    print(f"H-MAIN (within-model, Qwen3.6-35B-A3B, temp={primary_temp}):")
    print(f"  eligible worlds:        {n_eligible} / 9   (excluded: {', '.join(excluded_list) or 'none'})")
    print(f"  mean Δδ:                {mean_delta_eligible:+.4f}   95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  Wilcoxon (one-sided):   p = {wilcoxon_p_eligible:.4f}")
    print(f"  effect size:            d_z = {d_z:.3f},  r_rb = {r_rb:.3f}")
    print(f"  model-ceiling check:    {ceiling_str}")
    print(f"  verdict:                {verdict}")
    print("=" * 80 + "\n")

    # ── Figures ──────────────────────────────────────────────────────────────
    figs_dir = Path(log_path).parent.parent / 'figs'
    figs_dir.mkdir(parents=True, exist_ok=True)

    if HAS_MPL:
        # Figure 1: paired δ slope plot
        fig1_path = figs_dir / 'fig1_delta_paired.png'
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = {'HOMOGENIZED': 'tomato', 'GROUNDED': 'steelblue', 'WEAK': 'orange',
                  'RANDOM': 'gray', 'AMBIGUOUS': 'purple', 'NO_DATA': 'lightgray'}
        for row in table2_rows:
            world_id = row['world_id']
            pa_rows = [r for r in table1_rows if r['world_id'] == world_id and r['condition'] == 'passive']
            on_rows = [r for r in table1_rows if r['world_id'] == world_id and r['condition'] == 'ON']
            pa_cls = pa_rows[0]['class'] if pa_rows else 'NO_DATA'
            color = colors.get(pa_cls, 'black')
            lw = 2.0 if row['eligible'] else 0.8
            alpha = 1.0 if row['eligible'] else 0.4
            ax.plot([0, 1], [row['d_passive'], row['d_ON']],
                    marker='o', color=color, linewidth=lw, alpha=alpha,
                    label=f"{world_id[:15]} ({pa_cls})")
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['passive', 'ON'])
        ax.set_ylabel('δ = G − H')
        ax.set_title(f'Fig 1: Paired δ slope (temp={primary_temp})\nRising lines = intervention breaks homogenization')
        ax.legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        fig.savefig(fig1_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[Figure 1 saved] {fig1_path}")

        # Figure 3: G vs H scatter
        fig3_path = figs_dir / 'fig3_GH_scatter.png'
        fig, ax = plt.subplots(figsize=(6, 5))
        cond_colors = {'passive': 'tomato', 'ON': 'steelblue'}
        for row in table1_rows:
            if row['G'] is not None and row['H'] is not None:
                c = cond_colors.get(row['condition'], 'gray')
                ax.scatter(row['G'], row['H'], color=c, alpha=0.7, s=60)
        # Connect passive→ON per world
        for world_id in sorted(worlds_set):
            pa_r = [r for r in table1_rows if r['world_id'] == world_id and r['condition'] == 'passive']
            on_r = [r for r in table1_rows if r['world_id'] == world_id and r['condition'] == 'ON']
            if pa_r and on_r and pa_r[0]['G'] is not None and on_r[0]['G'] is not None:
                ax.annotate('', xy=(on_r[0]['G'], on_r[0]['H']),
                             xytext=(pa_r[0]['G'], pa_r[0]['H']),
                             arrowprops=dict(arrowstyle='->', color='gray', alpha=0.4))
        maxval = 1.0
        ax.plot([0, maxval], [0, maxval], 'k--', linewidth=0.8, alpha=0.4, label='G=H')
        ax.set_xlabel('G (grounding)')
        ax.set_ylabel('H (homogenization)')
        ax.set_title('Fig 3: G vs H scatter (red=passive, blue=ON)\nBelow diagonal: H>G (homogenized)')
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato', label='passive'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', label='ON'),
        ]
        ax.legend(handles=legend_elements)
        plt.tight_layout()
        fig.savefig(fig3_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[Figure 3 saved] {fig3_path}")

        # Figure 2: temperature sweep (if sweep data available)
        sweep_data = defaultdict(lambda: defaultdict(dict))
        for r in records:
            t = r.get('temp', primary_temp)
            if abs(t - primary_temp) > 1e-6:  # only non-primary temps are "sweep"
                cond = r['condition']
                sweep_data[t][cond]['G'] = r.get('G')
                sweep_data[t][cond]['H'] = r.get('H')

        # Also add primary temp data for the sweep plot
        if sweep_data or True:
            fig2_path = figs_dir / 'fig2_temp_sweep.png'
            # Collect all temps present in data
            all_temps_in_data = sorted(set(r.get('temp', primary_temp) for r in records))
            if len(all_temps_in_data) > 1:
                # Real sweep data available
                fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
                for ax_idx, cond in enumerate(['passive', 'ON']):
                    ax = axes[ax_idx]
                    G_vals_by_temp = []
                    H_vals_by_temp = []
                    temps_for_plot = []
                    for t in all_temps_in_data:
                        # Get all records at this temp and condition
                        recs_t = [r for r in records if r['condition'] == cond
                                  and abs(r.get('temp', primary_temp) - t) < 1e-6
                                  and r.get('G') is not None]
                        if recs_t:
                            # Use first record (cell-level metrics)
                            unique_cells = {}
                            for r in recs_t:
                                ck = (r['world_id'], t)
                                if ck not in unique_cells:
                                    unique_cells[ck] = r
                            G_mean = sum(r['G'] for r in unique_cells.values()) / len(unique_cells)
                            H_mean = sum(r['H'] for r in unique_cells.values()) / len(unique_cells)
                            G_vals_by_temp.append(G_mean)
                            H_vals_by_temp.append(H_mean)
                            temps_for_plot.append(t)
                    if temps_for_plot:
                        ax.plot(temps_for_plot, G_vals_by_temp, 'o-', color='steelblue', label='G')
                        ax.plot(temps_for_plot, H_vals_by_temp, 's-', color='tomato', label='H')
                        ax.set_xlabel('temperature')
                        ax.set_ylabel('G / H')
                        ax.set_title(f'Fig 2: Temp sweep ({cond})')
                        ax.legend()
                plt.suptitle('Fig 2: H(temp) and G(temp) per condition')
                plt.tight_layout()
                fig.savefig(fig2_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"[Figure 2 saved] {fig2_path}")
            else:
                # No sweep data — save placeholder
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.text(0.5, 0.5, 'Temperature sweep data not yet available\n(Run sweep job with --temps 0.3 0.7)',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Fig 2: Temperature sweep (placeholder)')
                fig.savefig(fig2_path, dpi=150)
                plt.close(fig)
                print(f"[Figure 2 saved (placeholder)] {fig2_path}")

    return {
        'n_eligible': n_eligible,
        'mean_delta': mean_delta_eligible,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'wilcoxon_p': wilcoxon_p_eligible,
        'wilcoxon_p_full': wilcoxon_p_full,
        'd_z': d_z,
        'r_rb': r_rb,
        'ceiling': ceiling,
        'verdict': verdict,
        'table1_rows': table1_rows,
        'table2_rows': table2_rows,
        'excluded_worlds': excluded_worlds,
        'figs_dir': str(figs_dir),
    }
