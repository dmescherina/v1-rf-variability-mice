"""
m=1 RF Inspector
════════════════
Run:  streamlit run streamlit_rf_inspector.py
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys, os

# Live model rendering (seed-then-refine orientation) needs sparse_noise.
# If it can't be imported, fall back to the precomputed model_map in the pickle.
try:
    sys.path.insert(0, os.path.expanduser(
        '~/dev/neuroscience/v1-dimensionality-study/src'))
    from rf_analysis.sparse_noise import best_display_orientation
    _HAVE_SN = True
except Exception:
    _HAVE_SN = False

PIX = 4.65   # degrees of visual angle per pixel (locally sparse noise 4deg grid)


def _kappa_dir(r):
    """Direction-aware elongation κ_dir = σ_orth/σ_φ.

    Returns the exact value if the record carries it (rebuilt dataset);
    otherwise reconstructs it from stored fields: the fit's differentiation
    axis is a principal axis of the envelope, so κ_dir is 1/κ (differentiation
    along the major axis) or κ (along the minor) — φ vs θ_envelope says which.
    """
    kd = r.get('kappa_dir', None)
    try:
        if kd is not None and not np.isnan(float(kd)):
            return float(kd)
    except (TypeError, ValueError):
        pass
    ph, te, k = r.get('phi_deg'), r.get('theta_envelope'), r.get('kappa')
    try:
        if any(v is None for v in (ph, te, k)) or np.isnan(ph) or np.isnan(te) or np.isnan(k):
            return float('nan')
    except (TypeError, ValueError):
        return float('nan')
    d = abs(ph - te) % 180
    d = min(d, 180 - d)
    return (1.0 / float(k)) if d < 45 else float(k)


DATASET_PATH = Path(__file__).parent / 'data' / 'm1_neuron_dataset.pkl'

st.set_page_config(
    page_title='RF Inspector · m=1',
    page_icon='🧠',
    layout='wide',
    initial_sidebar_state='collapsed',
)

st.markdown("""
<style>
    .block-container { padding-top: 3.5rem; padding-bottom: 1rem;
                       padding-left: 2rem; padding-right: 2rem; }
    .neuron-name {
        font-size: 3.6rem; font-weight: 900; letter-spacing: -0.03em;
        line-height: 1; margin-bottom: 0.1rem;
        font-family: Georgia, 'Times New Roman', serif;
    }
    .neuron-sub { font-size: 0.9rem; color: #888; margin-bottom: 0.5rem;
                  font-family: 'Courier New', monospace; }
    .flag-warn { background:#fff3cd; color:#856404; padding:2px 10px;
                 border-radius:12px; font-size:0.78rem; font-weight:600;
                 margin-right:6px; display:inline-block; }
    .description-line { font-size:0.8rem; color:#666; line-height:1.6;
                        margin-top:0.4rem; padding:0.4rem 0.8rem;
                        border-left:3px solid #e0e0e0; background:#fafafa; }
    .param-header { font-size:0.68rem; font-weight:700; text-transform:uppercase;
                    letter-spacing:0.08em; color:#aaa; margin-bottom:0.4rem;
                    border-bottom:1px solid #e0e0e0; padding-bottom:2px; }
    .param-row { display:flex; justify-content:space-between; font-size:0.85rem;
                 padding:3px 0; border-bottom:1px solid #f3f3f3; }
    .param-key { color:#666; }
    .param-val { font-weight:600; font-family:'Courier New',monospace; }
    .stTextArea textarea { font-size:0.85rem; }
</style>
""", unsafe_allow_html=True)

# ── Load ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_dataset():
    with open(DATASET_PATH, 'rb') as f:
        return pickle.load(f)

try:
    dataset = load_dataset()
except FileNotFoundError:
    st.error(f'Dataset not found at `{DATASET_PATH}`. Run notebook Cell 8 first.')
    st.stop()

N = len(dataset)

# ── Build population dataframe (once) ─────────────────────────────────────
@st.cache_data
def build_pop_df(_dataset):
    rows = []
    for r in _dataset:
        acp = np.array(r.get('angle_corr_profile', [np.nan]*18))
        sh  = (float(np.nanmax(acp)/np.nanmean(acp))
               if np.nanmean(acp) > 0 else r.get('orientation_sharpness', 0.0))
        rows.append({
            'name':        r['name'],
            'sigma':       r['sigma_deg'],
            'kappa':       r['kappa'],
            'kappa_dir':   _kappa_dir(r),
            'r2':          r['r_squared'],
            'theta':       r['theta_envelope'],
            'delta_r2':    r.get('delta_r2_vs_m0'),
            'sharpness':   sh,
            'r2_tier':     ('≥0.50' if r['r_squared'] >= 0.5
                            else '0.40–0.50' if r['r_squared'] >= 0.4 else '<0.40'),
            'sharp_tier':  ('≥2.5' if sh >= 2.5
                            else '1.5–2.5' if sh >= 1.5 else '<1.5'),
        })
    return pd.DataFrame(rows)

pop = build_pop_df(dataset)

# ── Session state ─────────────────────────────────────────────────────────
if 'idx' not in st.session_state:
    st.session_state.idx = 0

# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
tab_neuron, tab_pop = st.tabs(['🔬  Neuron inspector', '📊  Population'])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — Neuron inspector
# ══════════════════════════════════════════════════════════════════════════
with tab_neuron:

    r = dataset[st.session_state.idx]

    # ── Header ────────────────────────────────────────────────────────────
    cre_short = (r['cre_line'].replace('Cux2-CreERT2','Cux2')
                              .replace('Slc17a7-IRES2-Cre','Slc17a7'))
    depth_str = f"{r['imaging_depth_um']} µm" if r['imaging_depth_um'] else '—'

    st.markdown(f'<div class="neuron-name">{r["name"]}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="neuron-sub">#{r["name_rank"]} of {N}'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;cell_id {r["cell_id"]}'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;{cre_short}'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;{depth_str}</div>',
        unsafe_allow_html=True)

    flag_html = ''
    if r['low_kappa']:  flag_html += '<span class="flag-warn">⚠ low-κ</span>'
    if r['low_r2']:     flag_html += '<span class="flag-warn">⚠ low-R²</span>'
    if flag_html: st.markdown(flag_html, unsafe_allow_html=True)

    st.markdown(
        '<div class="description-line">'
        f'Mouse V1 m=1 simple cells &nbsp;·&nbsp; {N} neurons, all manually verified'
        '&nbsp;&nbsp;|&nbsp;&nbsp;'
        '<b>Top:</b> raw RF &amp; smoothed RF &nbsp;|&nbsp; '
        '<b>Bottom:</b> smoothed + model contours &amp; Gaussian derivative model'
        '<br>'
        'σ = spatial scale (°) &nbsp;·&nbsp; κ = elongation &nbsp;·&nbsp; '
        'θ = preferred orientation &nbsp;·&nbsp; R² = fit quality'
        '</div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top:0.5rem"></div>', unsafe_allow_html=True)

    # ── Navigation ────────────────────────────────────────────────────────
    nav_prev, nav_slider, nav_next = st.columns([1, 16, 1])
    with nav_prev:
        if st.button('←', use_container_width=True, disabled=st.session_state.idx==0):
            st.session_state.idx -= 1; st.rerun()
    with nav_next:
        if st.button('→', use_container_width=True, disabled=st.session_state.idx==N-1):
            st.session_state.idx += 1; st.rerun()
    with nav_slider:
        new_idx = st.slider('', 0, N-1, st.session_state.idx,
                            format=f'%d / {N-1}', label_visibility='collapsed')
        if new_idx != st.session_state.idx:
            st.session_state.idx = new_idx; st.rerun()

    st.markdown('<div style="margin-top:0.3rem"></div>', unsafe_allow_html=True)

    # ── Image data ────────────────────────────────────────────────────────
    rf_raw     = r['rf_raw']
    rf_smooth  = r['rf_smooth']
    rf_display = r['rf_display']
    model_map  = r['model_map']
    vmax       = r['vmax']
    x0         = r['x0_px']
    y0         = r['y0_px']
    kappa      = r['kappa']
    phi        = r['phi_deg']
    theta_env  = r['theta_envelope']
    grid_h, grid_w = r['grid_shape']
    m_peak     = r['model_abs_max']
    phi_confidence = r.get('phi_confidence', 0.0) or 0.0

    # ── Render the model panel LIVE with the seed-then-refine orientation ──
    # The pickle's precomputed model_map was selected on a coarse 10° grid,
    # which is what made the drawn dipoles look quantised / over-aligned with
    # the coordinate grid.  Recompute it here at the refined continuous angle
    # (seeded from φ and θ_envelope).  Falls back to the stored map if
    # sparse_noise isn't importable or anything goes wrong.
    if _HAVE_SN:
        try:
            su_px = r['sigma_x_deg'] / PIX
            sv_px = r['sigma_y_deg'] / PIX
            _disp = best_display_orientation(
                rf_smooth, x0, y0, su_px, sv_px, m=1,
                extra_angles_deg=(theta_env, phi),
            )
            model_map = _disp['model_map']
            m_peak    = float(np.abs(model_map).max())
        except Exception:
            pass   # keep precomputed model_map / m_peak

    # ── 2×2 figure ────────────────────────────────────────────────────────
    # Orientation bar removed — no φ/edge marker is drawn on the panels.
    CMAP = 'RdBu_r'
    PANEL_TITLES = ['raw RF', 'smoothed RF  (σ=0.75)',
                    'smoothed (low contrast)  +  model contours',
                    'model  (m=1 Gaussian derivative)']

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor('white')

    axes[0,0].imshow(rf_raw, cmap=CMAP, vmin=-vmax, vmax=vmax,
                     origin='lower', aspect='equal', interpolation='nearest')
    axes[0,1].imshow(rf_smooth, cmap=CMAP, vmin=-vmax, vmax=vmax,
                     origin='lower', aspect='equal', interpolation='nearest')

    axes[1,0].imshow(rf_display, cmap=CMAP, vmin=-vmax*2.5, vmax=vmax*2.5,
                     origin='lower', aspect='equal', interpolation='nearest')
    if m_peak > 1e-9:
        fracs  = r['contour_fracs']
        levels = sorted([-f*m_peak for f in fracs]+[f*m_peak for f in fracs])
        try:
            axes[1,0].contour(model_map, levels=levels, cmap=CMAP,
                              linewidths=1.4, alpha=0.9,
                              vmin=-m_peak, vmax=m_peak, origin='lower')
        except Exception:
            pass

    m_vmax = max(m_peak, 1e-9)
    axes[1,1].imshow(model_map, cmap=CMAP, vmin=-m_vmax, vmax=m_vmax,
                     origin='lower', aspect='equal', interpolation='nearest')

    border_colors = ['#888','#888','#2196F3','#2196F3']
    for ax, title, bc in zip(axes.flat, PANEL_TITLES, border_colors):
        ax.set_title(title, fontsize=9, pad=4, color='#333')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(bc); sp.set_linewidth(1.2)

    fig.tight_layout(pad=0.9, h_pad=1.0, w_pad=0.8)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Params ────────────────────────────────────────────────────────────
    st.divider()

    def param_block(header, rows):
        html = f'<div class="param-header">{header}</div>'
        for key, val in rows:
            html += (f'<div class="param-row"><span class="param-key">{key}</span>'
                     f'<span class="param-val">{val}</span></div>')
        st.markdown(html, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kd = _kappa_dir(r)
        kd_str = '—' if kd != kd else f"{kd:.3f}  ({'⊥-elong' if kd >= 1 else '∥-elong, Wendell'})"
        param_block('Lindeberg parameters', [
            ('σ (scale)',      f"{r['sigma_deg']:.2f}°"),
            ('κ (max/min)',    f"{r['kappa']:.3f}"),
            ('κ_dir (σ⊥/σ∥)',  kd_str),
            ('σ_major',        f"{r['sigma_x_deg']:.2f}°"),
            ('σ_minor',        f"{r['sigma_y_deg']:.2f}°"),
            ('amplitude A',    f"{r['amplitude']:.4f}"),
        ])

    with c2:
        sharpness = r.get('orientation_sharpness', 0.0) or 0.0
        sh_label  = (f"{sharpness:.2f}x  (reliable)" if sharpness >= 2.5
                     else f"{sharpness:.2f}x  (weak)" if sharpness >= 1.5
                     else f"{sharpness:.2f}x  (flat)")
        def _adiff(a, b):
            d = abs(a-b) % 180; return min(d, 180-d)
        phi_theta_diff = _adiff(phi, theta_env)
        param_block('Orientation', [
            ('θ_envelope',          f"{r['theta_envelope']:.1f}°"),
            ('orient. sharpness',   sh_label),
            ('|φ − θ| diagnostic',  f"{phi_theta_diff:.1f}°"),
            ('RF centre  x',        f"{r['x0_deg']:.1f}°"),
            ('RF centre  y',        f"{r['y0_deg']:.1f}°"),
        ])

    with c3:
        r2_m0 = f"{r['r2_m0']:.3f}"  if r['r2_m0']  is not None else '—'
        r2_m1 = f"{r['r2_m1']:.3f}"  if r['r2_m1']  is not None else '—'
        r2_m2 = f"{r['r2_m2']:.3f}"  if r['r2_m2']  is not None else '—'
        dr2   = f"{r['delta_r2_vs_m0']:.4f}" if r['delta_r2_vs_m0'] is not None else '—'
        param_block('Fit quality', [
            ('R²',               f"{r['r_squared']:.4f}"),
            ('RMSE',             f"{r['rmse']:.5f}"),
            ('AIC',              f"{r['aic']:.1f}"),
            ('ΔR²  vs m=0',      dr2),
            ('R²  m=0/m=1/m=2',  f"{r2_m0} / {r2_m1} / {r2_m2}"),
        ])

    with c4:
        cortex = (f"({r['cortex_x_um']:.0f}, {r['cortex_y_um']:.0f}) µm"
                  if r.get('cortex_x_um') is not None else '—')
        param_block('Identity', [
            ('name',          r['name']),
            ('cell_id',       str(r['cell_id'])),
            ('container_id',  str(r['container_id'])),
            ('cre_line',      cre_short),
            ('depth',         depth_str),
            ('cortex coords', cortex),
        ])

    # ── Comment ───────────────────────────────────────────────────────────
    st.divider()
    st.markdown('**Notes**')
    comment_col, btn_col = st.columns([10, 1])
    with comment_col:
        new_comment = st.text_area('', value=r['comment'], height=70,
                                   label_visibility='collapsed',
                                   key=f'comment_{st.session_state.idx}')
    with btn_col:
        st.markdown('<div style="margin-top:1.8rem"></div>', unsafe_allow_html=True)
        if st.button('Save', use_container_width=True):
            dataset[st.session_state.idx]['comment'] = new_comment
            with open(DATASET_PATH, 'wb') as f:
                pickle.dump(dataset, f, protocol=4)
            st.success('✓')


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Population distributions
# ══════════════════════════════════════════════════════════════════════════
with tab_pop:

    st.markdown(
        '### Population distributions  —  31 verified m=1 simple cells\n'
        'Mouse V1 Layer 2/3  ·  Allen Brain Observatory  ·  GCaMP6f 2-photon\n\n'
        'Colour encodes **fit quality** (R² tier) on parameter distributions, '
        'and **orientation sharpness** on the orientation panel.',
        unsafe_allow_html=False)

    C_BLUE = '#2563eb'
    C_TEAL = '#0891b2'
    R2_OPACITY = {'≥0.50': 0.90, '0.40–0.50': 0.50, '<0.40': 0.18}
    SH_OPACITY = {'≥2.5':  0.90, '1.5–2.5':   0.50, '<1.5':  0.18}
    R2_LABELS  = {'≥0.50': 'R²≥0.50 (reliable)', '0.40–0.50': 'R²=0.40–0.50', '<0.40': 'R²<0.40 (weak)'}
    SH_LABELS  = {'≥2.5': 'sharp ≥2.5', '1.5–2.5': 'sharp 1.5–2.5', '<1.5': 'sharp <1.5 (flat)'}

    PL = dict(
        paper_bgcolor='white', plot_bgcolor='#f8fafc',
        font_family='Inter, Helvetica, Arial, sans-serif',
        font_color='#1e293b', font_size=12,
        margin=dict(l=20, r=20, t=40, b=30),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1, font_size=10),
    )

    def vline(fig, x, color, dash, label):
        fig.add_vline(x=float(x), line=dict(color=color, dash=dash, width=1.5),
                      annotation_text=label, annotation_position='top',
                      annotation_font_size=9, annotation_font_color=color)

    # ── Filters ───────────────────────────────────────────────────────────
    with st.expander('🔍  Filters', expanded=False):
        fc1, fc2, fc3 = st.columns([2, 1.5, 1.5])
        with fc1:
            sel_names = st.multiselect(
                'Neurons', options=sorted(pop['name'].tolist()),
                default=[], placeholder='All neurons')
        with fc2:
            r2_range = st.slider('R² range', 0.30, 1.00, (0.30, 1.00), step=0.01)
        with fc3:
            k_range = st.slider('κ range', 1.00, 3.00, (1.00, 3.00), step=0.05)

    # Build filtered frame — always reset index so boolean masks align
    pf = pop.copy()
    if sel_names:
        pf = pf[pf['name'].isin(sel_names)]
    pf = pf[pf['r2'].between(*r2_range) & pf['kappa'].between(*k_range)]
    pf = pf.reset_index(drop=True)   # ← prevents IndexingError
    NF = len(pf)
    if NF < N:
        st.caption(f'Showing {NF} of {N} neurons after filters')

    if NF == 0:
        st.warning('No neurons match the current filters.')
        st.stop()

    # ── Row 1: σ | κ | R² | ΔR² ──────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        fig = go.Figure(layout=PL)
        for tier, op in R2_OPACITY.items():
            sub = pf[pf['r2_tier'] == tier]
            fig.add_trace(go.Histogram(
                x=sub['sigma'], name=R2_LABELS[tier], nbinsx=12,
                marker_color=C_BLUE, opacity=op,
                marker_line=dict(color='white', width=0.5)))
        cv = pf['sigma'].std() / pf['sigma'].mean()
        vline(fig, pf['sigma'].mean(), '#334155', 'dot', f"μ={pf['sigma'].mean():.2f}°")
        fig.update_layout(barmode='overlay', title=f'σ  (CV={cv:.3f})',
                          xaxis_title='σ (°)', yaxis_title='count', **PL)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(layout=PL)
        for tier, op in R2_OPACITY.items():
            sub = pf[pf['r2_tier'] == tier]
            fig.add_trace(go.Histogram(
                x=sub['kappa'], name=R2_LABELS[tier], nbinsx=12,
                marker_color=C_BLUE, opacity=op,
                marker_line=dict(color='white', width=0.5)))
        vline(fig, pf['kappa'].median(), '#334155', 'dot', f"med={pf['kappa'].median():.2f}")
        fig.update_layout(barmode='overlay', title='κ  (elongation)',
                          xaxis_title='κ', yaxis_title='count', **PL)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        fig = go.Figure(layout=PL)
        fig.add_trace(go.Histogram(x=pf['r2'], nbinsx=14,
                                   marker_color='#3b82f6', opacity=0.85,
                                   marker_line=dict(color='white', width=0.5),
                                   name='R²'))
        vline(fig, 0.40, '#ef4444', 'dash', 'R²=0.40')
        vline(fig, 0.50, '#22c55e', 'dash', 'R²=0.50')
        vline(fig, pf['r2'].mean(), '#334155', 'dot', f"μ={pf['r2'].mean():.3f}")
        fig.update_layout(title='Fit quality R²', xaxis_title='R²',
                          yaxis_title='count', **PL)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        dr2 = pf['delta_r2'].dropna()
        fig = go.Figure(layout=PL)
        fig.add_trace(go.Histogram(x=dr2, nbinsx=14, marker_color='#3b82f6',
                                   opacity=0.85,
                                   marker_line=dict(color='white', width=0.5),
                                   name='ΔR²'))
        vline(fig, 0.08, '#f97316', 'dash', 'threshold=0.08')
        if len(dr2):
            vline(fig, float(dr2.mean()), '#334155', 'dot', f"μ={dr2.mean():.3f}")
        fig.update_layout(title='ΔR²  (m=1 over m=0)', xaxis_title='ΔR²',
                          yaxis_title='count', **PL)
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: θ | σ–κ scatter | sharpness ────────────────────────────────
    col5, col6, col7 = st.columns([1.1, 1.4, 1.1])

    with col5:
        from scipy.stats import chisquare as _chi2st
        bins_t = np.arange(0, 181, 20)
        fig = go.Figure(layout=PL)
        for tier, op in SH_OPACITY.items():
            sub_t = pf[pf['sharp_tier'] == tier]['theta'].dropna()
            counts, _ = np.histogram(sub_t, bins=bins_t)
            fig.add_trace(go.Bar(
                x=(bins_t[:-1] + bins_t[1:]) / 2, y=counts,
                name=SH_LABELS[tier], marker_color=C_TEAL,
                marker_line=dict(color='white', width=0.5),
                width=18, opacity=op))
        fig.add_hline(y=NF / 9, line_dash='dash', line_color='#334155',
                      line_width=1.5, annotation_text='Uniform',
                      annotation_font_size=9)
        obs_t, _ = np.histogram(pf['theta'].dropna(), bins=9, range=(0, 180))
        chi2, pv_chi = _chi2st(obs_t)
        fig.update_layout(
            barmode='stack',
            title=f'Orientation θ  (χ²={chi2:.1f} p={pv_chi:.2f})<br>'
                  f'<sub>colour = orientation sharpness tier</sub>',
            xaxis_title='θ_envelope (°)', yaxis_title='count',
            xaxis=dict(range=[0, 180], dtick=30), **PL)
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        from scipy.stats import pearsonr as _pr
        fig = go.Figure(layout=PL)
        for tier, op in R2_OPACITY.items():
            sub = pf[pf['r2_tier'] == tier]
            fig.add_trace(go.Scatter(
                x=sub['sigma'], y=sub['kappa'],
                mode='markers+text',
                text=sub['name'],
                textposition='top right',
                textfont=dict(size=8, color='#64748b'),
                marker=dict(size=10, color=C_BLUE, opacity=op,
                            line=dict(color='white', width=1)),
                name=R2_LABELS[tier],
                customdata=sub[['r2', 'sharpness']].values,
                hovertemplate=(
                    '<b>%{text}</b><br>'
                    'σ=%{x:.2f}°  κ=%{y:.3f}<br>'
                    'R²=%{customdata[0]:.3f}  sharpness=%{customdata[1]:.2f}x'
                    '<extra></extra>')))
        valid = pf[['sigma', 'kappa']].dropna()
        rv, pv_r = _pr(valid['sigma'], valid['kappa']) if len(valid) > 2 else (float('nan'), float('nan'))
        fig.update_layout(
            title=f'σ–κ coupling   r={rv:.3f}   p={pv_r:.3f}',
            xaxis_title='σ (°)', yaxis_title='κ', **PL)
        st.plotly_chart(fig, use_container_width=True)

    with col7:
        fig = go.Figure(layout=PL)
        fig.add_trace(go.Histogram(
            x=pf['sharpness'], nbinsx=12,
            marker_color='#60a5fa', opacity=0.85,
            marker_line=dict(color='white', width=0.5),
            name='all neurons'))
        fig.add_vline(x=1.5, line_dash='dash', line_color='#f97316',
                      line_width=1.5, annotation_text='weak', annotation_font_size=9)
        fig.add_vline(x=2.5, line_dash='dash', line_color='#22c55e',
                      line_width=1.5, annotation_text='reliable', annotation_font_size=9)
        fig.add_vline(x=pf['sharpness'].mean(), line_dash='dot',
                      line_color='#334155', line_width=1.5,
                      annotation_text=f"μ={pf['sharpness'].mean():.2f}x",
                      annotation_font_size=9)
        fig.update_layout(
            title='Orientation sharpness<br><sub>max/mean |r| — 10° grid</sub>',
            xaxis_title='sharpness (×)', yaxis_title='count', **PL)
        st.plotly_chart(fig, use_container_width=True)

    # ── Summary stats ─────────────────────────────────────────────────────
    st.divider()
    from scipy.stats import circstd as _csd, chisquare as _chi, pearsonr as _pr2

    cv_sig   = pf['sigma'].std() / pf['sigma'].mean()
    rv2, pv2 = _pr2(pf['sigma'], pf['kappa']) if NF > 2 else (float('nan'), float('nan'))
    # circstd: double angles for π-periodicity, halve result
    csd_deg  = np.rad2deg(_csd(np.deg2rad(pf['theta'].dropna()) * 2)) / 2
    obs9, _  = np.histogram(pf['theta'].dropna(), bins=9, range=(0, 180))
    chi2s, chi2p = _chi(obs9)

    c_a, c_b, c_c, c_d = st.columns(4)

    with c_a:
        st.markdown('**Spatial scale σ**')
        st.markdown(f"""
| stat | value |
|------|-------|
| mean ± SD | {pf['sigma'].mean():.2f} ± {pf['sigma'].std():.2f}° |
| range | {pf['sigma'].min():.1f} – {pf['sigma'].max():.1f}° |
| CV | {cv_sig:.3f} |
| Lindeberg threshold | 0.300 |
""")
        verdict = '✓ below cascade threshold' if cv_sig < 0.3 else '⚠ above cascade threshold'
        st.info(f'CV(σ) = {cv_sig:.3f} — {verdict}.')

    with c_b:
        st.markdown('**Elongation κ**  (max/min vs direction-aware)')
        has_kd = 'kappa_dir' in pf.columns and pf['kappa_dir'].notna().any()
        kd = pf['kappa_dir'] if has_kd else pd.Series([np.nan])
        st.markdown(f"""
| stat | κ (max/min) | κ_dir (σ⊥/σ∥) |
|------|------|------|
| median | {pf['kappa'].median():.3f} | {kd.median():.3f} |
| mean ± SD | {pf['kappa'].mean():.3f} ± {pf['kappa'].std():.3f} | {kd.mean():.3f} ± {kd.std():.3f} |
| range | {pf['kappa'].min():.2f} – {pf['kappa'].max():.2f} | {kd.min():.2f} – {kd.max():.2f} |
| κ_dir < 1 (Wendell) | — | {int((kd < 1).sum())}/{NF} |
""")
        st.info(('κ range {:.2f}–{:.2f} confirms predicted variability in elongation. '
                 'κ_dir<1 cells are elongated along the differentiation axis (Wendell).'
                 ).format(pf['kappa'].min(), pf['kappa'].max())
                if has_kd else
                'Range {:.2f}–{:.2f} confirms predicted variability. Rebuild dataset to populate κ_dir.'
                .format(pf['kappa'].min(), pf['kappa'].max()))

    with c_c:
        st.markdown('**Orientation θ & σ–κ coupling**')
        theta_v = '✓ uniform (salt-and-pepper)' if chi2p > 0.05 else '⚠ deviation from uniform'
        coup_v  = '✓ significant' if pv2 < 0.05 else f'underpowered (n={NF})'
        st.markdown(f"""
| stat | value |
|------|-------|
| θ circular SD | {csd_deg:.1f}° (uniform≈54°) |
| χ² uniformity | {chi2s:.1f},  p={chi2p:.2f} |
| σ–κ  r | {rv2:.3f} |
| σ–κ  p | {pv2:.4f} |
""")
        st.info(f'Orientation: {theta_v}  (p={chi2p:.2f}).  '
                f'σ–κ coupling: {coup_v}  (r={rv2:.3f}, p={pv2:.4f}).')

    with c_d:
        st.markdown('**Fit quality & sharpness**')
        st.markdown(f"""
| stat | value |
|------|-------|
| R² mean ± SD | {pf['r2'].mean():.3f} ± {pf['r2'].std():.3f} |
| R² ≥ 0.50 | {(pf['r2'] >= 0.5).sum()}/{NF} |
| sharpness mean | {pf['sharpness'].mean():.2f}× |
| sharpness range | {pf['sharpness'].min():.2f} – {pf['sharpness'].max():.2f}× |
| any > 2.5 | {'yes' if (pf['sharpness'] >= 2.5).any() else 'none'} |
""")
        st.info(f'Max sharpness {pf["sharpness"].max():.2f}×. '
                f'Population ceiling from 9×16, 4.65°/pixel grid.')
