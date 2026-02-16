"""
お金の流れ可視化アプリ (Streamlit版)
マネーフォワードME形式のCSVを読み込み、収支ダッシュボード・Sankeyフロー図を表示する。
"""

import io
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = ["日付", "金額（円）", "大項目", "中項目"]
WASTE_CATEGORIES = ["娯楽", "交際費", "衣服・美容", "趣味"]
WASTE_THRESHOLD = 0.30
SAVINGS_TARGET = 0.30  # 目標貯蓄率 30%

FIXED_COST_CATEGORIES = ["住宅", "水道・光熱費", "通信費", "保険"]
VARIABLE_COST_CATEGORIES = ["食費", "交通費", "交際費", "娯楽", "衣服・美容",
                            "教養・教育", "日用品", "趣味"]

CATEGORY_COLORS = [
    "#3498db", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22",
    "#34495e", "#16a085", "#2980b9", "#8e44ad", "#27ae60",
]
COLOR_INCOME = "#27ae60"
COLOR_EXPENSE = "#e74c3c"
COLOR_BALANCE = "#3498db"
COLOR_WASTE = "#e74c3c"

DEFAULT_BUDGETS = {
    "食費": 40000, "住宅": 85000, "水道・光熱費": 15000, "通信費": 8000,
    "交通費": 15000, "交際費": 15000, "娯楽": 5000, "衣服・美容": 10000,
    "教養・教育": 5000, "日用品": 5000, "趣味": 5000, "保険": 10000,
}

PIE_TOP_N = 6  # 円グラフ上位N件 + その他

# ---------------------------------------------------------------------------
# ヘルパー関数
# ---------------------------------------------------------------------------

def format_yen(amount: int | float) -> str:
    return f"¥{int(amount):,.0f}"


def is_waste_category(cat: str) -> bool:
    return cat in WASTE_CATEGORIES


def load_csv(file_or_path) -> pd.DataFrame:
    for enc in ("utf-8", "shift_jis", "cp932"):
        try:
            if isinstance(file_or_path, str):
                df = pd.read_csv(file_or_path, encoding=enc)
            else:
                file_or_path.seek(0)
                df = pd.read_csv(file_or_path, encoding=enc)
            if all(c in df.columns for c in REQUIRED_COLUMNS):
                return _clean(df)
        except (UnicodeDecodeError, UnicodeError):
            continue
    st.error("CSVの読み込みに失敗しました。マネーフォワードME形式か確認してください。")
    return pd.DataFrame()


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["金額（円）"] = (
        df["金額（円）"].astype(str)
        .str.replace(",", "", regex=False)
        .apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    )
    df["日付"] = df["日付"].astype(str)
    if "計算対象" in df.columns:
        df["計算対象"] = pd.to_numeric(df["計算対象"], errors="coerce").fillna(1).astype(int)
    else:
        df["計算対象"] = 1
    if "振替" in df.columns:
        df["振替"] = pd.to_numeric(df["振替"], errors="coerce").fillna(0).astype(int)
    else:
        df["振替"] = 0
    df["大項目"] = df["大項目"].fillna("不明")
    df["中項目"] = df["中項目"].fillna("不明")
    return df


def get_active_records(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["計算対象"] == 1) & (df["振替"] == 0)].copy()


def get_months(df: pd.DataFrame) -> list[str]:
    months = (
        df["日付"].str[:7].str.replace("/", "-", regex=False)
        .dropna().unique().tolist()
    )
    months = [m for m in months if len(m) == 7 and m[0].isdigit()]
    months.sort(reverse=True)
    return months


def get_years(df: pd.DataFrame) -> list[str]:
    years = df["日付"].str[:4].dropna().unique().tolist()
    years = [y for y in years if len(y) == 4 and y.isdigit()]
    years.sort(reverse=True)
    return years


def filter_by_period(df: pd.DataFrame, mode: str, value: str | None) -> pd.DataFrame:
    if not value or mode == "全期間":
        return df
    if mode == "年":
        return df[df["日付"].str[:4] == value].copy()
    if mode == "クイック":
        return _filter_quick(df, value)
    ym = df["日付"].str[:7].str.replace("/", "-", regex=False)
    return df[ym == value].copy()


def _filter_quick(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """クイック選択: 直近N ヶ月"""
    month_map = {"直近3ヶ月": 3, "直近6ヶ月": 6, "直近12ヶ月": 12}
    n = month_map.get(label, 3)
    now = datetime.now()
    cutoff = now - timedelta(days=n * 31)
    cutoff_str = cutoff.strftime("%Y/%m")
    ym = df["日付"].str[:7]
    return df[ym >= cutoff_str].copy()


def _prev_month(ym: str) -> str:
    """YYYY-MM → 前月の YYYY-MM"""
    y, m = int(ym[:4]), int(ym[5:7])
    m -= 1
    if m == 0:
        m, y = 12, y - 1
    return f"{y:04d}-{m:02d}"


def _prev_year_month(ym: str) -> str:
    """YYYY-MM → 前年同月の YYYY-MM"""
    y, m = int(ym[:4]), int(ym[5:7])
    return f"{y - 1:04d}-{m:02d}"


def build_sankey_data(df: pd.DataFrame, detail: bool = False,
                      min_amount: int = 0, show_cats: list[str] | None = None):
    """Sankey 用の nodes / links を構築する。"""
    active = get_active_records(df)
    income = active[active["金額（円）"] > 0]
    expense = active[active["金額（円）"] < 0].copy()
    expense["金額（円）"] = expense["金額（円）"].abs()

    # カテゴリフィルター
    if show_cats:
        expense = expense[expense["大項目"].isin(show_cats)]

    labels: list[str] = []
    label_idx: dict[str, int] = {}

    def idx(name: str) -> int:
        if name not in label_idx:
            label_idx[name] = len(labels)
            labels.append(name)
        return label_idx[name]

    sources, targets, values, link_colors = [], [], [], []

    income_total_label = "収入合計"
    idx(income_total_label)
    for cat_m, grp in income.groupby("中項目"):
        src = f"【収入】{cat_m}"
        s, t = idx(src), idx(income_total_label)
        v = int(grp["金額（円）"].sum())
        if v > 0:
            sources.append(s); targets.append(t); values.append(v)
            link_colors.append("rgba(39,174,96,0.35)")

    # 大項目集計 → 小さいものを「その他」にまとめる
    cat_l_sums = expense.groupby("大項目")["金額（円）"].sum().sort_values(ascending=False)
    other_total = 0
    main_cats = []
    for cat_l, v in cat_l_sums.items():
        if v < min_amount:
            other_total += v
        else:
            main_cats.append(cat_l)

    for cat_l in main_cats:
        grp = expense[expense["大項目"] == cat_l]
        s, t = idx(income_total_label), idx(cat_l)
        v = int(grp["金額（円）"].sum())
        if v > 0:
            sources.append(s); targets.append(t); values.append(v)
            c = "rgba(231,76,60,0.35)" if is_waste_category(cat_l) else "rgba(100,100,100,0.15)"
            link_colors.append(c)

    if other_total > 0:
        s, t = idx(income_total_label), idx("その他")
        sources.append(s); targets.append(t); values.append(int(other_total))
        link_colors.append("rgba(100,100,100,0.15)")

    if detail:
        for cat_l in main_cats:
            grp_all = expense[expense["大項目"] == cat_l]
            for cat_m, grp in grp_all.groupby("中項目"):
                s = idx(cat_l)
                sub_label = f"{cat_m}\u3000" if cat_m in label_idx else cat_m
                t = idx(sub_label)
                v = int(grp["金額（円）"].sum())
                if v > 0:
                    sources.append(s); targets.append(t); values.append(v)
                    c = "rgba(231,76,60,0.35)" if is_waste_category(cat_l) else "rgba(100,100,100,0.15)"
                    link_colors.append(c)

    node_colors: list[str] = []
    for lab in labels:
        if lab.startswith("【収入】"):
            node_colors.append(COLOR_INCOME)
        elif lab == income_total_label:
            node_colors.append("#2ecc71")
        elif is_waste_category(lab):
            node_colors.append(COLOR_WASTE)
        elif lab == "その他":
            node_colors.append("#95a5a6")
        else:
            h = sum(ord(c) for c in lab)
            node_colors.append(CATEGORY_COLORS[h % len(CATEGORY_COLORS)])

    return labels, sources, targets, values, node_colors, link_colors


# ---------------------------------------------------------------------------
# ページ: ウェルカム
# ---------------------------------------------------------------------------

def page_welcome():
    st.title("💰 お金の流れ可視化")
    st.markdown("マネーフォワードME形式のCSVをアップロードして、収支を可視化しましょう。")
    st.markdown(f"必須カラム: `{'`, `'.join(REQUIRED_COLUMNS)}`")

    # MoneyForward エクスポート手順
    with st.expander("📖 マネーフォワードMEからのCSVエクスポート手順"):
        st.markdown("""
1. [マネーフォワードME](https://moneyforward.com/) にログイン
2. **家計簿** → **家計簿（月別）** を開く
3. 画面右上の **「ダウンロード」** ボタンをクリック
4. **CSV形式** でダウンロード
5. ダウンロードしたCSVファイルをこの画面にアップロード

※ 複数月分をまとめてアップロードすると、月別推移も確認できます。
""")

    st.markdown("")

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🎮 デモ版を表示", type="primary", use_container_width=True):
            try:
                demo_df = load_csv("demo_data.csv")
                if not demo_df.empty:
                    st.session_state["df"] = demo_df
                    st.session_state["data_source"] = "デモデータ"
                    st.rerun()
            except FileNotFoundError:
                st.error("demo_data.csv が見つかりません。")

    st.divider()

    uploaded = st.file_uploader(
        "CSVファイルをドラッグ＆ドロップまたは選択",
        type=["csv"],
        help="マネーフォワードMEからエクスポートしたCSVファイル",
    )
    if uploaded is not None:
        df = load_csv(io.BytesIO(uploaded.getvalue()))
        if not df.empty:
            st.success(f"{len(df)}件のデータを読み込みました。")
            preview_cols = [c for c in ["日付", "内容", "金額（円）", "大項目", "中項目", "保有金融機関"] if c in df.columns]
            st.dataframe(df[preview_cols].head(10), use_container_width=True)
            if st.button("✅ このデータを使用する", type="primary"):
                st.session_state["df"] = df
                st.session_state["data_source"] = uploaded.name
                st.rerun()


# ---------------------------------------------------------------------------
# ページ: ダッシュボード
# ---------------------------------------------------------------------------

def page_dashboard(df: pd.DataFrame, period_mode: str, period_value: str | None):
    active_all = get_active_records(df)
    filtered = filter_by_period(active_all, period_mode, period_value)
    if filtered.empty:
        st.info("選択した期間にデータがありません。")
        return

    income_total = int(filtered[filtered["金額（円）"] > 0]["金額（円）"].sum())
    expense_abs = int(filtered[filtered["金額（円）"] < 0]["金額（円）"].sum())
    balance = income_total + expense_abs  # expense_abs is negative
    savings_rate = balance / income_total if income_total > 0 else 0

    # ====== メトリクスカード + 貯蓄率ゲージ ======
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("収入", format_yen(income_total))
    c2.metric("支出", format_yen(abs(expense_abs)))
    c3.metric("収支", format_yen(balance), delta=f"{balance:+,.0f}")

    # 貯蓄率ゲージ
    gauge_color = COLOR_INCOME if savings_rate >= SAVINGS_TARGET else "#f39c12" if savings_rate >= 0 else COLOR_EXPENSE
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=savings_rate * 100,
        number={"suffix": "%", "font": {"size": 28}},
        title={"text": "貯蓄率", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": gauge_color},
            "steps": [
                {"range": [0, SAVINGS_TARGET * 100], "color": "#f0f0f0"},
                {"range": [SAVINGS_TARGET * 100, 100], "color": "#e8f5e9"},
            ],
            "threshold": {
                "line": {"color": COLOR_INCOME, "width": 3},
                "thickness": 0.8,
                "value": SAVINGS_TARGET * 100,
            },
        },
    ))
    fig_gauge.update_layout(height=180, margin=dict(t=30, b=10, l=30, r=30))
    c4.plotly_chart(fig_gauge, use_container_width=True)

    # ====== 前月比・前年同月比 ======
    if period_mode == "月" and period_value:
        _show_comparison(active_all, period_value)

    # ====== 円グラフ: カテゴリ別支出 (上位N + その他 & ドリルダウン) ======
    expenses = filtered[filtered["金額（円）"] < 0].copy()
    expenses["金額（円）"] = expenses["金額（円）"].abs()
    cat_sum = expenses.groupby("大項目")["金額（円）"].sum().sort_values(ascending=False)

    if not cat_sum.empty:
        _draw_pie_with_drilldown(expenses, cat_sum)

    # ====== 棒グラフ + 収支差額折れ線 ======
    _draw_bar_with_balance_line(active_all, period_mode)

    # ====== 固定費 vs 変動費 ======
    if not cat_sum.empty:
        _draw_fixed_vs_variable(cat_sum)

    # ====== 浪費チェック ======
    if not cat_sum.empty:
        _draw_waste_alert(cat_sum)


def _show_comparison(active_all: pd.DataFrame, current_month: str):
    """前月比・前年同月比を表示"""
    def _month_totals(ym: str):
        ym_col = active_all["日付"].str[:7].str.replace("/", "-", regex=False)
        m = active_all[ym_col == ym]
        inc = int(m[m["金額（円）"] > 0]["金額（円）"].sum())
        exp = int(m[m["金額（円）"] < 0]["金額（円）"].sum())
        return inc, exp

    cur_inc, cur_exp = _month_totals(current_month)
    prev_m = _prev_month(current_month)
    prev_y = _prev_year_month(current_month)
    prev_inc, prev_exp = _month_totals(prev_m)
    prevy_inc, prevy_exp = _month_totals(prev_y)

    items = []
    if prev_inc or prev_exp:
        diff_exp = abs(cur_exp) - abs(prev_exp)
        diff_inc = cur_inc - prev_inc
        items.append(f"**前月比** ({prev_m}): 収入 {diff_inc:+,.0f}円 / 支出 {diff_exp:+,.0f}円")
    if prevy_inc or prevy_exp:
        diff_exp = abs(cur_exp) - abs(prevy_exp)
        diff_inc = cur_inc - prevy_inc
        items.append(f"**前年同月比** ({prev_y}): 収入 {diff_inc:+,.0f}円 / 支出 {diff_exp:+,.0f}円")

    if items:
        # カテゴリ別の大きな変動をハイライト
        ym_col = active_all["日付"].str[:7].str.replace("/", "-", regex=False)
        cur_cats = active_all[(ym_col == current_month) & (active_all["金額（円）"] < 0)].copy()
        cur_cats["金額（円）"] = cur_cats["金額（円）"].abs()
        cur_by_cat = cur_cats.groupby("大項目")["金額（円）"].sum()

        prev_cats = active_all[(ym_col == prev_m) & (active_all["金額（円）"] < 0)].copy()
        prev_cats["金額（円）"] = prev_cats["金額（円）"].abs()
        prev_by_cat = prev_cats.groupby("大項目")["金額（円）"].sum()

        alerts = []
        for cat in cur_by_cat.index:
            cur_v = cur_by_cat.get(cat, 0)
            prev_v = prev_by_cat.get(cat, 0)
            if prev_v > 0:
                diff = cur_v - prev_v
                if abs(diff) >= 5000:  # 5000円以上の変動
                    emoji = "📈" if diff > 0 else "📉"
                    alerts.append(f"{emoji} {cat}: {diff:+,.0f}円（前月比）")

        with st.expander("📊 前月比・前年同月比", expanded=True):
            for item in items:
                st.markdown(item)
            if alerts:
                st.markdown("---")
                st.markdown("**カテゴリ別の大きな変動（±5,000円以上）:**")
                for a in alerts:
                    st.markdown(f"- {a}")


def _draw_pie_with_drilldown(expenses: pd.DataFrame, cat_sum: pd.Series):
    """円グラフ: 上位N + その他、クリックで中項目ドリルダウン"""
    # 上位N + その他
    if len(cat_sum) > PIE_TOP_N:
        top = cat_sum.iloc[:PIE_TOP_N]
        other = cat_sum.iloc[PIE_TOP_N:].sum()
        display_sum = pd.concat([top, pd.Series({"その他": other})])
    else:
        display_sum = cat_sum

    pie_colors = []
    for i, c in enumerate(display_sum.index):
        if c == "その他":
            pie_colors.append("#95a5a6")
        elif is_waste_category(c):
            pie_colors.append(COLOR_WASTE)
        else:
            pie_colors.append(CATEGORY_COLORS[i % len(CATEGORY_COLORS)])

    fig_pie = go.Figure(go.Pie(
        labels=display_sum.index,
        values=display_sum.values,
        hole=0.4,
        marker=dict(colors=pie_colors, line=dict(color="#fff", width=2)),
        textinfo="percent",
        textposition="inside",
        textfont=dict(size=13, color="#fff"),
        hovertemplate="<b>%{label}</b><br>¥%{value:,.0f}<br>%{percent}<extra></extra>",
        pull=[0.03] * len(display_sum),
    ))
    fig_pie.update_layout(
        title="カテゴリ別支出",
        margin=dict(t=50, b=30, l=20, r=20),
        height=420,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5,
            font=dict(size=13),
        ),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # ドリルダウン: 大項目を選ぶと中項目の内訳を表示
    selected_cat = st.selectbox(
        "カテゴリを選択して中項目の内訳を表示",
        ["（選択してください）"] + list(cat_sum.index),
        key="drilldown_cat",
    )
    if selected_cat != "（選択してください）":
        sub = expenses[expenses["大項目"] == selected_cat]
        sub_sum = sub.groupby("中項目")["金額（円）"].sum().sort_values(ascending=False)
        if not sub_sum.empty:
            sub_colors = [CATEGORY_COLORS[i % len(CATEGORY_COLORS)] for i in range(len(sub_sum))]
            fig_sub = go.Figure(go.Pie(
                labels=sub_sum.index, values=sub_sum.values, hole=0.35,
                marker=dict(colors=sub_colors, line=dict(color="#fff", width=2)),
                textinfo="label+percent", textposition="inside",
                textfont=dict(size=12, color="#fff"),
                hovertemplate="<b>%{label}</b><br>¥%{value:,.0f}<br>%{percent}<extra></extra>",
            ))
            fig_sub.update_layout(
                title=f"{selected_cat} の内訳",
                margin=dict(t=50, b=20, l=20, r=20), height=350,
                showlegend=True,
                legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_sub, use_container_width=True)


def _draw_bar_with_balance_line(active_all: pd.DataFrame, period_mode: str):
    """棒グラフ + 収支差額の折れ線"""
    if period_mode == "年":
        active_all = active_all.copy()
        active_all["期間"] = active_all["日付"].str[:4]
        title_bar = "年別収支推移"
    else:
        active_all = active_all.copy()
        active_all["期間"] = active_all["日付"].str[:7].str.replace("/", "-", regex=False)
        title_bar = "月別収支推移"

    m_inc = active_all[active_all["金額（円）"] > 0].groupby("期間")["金額（円）"].sum()
    m_exp = active_all[active_all["金額（円）"] < 0].groupby("期間")["金額（円）"].sum().abs()
    periods = sorted(set(m_inc.index) | set(m_exp.index))

    if not periods:
        return

    inc_vals = [m_inc.get(p, 0) for p in periods]
    exp_vals = [m_exp.get(p, 0) for p in periods]
    bal_vals = [m_inc.get(p, 0) - m_exp.get(p, 0) for p in periods]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=periods, y=inc_vals, name="収入", marker_color=COLOR_INCOME,
        hovertemplate="収入: ¥%{y:,.0f}<extra></extra>",
    ))
    fig_bar.add_trace(go.Bar(
        x=periods, y=exp_vals, name="支出", marker_color=COLOR_EXPENSE,
        hovertemplate="支出: ¥%{y:,.0f}<extra></extra>",
    ))
    # 収支差額の折れ線
    fig_bar.add_trace(go.Scatter(
        x=periods, y=bal_vals, name="収支差額", mode="lines+markers",
        line=dict(color=COLOR_BALANCE, width=3),
        marker=dict(size=8),
        hovertemplate="収支: ¥%{y:,.0f}<extra></extra>",
        yaxis="y",
    ))
    fig_bar.update_layout(
        title=title_bar, barmode="group",
        margin=dict(t=40, b=20, l=20, r=20), height=420,
        yaxis_tickformat=",",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_bar, use_container_width=True)


def _draw_fixed_vs_variable(cat_sum: pd.Series):
    """固定費 vs 変動費の内訳"""
    fixed = sum(cat_sum.get(c, 0) for c in FIXED_COST_CATEGORIES)
    variable = sum(cat_sum.get(c, 0) for c in VARIABLE_COST_CATEGORIES)
    other = cat_sum.sum() - fixed - variable

    st.subheader("固定費 vs 変動費")
    col1, col2, col3 = st.columns(3)
    col1.metric("固定費", format_yen(fixed))
    col2.metric("変動費", format_yen(variable))
    col3.metric("その他", format_yen(max(other, 0)))

    vals = [fixed, variable]
    labs = ["固定費", "変動費"]
    if other > 0:
        vals.append(other)
        labs.append("その他")

    fig = go.Figure(go.Pie(
        labels=labs, values=vals, hole=0.4,
        marker=dict(colors=["#3498db", "#f39c12", "#95a5a6"][:len(vals)],
                    line=dict(color="#fff", width=2)),
        textinfo="label+percent", textposition="inside",
        textfont=dict(size=13, color="#fff"),
        hovertemplate="<b>%{label}</b><br>¥%{value:,.0f}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(t=20, b=20, l=20, r=20), height=300, showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("固定費・変動費の分類"):
        fc = {c: cat_sum.get(c, 0) for c in FIXED_COST_CATEGORIES if cat_sum.get(c, 0) > 0}
        vc = {c: cat_sum.get(c, 0) for c in VARIABLE_COST_CATEGORIES if cat_sum.get(c, 0) > 0}
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**固定費**")
            for c, v in sorted(fc.items(), key=lambda x: -x[1]):
                st.markdown(f"- {c}: {format_yen(v)}")
        with c2:
            st.markdown("**変動費**")
            for c, v in sorted(vc.items(), key=lambda x: -x[1]):
                st.markdown(f"- {c}: {format_yen(v)}")


def _draw_waste_alert(cat_sum: pd.Series):
    """浪費チェック"""
    total_expense = cat_sum.sum()
    waste_cats = {c: v for c, v in cat_sum.items() if is_waste_category(c)}
    waste_total = sum(waste_cats.values())
    waste_pct = waste_total / total_expense if total_expense > 0 else 0

    st.subheader("浪費チェック")
    if waste_pct > WASTE_THRESHOLD:
        st.warning(f"浪費系カテゴリが支出の **{waste_pct:.0%}** を占めています（閾値: {WASTE_THRESHOLD:.0%}）")
    else:
        st.success(f"浪費系カテゴリは支出の **{waste_pct:.0%}** です（閾値: {WASTE_THRESHOLD:.0%} 以内）")
    for cat, val in waste_cats.items():
        pct = val / total_expense if total_expense > 0 else 0
        st.markdown(f"- **{cat}**: {format_yen(val)}（{pct:.1%}）")


# ---------------------------------------------------------------------------
# ページ: 予算比較
# ---------------------------------------------------------------------------

def page_budget(df: pd.DataFrame, period_mode: str, period_value: str | None):
    st.subheader("カテゴリ別 予算 vs 実績")

    # 予算設定 (session_state に保存)
    if "budgets" not in st.session_state:
        st.session_state["budgets"] = DEFAULT_BUDGETS.copy()
    budgets = st.session_state["budgets"]

    with st.expander("⚙ 予算設定を編集"):
        active = get_active_records(df)
        all_cats = sorted(active[active["金額（円）"] < 0]["大項目"].unique())
        cols = st.columns(3)
        for i, cat in enumerate(all_cats):
            with cols[i % 3]:
                budgets[cat] = st.number_input(
                    cat, min_value=0, step=1000,
                    value=budgets.get(cat, 10000), key=f"budget_{cat}",
                )
        st.session_state["budgets"] = budgets

    # 実績
    filtered = filter_by_period(get_active_records(df), period_mode, period_value)
    expenses = filtered[filtered["金額（円）"] < 0].copy()
    expenses["金額（円）"] = expenses["金額（円）"].abs()
    actual = expenses.groupby("大項目")["金額（円）"].sum()

    # 月数で按分 (全期間・年の場合)
    months_in_period = max(1, len(get_months(filtered)))
    if period_mode in ("全期間", "年", "クイック") and months_in_period > 1:
        st.caption(f"※ {months_in_period}ヶ月分のデータ → 月平均で表示")
        actual = actual / months_in_period

    all_cats_budget = sorted(set(list(budgets.keys()) + list(actual.index)))
    cats, budget_vals, actual_vals, diff_vals = [], [], [], []
    for cat in all_cats_budget:
        b = budgets.get(cat, 0)
        a = actual.get(cat, 0)
        if b > 0 or a > 0:
            cats.append(cat)
            budget_vals.append(b)
            actual_vals.append(a)
            diff_vals.append(b - a)

    if not cats:
        st.info("データがありません。")
        return

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=cats, x=budget_vals, name="予算", orientation="h",
        marker_color="#bdc3c7",
        hovertemplate="%{y}: 予算 ¥%{x:,.0f}<extra></extra>",
    ))
    bar_colors = [COLOR_EXPENSE if a > b else COLOR_INCOME
                  for a, b in zip(actual_vals, budget_vals)]
    fig.add_trace(go.Bar(
        y=cats, x=actual_vals, name="実績", orientation="h",
        marker_color=bar_colors,
        hovertemplate="%{y}: 実績 ¥%{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        barmode="group", height=max(300, len(cats) * 45 + 80),
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis_tickformat=",",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 差額サマリー
    over = [(c, d) for c, d in zip(cats, diff_vals) if d < 0]
    under = [(c, d) for c, d in zip(cats, diff_vals) if d >= 0]
    if over:
        st.warning("**予算オーバー:**")
        for c, d in sorted(over, key=lambda x: x[1]):
            st.markdown(f"- {c}: **{format_yen(abs(d))} オーバー**")
    if under:
        st.success("**予算内:**")
        for c, d in sorted(under, key=lambda x: -x[1]):
            st.markdown(f"- {c}: {format_yen(d)} 余裕あり")


# ---------------------------------------------------------------------------
# ページ: フロー図
# ---------------------------------------------------------------------------

def page_sankey(df: pd.DataFrame, period_mode: str, period_value: str | None):
    filtered = filter_by_period(df, period_mode, period_value)
    active = get_active_records(filtered)
    if active.empty:
        st.info("選択した期間にデータがありません。")
        return

    # オプション
    col_opt1, col_opt2 = st.columns([1, 2])
    with col_opt1:
        detail = st.toggle("中項目まで表示", value=False)
    with col_opt2:
        all_cats = sorted(active[active["金額（円）"] < 0]["大項目"].unique())
        show_cats = st.multiselect("表示カテゴリ（空=全て）", all_cats, default=[], key="sankey_cats")

    min_amount = st.slider("最小表示金額（これ未満は「その他」にまとめる）",
                           0, 20000, 0, step=1000, key="sankey_min")

    labels, sources, targets, values, node_colors, link_colors = build_sankey_data(
        filtered, detail=detail, min_amount=min_amount,
        show_cats=show_cats if show_cats else None,
    )

    if not sources:
        st.info("フロー図を描画するデータがありません。")
        return

    if detail:
        height = max(450, len(labels) * 22 + 80)
    else:
        height = max(400, len(labels) * 30 + 80)

    # ラベルに金額を付与
    label_values = [0] * len(labels)
    for s, t, v in zip(sources, targets, values):
        label_values[t] += v
        if labels[s].startswith("【収入】"):
            label_values[s] += v
    for i, lab in enumerate(labels):
        if lab == "収入合計":
            label_values[i] = sum(v for s, v in zip(sources, values) if labels[s].startswith("【収入】"))

    display_labels = [f"{lab}  ¥{label_values[i]:,.0f}" for i, lab in enumerate(labels)]

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=18, thickness=28, label=display_labels, color=node_colors,
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=sources, target=targets, value=values, color=link_colors,
            hovertemplate="%{source.label}<br>→ %{target.label}<extra></extra>",
        ),
    ))
    fig.update_layout(
        title=dict(text="お金の流れ", font=dict(size=20)),
        font=dict(size=15, color="#222", family="'Noto Sans JP', 'Yu Gothic UI', Meiryo, sans-serif"),
        height=height,
        margin=dict(t=60, b=30, l=10, r=10),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# サイドバー
# ---------------------------------------------------------------------------

def sidebar_data_management():
    source = st.session_state.get("data_source", "")
    df = st.session_state.get("df", pd.DataFrame())
    active = get_active_records(df) if not df.empty else pd.DataFrame()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**📂 読込データ**")
    st.sidebar.caption(f"ソース: {source}")
    st.sidebar.caption(f"全件: {len(df)} / 有効: {len(active)}")

    if not df.empty:
        months = get_months(active)
        if months:
            st.sidebar.caption(f"期間: {months[-1]} 〜 {months[0]}")

    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        if st.button("🔄 別データ", use_container_width=True, key="btn_reload"):
            for k in ["df", "data_source"]:
                st.session_state.pop(k, None)
            st.rerun()
    with col_b:
        if st.button("🎮 デモ版", use_container_width=True, key="btn_demo_side"):
            try:
                demo_df = load_csv("demo_data.csv")
                if not demo_df.empty:
                    st.session_state["df"] = demo_df
                    st.session_state["data_source"] = "デモデータ"
                    st.rerun()
            except FileNotFoundError:
                st.sidebar.error("demo_data.csv が見つかりません")

    with st.sidebar.expander("📎 CSVを追加読込"):
        uploaded = st.file_uploader("CSVファイル", type=["csv"], key="sidebar_upload")
        if uploaded is not None:
            new_df = load_csv(io.BytesIO(uploaded.getvalue()))
            if not new_df.empty:
                st.sidebar.success(f"{len(new_df)}件")
                if st.button("差し替える", key="btn_replace"):
                    st.session_state["df"] = new_df
                    st.session_state["data_source"] = uploaded.name
                    st.rerun()


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="お金の流れ可視化", page_icon="💰", layout="wide")

    df: pd.DataFrame = st.session_state.get("df", pd.DataFrame())
    if df.empty:
        page_welcome()
        return

    # サイドバー
    st.sidebar.title("💰 お金の流れ")
    page = st.sidebar.radio("ページ", ["ダッシュボード", "フロー図", "予算比較"])

    # 期間フィルター
    st.sidebar.markdown("---")

    # クイック選択ボタン
    st.sidebar.markdown("**クイック選択**")
    qcol1, qcol2, qcol3 = st.sidebar.columns(3)
    quick_val = None
    if qcol1.button("3ヶ月", use_container_width=True, key="q3"):
        st.session_state["period_mode"] = "クイック"
        st.session_state["quick_val"] = "直近3ヶ月"
    if qcol2.button("6ヶ月", use_container_width=True, key="q6"):
        st.session_state["period_mode"] = "クイック"
        st.session_state["quick_val"] = "直近6ヶ月"
    if qcol3.button("12ヶ月", use_container_width=True, key="q12"):
        st.session_state["period_mode"] = "クイック"
        st.session_state["quick_val"] = "直近12ヶ月"

    period_mode = st.sidebar.radio(
        "期間の単位", ["全期間", "月", "年"],
        horizontal=True,
        index=["全期間", "月", "年"].index(
            st.session_state.get("period_mode", "全期間")
            if st.session_state.get("period_mode", "全期間") in ("全期間", "月", "年")
            else "全期間"
        ),
    )
    # クイック選択がアクティブならそちらを優先
    if st.session_state.get("period_mode") == "クイック":
        period_mode = "クイック"
        quick_val = st.session_state.get("quick_val", "直近3ヶ月")
        st.sidebar.info(f"📅 {quick_val}")
        if st.sidebar.button("クイック選択を解除"):
            st.session_state.pop("period_mode", None)
            st.session_state.pop("quick_val", None)
            st.rerun()
    else:
        st.session_state.pop("period_mode", None)
        st.session_state.pop("quick_val", None)

    period_value: str | None = None
    active = get_active_records(df)
    if period_mode == "月":
        months = get_months(active)
        if months:
            period_value = st.sidebar.selectbox("月を選択", months)
    elif period_mode == "年":
        years = get_years(active)
        if years:
            period_value = st.sidebar.selectbox("年を選択", years)
    elif period_mode == "クイック":
        period_value = quick_val

    sidebar_data_management()

    # ページ描画
    if page == "ダッシュボード":
        st.title("ダッシュボード")
        page_dashboard(df, period_mode, period_value)
    elif page == "フロー図":
        st.title("フロー図")
        page_sankey(df, period_mode, period_value)
    elif page == "予算比較":
        st.title("予算比較")
        page_budget(df, period_mode, period_value)


if __name__ == "__main__":
    main()
