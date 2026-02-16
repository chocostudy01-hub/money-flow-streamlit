"""
お金の流れ可視化アプリ (Streamlit版)
マネーフォワードME形式のCSVを読み込み、収支ダッシュボード・Sankeyフロー図を表示する。
"""

import io
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = ["日付", "金額（円）", "大項目", "中項目"]
WASTE_CATEGORIES = ["娯楽", "交際費", "衣服・美容", "趣味"]
WASTE_THRESHOLD = 0.30  # 30%

CATEGORY_COLORS = [
    "#3498db", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22",
    "#34495e", "#16a085", "#2980b9", "#8e44ad", "#27ae60",
]
COLOR_INCOME = "#27ae60"
COLOR_EXPENSE = "#e74c3c"
COLOR_BALANCE = "#3498db"
COLOR_WASTE = "#e74c3c"

# ---------------------------------------------------------------------------
# ヘルパー関数
# ---------------------------------------------------------------------------

def format_yen(amount: int) -> str:
    """金額を ¥1,234 形式にフォーマット"""
    return f"¥{amount:,.0f}"


def is_waste_category(cat: str) -> bool:
    return cat in WASTE_CATEGORIES


def load_csv(file_or_path) -> pd.DataFrame:
    """CSVを読み込み、標準カラム名の DataFrame を返す。
    encoding は utf-8 → shift_jis の順で試行する。"""
    for enc in ("utf-8", "shift_jis", "cp932"):
        try:
            if isinstance(file_or_path, (str,)):
                df = pd.read_csv(file_or_path, encoding=enc)
            else:
                file_or_path.seek(0)
                df = pd.read_csv(file_or_path, encoding=enc)
            # 必須カラムチェック
            if all(c in df.columns for c in REQUIRED_COLUMNS):
                return _clean(df)
        except (UnicodeDecodeError, UnicodeError):
            continue
    st.error("CSVの読み込みに失敗しました。マネーフォワードME形式か確認してください。")
    return pd.DataFrame()


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """型変換・欠損補完"""
    df = df.copy()
    # 金額: カンマ除去して整数化
    df["金額（円）"] = (
        df["金額（円）"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
    )
    # 日付文字列
    df["日付"] = df["日付"].astype(str)
    # フラグ
    if "計算対象" in df.columns:
        df["計算対象"] = pd.to_numeric(df["計算対象"], errors="coerce").fillna(1).astype(int)
    else:
        df["計算対象"] = 1
    if "振替" in df.columns:
        df["振替"] = pd.to_numeric(df["振替"], errors="coerce").fillna(0).astype(int)
    else:
        df["振替"] = 0
    # カテゴリ欠損
    df["大項目"] = df["大項目"].fillna("不明")
    df["中項目"] = df["中項目"].fillna("不明")
    return df


def get_active_records(df: pd.DataFrame) -> pd.DataFrame:
    """計算対象かつ振替でないレコードを返す"""
    return df[(df["計算対象"] == 1) & (df["振替"] == 0)].copy()


def get_months(df: pd.DataFrame) -> list[str]:
    """YYYY-MM 形式の月リストを新しい順に返す"""
    months = (
        df["日付"]
        .str[:7]
        .str.replace("/", "-", regex=False)
        .dropna()
        .unique()
        .tolist()
    )
    months = [m for m in months if len(m) == 7 and m[0].isdigit()]
    months.sort(reverse=True)
    return months


def filter_by_month(df: pd.DataFrame, month: str | None) -> pd.DataFrame:
    if not month:
        return df
    ym = df["日付"].str[:7].str.replace("/", "-", regex=False)
    return df[ym == month].copy()


def build_sankey_data(df: pd.DataFrame):
    """Sankey 用の nodes / links を構築する。
    収入源 → 収入合計 → 大項目 → 中項目"""
    active = get_active_records(df)
    income = active[active["金額（円）"] > 0]
    expense = active[active["金額（円）"] < 0].copy()
    expense["金額（円）"] = expense["金額（円）"].abs()

    labels: list[str] = []
    label_idx: dict[str, int] = {}

    def idx(name: str) -> int:
        if name not in label_idx:
            label_idx[name] = len(labels)
            labels.append(name)
        return label_idx[name]

    sources, targets, values, link_colors = [], [], [], []

    # --- 収入源 → 収入合計 ---
    income_total_label = "収入合計"
    idx(income_total_label)
    for cat_m, grp in income.groupby("中項目"):
        src = f"【収入】{cat_m}"
        s = idx(src)
        t = idx(income_total_label)
        v = int(grp["金額（円）"].sum())
        if v > 0:
            sources.append(s)
            targets.append(t)
            values.append(v)
            link_colors.append("rgba(39,174,96,0.3)")

    # --- 収入合計 → 大項目 ---
    for cat_l, grp in expense.groupby("大項目"):
        s = idx(income_total_label)
        t = idx(cat_l)
        v = int(grp["金額（円）"].sum())
        if v > 0:
            sources.append(s)
            targets.append(t)
            values.append(v)
            c = "rgba(231,76,60,0.3)" if is_waste_category(cat_l) else "rgba(0,0,0,0.1)"
            link_colors.append(c)

    # --- 大項目 → 中項目 ---
    for (cat_l, cat_m), grp in expense.groupby(["大項目", "中項目"]):
        s = idx(cat_l)
        # 中項目名の重複回避 (全角スペース付加)
        sub_label = f"{cat_m}\u3000" if cat_m in label_idx else cat_m
        t = idx(sub_label)
        v = int(grp["金額（円）"].sum())
        if v > 0:
            sources.append(s)
            targets.append(t)
            values.append(v)
            c = "rgba(231,76,60,0.3)" if is_waste_category(cat_l) else "rgba(0,0,0,0.1)"
            link_colors.append(c)

    # --- ノード色 ---
    node_colors: list[str] = []
    for lab in labels:
        if lab.startswith("【収入】"):
            node_colors.append(COLOR_INCOME)
        elif lab == income_total_label:
            node_colors.append("#2ecc71")
        elif is_waste_category(lab):
            node_colors.append(COLOR_WASTE)
        else:
            h = sum(ord(c) for c in lab)
            node_colors.append(CATEGORY_COLORS[h % len(CATEGORY_COLORS)])

    return labels, sources, targets, values, node_colors, link_colors


# ---------------------------------------------------------------------------
# ページ描画
# ---------------------------------------------------------------------------

def page_dashboard(df: pd.DataFrame, month: str | None):
    filtered = filter_by_month(get_active_records(df), month)
    if filtered.empty:
        st.info("データがありません。CSVを読み込んでください。")
        return

    income_total = int(filtered[filtered["金額（円）"] > 0]["金額（円）"].sum())
    expense_total = int(filtered[filtered["金額（円）"] < 0]["金額（円）"].sum())
    balance = income_total + expense_total  # expense is negative

    # --- メトリクスカード ---
    c1, c2, c3 = st.columns(3)
    c1.metric("収入", format_yen(income_total))
    c2.metric("支出", format_yen(abs(expense_total)))
    c3.metric("収支", format_yen(balance), delta=f"{balance:+,.0f}")

    # --- チャート ---
    col_pie, col_bar = st.columns(2)

    # ドーナツ: カテゴリ別支出
    expenses = filtered[filtered["金額（円）"] < 0].copy()
    expenses["金額（円）"] = expenses["金額（円）"].abs()
    cat_sum = expenses.groupby("大項目")["金額（円）"].sum().sort_values(ascending=False)

    if not cat_sum.empty:
        pie_colors = [
            COLOR_WASTE if is_waste_category(c) else CATEGORY_COLORS[i % len(CATEGORY_COLORS)]
            for i, c in enumerate(cat_sum.index)
        ]
        fig_pie = go.Figure(
            go.Pie(
                labels=cat_sum.index,
                values=cat_sum.values,
                hole=0.45,
                marker=dict(colors=pie_colors),
                textinfo="label+percent",
                hovertemplate="%{label}: ¥%{value:,.0f} (%{percent})<extra></extra>",
            )
        )
        fig_pie.update_layout(
            title="カテゴリ別支出",
            margin=dict(t=40, b=20, l=20, r=20),
            height=400,
            showlegend=True,
            legend=dict(orientation="v", x=1.05),
        )
        col_pie.plotly_chart(fig_pie, use_container_width=True)

    # 棒グラフ: 月別収支推移
    active = get_active_records(df)  # 全期間データを使用
    active["月"] = active["日付"].str[:7].str.replace("/", "-", regex=False)
    monthly_income = active[active["金額（円）"] > 0].groupby("月")["金額（円）"].sum()
    monthly_expense = active[active["金額（円）"] < 0].groupby("月")["金額（円）"].sum().abs()
    months_all = sorted(set(monthly_income.index) | set(monthly_expense.index))

    if months_all:
        fig_bar = go.Figure()
        fig_bar.add_trace(
            go.Bar(
                x=months_all,
                y=[monthly_income.get(m, 0) for m in months_all],
                name="収入",
                marker_color=COLOR_INCOME,
                hovertemplate="収入: ¥%{y:,.0f}<extra></extra>",
            )
        )
        fig_bar.add_trace(
            go.Bar(
                x=months_all,
                y=[monthly_expense.get(m, 0) for m in months_all],
                name="支出",
                marker_color=COLOR_EXPENSE,
                hovertemplate="支出: ¥%{y:,.0f}<extra></extra>",
            )
        )
        fig_bar.update_layout(
            title="月別収支推移",
            barmode="group",
            margin=dict(t=40, b=20, l=20, r=20),
            height=400,
            yaxis_tickformat=",",
        )
        col_bar.plotly_chart(fig_bar, use_container_width=True)

    # --- 浪費アラート ---
    if not cat_sum.empty:
        total_expense = cat_sum.sum()
        waste_cats = {c: v for c, v in cat_sum.items() if is_waste_category(c)}
        waste_total = sum(waste_cats.values())
        waste_pct = waste_total / total_expense if total_expense > 0 else 0

        st.subheader("浪費チェック")
        if waste_pct > WASTE_THRESHOLD:
            st.warning(
                f"浪費系カテゴリが支出の **{waste_pct:.0%}** を占めています（閾値: {WASTE_THRESHOLD:.0%}）"
            )
        else:
            st.success(
                f"浪費系カテゴリは支出の **{waste_pct:.0%}** です（閾値: {WASTE_THRESHOLD:.0%} 以内）"
            )
        for cat, val in waste_cats.items():
            pct = val / total_expense if total_expense > 0 else 0
            st.markdown(f"- **{cat}**: {format_yen(val)}（{pct:.1%}）")


def page_sankey(df: pd.DataFrame, month: str | None):
    filtered = filter_by_month(df, month)
    active = get_active_records(filtered)
    if active.empty:
        st.info("データがありません。CSVを読み込んでください。")
        return

    labels, sources, targets, values, node_colors, link_colors = build_sankey_data(filtered)

    if not sources:
        st.info("フロー図を描画するデータがありません。")
        return

    height = max(500, len(labels) * 25 + 100)

    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=12,
                thickness=20,
                label=labels,
                color=node_colors,
                hovertemplate="%{label}: ¥%{value:,.0f}<extra></extra>",
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
                hovertemplate="%{source.label} → %{target.label}: ¥%{value:,.0f}<extra></extra>",
            ),
        )
    )
    fig.update_layout(
        title="お金の流れ (Sankey)",
        font_size=12,
        height=height,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def page_upload():
    st.header("CSV取込")
    st.markdown("マネーフォワードME形式のCSVをアップロードしてください。")
    st.markdown(f"必須カラム: `{'`, `'.join(REQUIRED_COLUMNS)}`")

    # デモデータ読み込み
    if st.button("デモデータを読み込む"):
        try:
            demo_df = load_csv("demo_data.csv")
            if not demo_df.empty:
                st.session_state["df"] = demo_df
                st.success(f"デモデータを読み込みました（{len(demo_df)}件）")
                st.rerun()
        except FileNotFoundError:
            st.error("demo_data.csv が見つかりません。")

    st.divider()

    uploaded = st.file_uploader("CSVファイルを選択", type=["csv"])
    if uploaded is not None:
        df = load_csv(io.BytesIO(uploaded.getvalue()))
        if not df.empty:
            st.success(f"{len(df)}件のデータを読み込みました。")
            # プレビュー
            preview_cols = [c for c in ["日付", "内容", "金額（円）", "大項目", "中項目", "保有金融機関"] if c in df.columns]
            st.dataframe(df[preview_cols].head(10), use_container_width=True)
            if st.button("このデータを使用する"):
                st.session_state["df"] = df
                st.success("データを保存しました。")
                st.rerun()


# ---------------------------------------------------------------------------
# メインレイアウト
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="お金の流れ可視化",
        page_icon="💰",
        layout="wide",
    )

    # --- サイドバー ---
    st.sidebar.title("💰 お金の流れ")
    page = st.sidebar.radio("ページ", ["ダッシュボード", "フロー図", "CSV取込"])

    # データ初期化: session_state にデータがなければデモデータ読み込みを試みる
    if "df" not in st.session_state:
        try:
            st.session_state["df"] = load_csv("demo_data.csv")
        except Exception:
            st.session_state["df"] = pd.DataFrame()

    df: pd.DataFrame = st.session_state.get("df", pd.DataFrame())

    # 月フィルター
    month = None
    if not df.empty and page != "CSV取込":
        active = get_active_records(df)
        months = get_months(active)
        if months:
            options = ["全期間"] + months
            selected = st.sidebar.selectbox("月を選択", options)
            if selected != "全期間":
                month = selected

    # --- ページ描画 ---
    if page == "ダッシュボード":
        st.title("ダッシュボード")
        page_dashboard(df, month)
    elif page == "フロー図":
        st.title("フロー図")
        page_sankey(df, month)
    else:
        page_upload()


if __name__ == "__main__":
    main()
