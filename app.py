import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.data_loader.loader import DataLoader
from modules.data_loader.markets import MARKET_CONFIG, Market
from modules.data_loader.models import AutoFetchResult
from modules.pipeline import ValuationPipeline

st.set_page_config(page_title="企业估值辅助工具", page_icon="📊", layout="wide")

st.title("企业估值辅助工具 (MVP)")

MODULES = [
    "总览",
    "1. 数据导入 (data_loader)",
    "2. 财务比率 (ratio_analysis)",
    "3. 财务预测 (forecasting)",
    "4. 资本成本 (cost_of_capital)",
    "5. 绝对估值 (dcf_valuation)",
    "6. 相对估值 (relative_valuation)",
    "7. 图表与评级 (football_chart)",
]

page = st.sidebar.radio("功能模块", MODULES)

DATA_RANGE_OPTIONS: dict[str, int | None] = {
    "3年": 3,
    "5年": 5,
    "10年": 10,
    "全部可用数据": None,
}
data_range_label = st.sidebar.selectbox(
    "数据范围",
    options=list(DATA_RANGE_OPTIONS.keys()),
    index=1,
    help="按当前日期倒推完整财年。例如 2026 年选 5 年时加载 2021–2025；某年缺失则向前补齐。更改后请重新抓取。",
)
st.session_state.financial_years = DATA_RANGE_OPTIONS[data_range_label]

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
**模块调用关系**

```
data_loader
  ├─> ratio_analysis ──> relative_valuation ──┐
  ├─> forecasting ──> dcf_valuation           │
  │                      ↑ cost_of_capital     │
  └─> (snapshot) ──> relative_valuation        │
                                               v
                                        football_chart
```
"""
)

pipeline = ValuationPipeline()

if "fetch_result" not in st.session_state:
    st.session_state.fetch_result = None
if "financials_df" not in st.session_state:
    st.session_state.financials_df = None


def render_sentiment_dashboard(stats) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("正面", stats.positive)
    col2.metric("中性", stats.neutral)
    col3.metric("负面", stats.negative)
    col4.metric("合计", stats.total)
    if stats.total > 0:
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["正面", "中性", "负面"],
                    values=[stats.positive, stats.neutral, stats.negative],
                    hole=0.45,
                    marker={"colors": ["#2ecc71", "#95a5a6", "#e74c3c"]},
                )
            ]
        )
        fig.update_layout(title="新闻情绪分布", height=320, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)


def render_macro_cards(macro) -> None:
    c1, c2, c3 = st.columns(3)
    if macro.gdp_growth is not None:
        c1.metric(
            f"GDP 增速 ({macro.gdp_growth_year})",
            f"{macro.gdp_growth:.1f}%",
            help=f"数据来源: IMF · {macro.country_name}",
        )
    else:
        c1.metric("GDP 增速", "—")
    if macro.inflation is not None:
        c2.metric(
            f"通胀率 ({macro.inflation_year})",
            f"{macro.inflation:.1f}%",
            help="数据来源: IMF PCPIPCH",
        )
    else:
        c2.metric("通胀率", "—")
    c3.metric("国家/地区", macro.country_name)


def render_fetch_errors(result: AutoFetchResult) -> None:
    if result.errors:
        with st.expander("API 降级详情（部分数据源不可用）", expanded=result.mode == "manual"):
            for source, msg in result.errors.items():
                st.warning(f"**{source}**: {msg}")
    for warn in result.warnings:
        st.info(warn)

# ── 示例数据（含周转率所需扩展字段） ──
SAMPLE_FINANCIALS = pd.DataFrame(
    {
        "year": [2021, 2022, 2023, 2024],
        "revenue": [800, 920, 1050, 1180],
        "net_income": [80, 92, 105, 118],
        "total_assets": [600, 650, 710, 780],
        "total_equity": [400, 430, 470, 520],
        "gross_profit": [320, 360, 410, 460],
        "operating_cost": [480, 560, 640, 720],
        "accounts_receivable": [90, 95, 100, 110],
        "inventory": [120, 130, 140, 150],
        "accounts_payable": [70, 72, 78, 85],
        "rd_expense": [40, 45, 50, 55],
        "pretax_profit": [105, 120, 138, 155],
        "income_tax": [25, 28, 33, 37],
    }
)


def get_active_financials() -> tuple[pd.DataFrame, str]:
    """按优先级获取当前财务数据：session > 抓取结果 > 示例。"""
    if st.session_state.get("financials_df") is not None:
        return st.session_state.financials_df, "模块一已保存数据"
    fetch = st.session_state.get("fetch_result")
    if fetch is not None and fetch.financials is not None:
        st.session_state.financials_df = fetch.financials
        return fetch.financials, f"模块一自动抓取（{fetch.market} {fetch.stock_code}）"
    return SAMPLE_FINANCIALS, "内置示例数据"


def _render_annual_forecast(
    fin_df,
    market,
    tax_rate_input,
    macro,
    ctx_key,
    pipeline,
) -> None:
    from modules.forecasting.assumptions import validate_terminal_growth
    from modules.forecasting.group9_assumptions import (
        build_group9_assumptions_table,
        infer_product_services_share,
        sellside_blended_growth,
    )
    from modules.forecasting.session_helpers import (
        default_gdp_growth,
        forecast_amount_scale,
        forecast_for_display,
        forecast_table_column_config,
    )
    from modules.forecasting.three_statement import infer_forecast_drivers, infer_group9_drivers

    default_gdp = default_gdp_growth(market, macro)
    default_terminal = float(min(st.session_state.get("terminal_growth", min(0.03, default_gdp)), default_gdp))
    gdp_source = (
        f"IMF {macro.gdp_growth_year} 年 {macro.country_name} {macro.gdp_growth:.1f}%"
        if macro is not None and macro.gdp_growth is not None
        else f"{'A 股' if market == 'A股' else '美股'}默认 {default_gdp:.1%}"
    )

    st.subheader("卖方 Y1 收入增速（Deutsche / JPM / Wells Fargo）")
    ss1, ss2, ss3 = st.columns(3)
    db_g = ss1.number_input("Deutsche Bank Y1 增速", 0.0, 0.30, 0.06, 0.005, format="%.3f")
    jpm_g = ss2.number_input("JPMorgan Y1 增速", 0.0, 0.30, 0.07, 0.005, format="%.3f")
    wf_g = ss3.number_input("Wells Fargo Y1 增速", 0.0, 0.30, 0.065, 0.005, format="%.3f")
    blended = sellside_blended_growth(db_g, jpm_g, wf_g)
    st.caption(f"三家算术平均 Y1 增速：**{blended:.1%}** → taper 至终值 g")

    prod_share, svc_share = infer_product_services_share(fin_df)
    seg1, seg2, seg3 = st.columns(3)
    pg1 = seg1.number_input("Product Y1 增速", -0.05, 0.30, blended - 0.01, 0.005, format="%.3f")
    sg1 = seg2.number_input("Services Y1 增速", -0.05, 0.30, blended + 0.02, 0.005, format="%.3f")
    seg3.caption(f"收入结构 Product **{prod_share:.0%}** / Services **{svc_share:.0%}**")

    forecast_years = st.slider("预测年数", 5, 10, 6, 1, key="annual_forecast_years")
    gm_target = st.number_input("毛利率目标（GM ramp 上限）", 0.30, 0.70, 0.50, 0.01, format="%.2f")
    rd_terminal = st.number_input("R&D/收入终值", 0.04, 0.15, 0.08, 0.005, format="%.3f")

    if st.button("按卖方输入重新生成假设表", key="btn_rebuild_group9_assumptions"):
        st.session_state.pop("assumptions_df", None)
        st.session_state.pop("assumption_years", None)

    if (
        "assumptions_df" not in st.session_state
        or st.session_state.get("assumption_years") != forecast_years
        or st.session_state.get("group9_ctx") != ctx_key
    ):
        st.session_state.assumptions_df = build_group9_assumptions_table(
            fin_df,
            forecast_years=forecast_years,
            terminal_growth=default_terminal,
            sellside_blended=blended,
            product_growth_y1=pg1,
            services_growth_y1=sg1,
            gm_target=gm_target,
            rd_terminal_pct=rd_terminal,
        )
        st.session_state.assumption_years = forecast_years
        st.session_state.group9_ctx = ctx_key

    st.caption("收入 Product/Services 分项增速 + GM 每年 +1pct + R&D taper 至 8% + SG&A 维持")
    edited = st.data_editor(
        st.session_state.assumptions_df,
        num_rows="fixed",
        column_config={
            "year": st.column_config.NumberColumn("年份", disabled=True, format="%d"),
            "revenue_growth": st.column_config.NumberColumn("合计增速", format="%.2%%"),
            "product_growth": st.column_config.NumberColumn("Product 增速", format="%.2%%"),
            "services_growth": st.column_config.NumberColumn("Services 增速", format="%.2%%"),
            "gross_margin": st.column_config.NumberColumn("毛利率", format="%.2%%"),
            "rd_pct": st.column_config.NumberColumn("R&D/收入", format="%.2%%"),
            "sga_pct": st.column_config.NumberColumn("SG&A/收入", format="%.2%%"),
            "product_share": st.column_config.NumberColumn("Product 占比", format="%.2%%", disabled=True),
            "services_share": st.column_config.NumberColumn("Services 占比", format="%.2%%", disabled=True),
        },
        use_container_width=True,
        key=f"group9_assumptions_{ctx_key}",
    )
    st.session_state.assumptions_df = edited

    gdp_col, tg_col = st.columns(2)
    gdp_growth = gdp_col.number_input(
        f"GDP 增速上限（{gdp_source}）",
        0.0,
        0.15,
        float(st.session_state.get("gdp_growth", default_gdp)),
        0.005,
        format="%.3f",
        key="annual_gdp_cap",
    )
    st.session_state.gdp_growth = gdp_growth
    terminal_input = tg_col.number_input(
        "终值永续增长率",
        0.0,
        float(gdp_growth),
        float(min(st.session_state.get("terminal_growth", min(0.03, default_gdp)), gdp_growth)),
        0.005,
        format="%.3f",
        key="annual_terminal_g",
    )
    tg_check = validate_terminal_growth(terminal_input, gdp_growth)
    st.session_state.terminal_growth = tg_check.terminal_growth
    if tg_check.is_valid:
        st.success(tg_check.message)
    else:
        st.warning(tg_check.message)

    st.subheader("三表联动预测（FCFE · 权益 plug）")
    st.caption(
        "**FCFE** = 净利润 + 折旧 − CapEx − Δ营运资本 + 净增债务 · "
        "营运资本 = 应收 + 存货 − 应付 · 权益作平衡 plug"
    )
    forecast = pipeline.forecaster.forecast_from_assumptions(fin_df, edited, tax_rate=tax_rate_input)
    st.session_state.forecast_df = forecast
    st.dataframe(
        forecast_for_display(forecast, market),
        column_config=forecast_table_column_config(market),
        use_container_width=True,
        hide_index=True,
    )
    if "product_revenue" in forecast.columns:
        seg_show = forecast[["year", "product_revenue", "services_revenue", "revenue"]].copy()
        scale, unit = forecast_amount_scale(market)
        seg_show["product_revenue"] /= scale
        seg_show["services_revenue"] /= scale
        seg_show["revenue"] /= scale
        st.caption(f"分业务收入（{unit}）")
        st.dataframe(seg_show, use_container_width=True, hide_index=True)

    with st.expander("预测驱动比率（自历史推断）"):
        drv = infer_group9_drivers(fin_df)
        st.write(
            f"**预测驱动** · AR {drv.ar_turnover:.1f}x（基准年锁定）· "
            f"Inv {drv.inventory_turnover:.1f}x · AP {drv.ap_turnover:.1f}x · "
            f"D/A {drv.debt_to_assets:.1%}（四年均值）· CapEx {drv.capex_pct_revenue:.1%} · "
            f"D/A折旧 {drv.da_pct_revenue:.1%} · SG&A {drv.sga_pct_revenue:.1%}（基准年）"
        )


def _render_quarterly_forecast(fin_df, market, tax_rate_input, pipeline) -> None:
    from modules.data_loader.us_quarterly_fetcher import fetch_us_quarterly_financials
    from modules.forecasting.quarterly_forecast import (
        QuarterlyInputs,
        build_quarterly_four_statements,
    )

    st.subheader("FY 季度四表预测（Q1–Q3 实际 + Q4 假设 · 现金 plug）")
    st.caption("AkShare 单季报 + Products/Services 分项 + IS/BS/CF/Equity 四表")

    fetch_result = st.session_state.get("fetch_result")
    ticker = (fetch_result.stock_code if fetch_result else None) or "AAPL"
    fiscal_year = st.number_input(
        "财年（Apple 以 Q4 所在日历年计）",
        min_value=2020,
        max_value=2030,
        value=int(fin_df["year"].max()) if not fin_df.empty else 2024,
        step=1,
        key="q_fiscal_year",
    )

    if st.button("拉取 Q1–Q3 季报", key="btn_fetch_quarterly"):
        with st.spinner(f"正在抓取 {ticker} 单季报…"):
            try:
                st.session_state.quarterly_actuals_df = fetch_us_quarterly_financials(ticker, years=4)
                st.success("季报已加载")
            except Exception as exc:
                st.warning(f"季报抓取失败，将使用 Apple 参考分项 + 年度/4 近似：{exc}")

    q1, q2, q3 = st.columns(3)
    q_pg = q1.number_input("Product 销售增速调整", -0.10, 0.20, -0.01, 0.005, format="%.3f", key="q_pg")
    q_sg = q2.number_input("Services 销售增速调整", -0.10, 0.30, 0.02, 0.005, format="%.3f", key="q_sg")
    q_gm = q3.number_input("目标毛利率", 0.30, 0.60, 0.445, 0.005, format="%.3f", key="q_gm")

    q4, q5, q6 = st.columns(3)
    q_rd = q4.number_input("R&D 费用（绝对值）", 0.0, 1e12, 13.6e9, 1e8, format="%.0f", key="q_rd")
    q_sga = q5.number_input("SG&A 费用（绝对值）", 0.0, 1e12, 13.6e9, 1e8, format="%.0f", key="q_sga")
    q_oi = q6.number_input("Other income", 0.0, 1e11, 250e6, 1e7, format="%.0f", key="q_oi")

    q_tax = st.number_input("Q4 有效税率", 0.0, 0.40, 0.16, 0.005, format="%.3f", key="q_tax")

    if st.button("生成季度四表", type="primary", key="btn_q4_forecast"):
        q_inputs = QuarterlyInputs(
            product_growth=q_pg,
            services_growth=q_sg,
            gross_margin=q_gm,
            rd_expense=q_rd,
            sga_expense=q_sga,
            other_income=q_oi,
            tax_rate=q_tax,
        )
        q_actuals = st.session_state.get("quarterly_actuals_df")
        four = build_quarterly_four_statements(
            fin_df,
            q_actuals,
            q_inputs,
            fiscal_year=int(fiscal_year),
            ticker=ticker if market == "美股" else None,
        )
        st.session_state.quarterly_four_statements = four
        st.session_state.quarterly_forecast_df = four.summary[four.summary["quarter"] == "Q4"]
        st.success("季度四表已生成")

    four = st.session_state.get("quarterly_four_statements")
    if four is not None:
        qdf = four.summary[four.summary["quarter"] == "Q4"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Q4 收入", f"{qdf['revenue'].iloc[0]/1e9:.1f}B")
        m2.metric("Q4 净利润", f"{qdf['net_income'].iloc[0]/1e9:.1f}B")
        m3.metric("Q4 FCFE", f"{qdf['free_cash_flow'].iloc[0]/1e9:.1f}B")
        m4.metric("现金（plug）", f"{qdf['cash'].iloc[0]/1e9:.1f}B")

        t1, t2, t3, t4, t5 = st.tabs(["利润表 IS", "资产负债表 BS", "现金流量表 CF", "权益变动", "Q4 明细"])
        with t1:
            st.dataframe(four.income_statement, use_container_width=True, hide_index=True)
        with t2:
            st.dataframe(four.balance_sheet, use_container_width=True, hide_index=True)
        with t3:
            st.dataframe(four.cash_flow, use_container_width=True, hide_index=True)
        with t4:
            st.dataframe(four.equity_statement, use_container_width=True, hide_index=True)
        with t5:
            st.dataframe(four.summary, use_container_width=True, hide_index=True)


SAMPLE_PEERS = pd.DataFrame(
    {
        "company": ["Peer A", "Peer B", "Peer C"],
        "pe": [12.0, 14.5, 13.2],
        "ps": [2.1, 2.4, 2.2],
        "pb": [1.8, 2.0, 1.9],
    }
)

if page == "总览":
    st.header("项目框架总览")
    st.success("7 个核心模块已就绪，可通过左侧导航查看各模块说明。")
    st.markdown(
        """
| 模块 | 职责 | 输入 | 输出 |
|------|------|------|------|
| `data_loader` | 导入并校验财务数据 | CSV / 上传文件 | 财务 DataFrame |
| `ratio_analysis` | 计算财务比率 | 财务 DataFrame | 比率表 |
| `forecasting` | 预测收入与 FCF | 财务 DataFrame | 预测 DataFrame |
| `cost_of_capital` | 计算 WACC | 市场参数 | WACC |
| `dcf_valuation` | DCF / RI 绝对估值 | 预测 + WACC | 企业/股权价值 |
| `relative_valuation` | 可比公司倍数估值 | 快照 + 可比倍数 | 隐含价值 |
| `football_chart` | 汇总区间与评级 | DCF + 相对估值 | 图表 + 评级 |
"""
    )

    if st.button("运行完整估值流水线 (示例数据)"):
        result = pipeline.run(SAMPLE_FINANCIALS, SAMPLE_PEERS)
        st.subheader("估值结果摘要")
        col1, col2, col3 = st.columns(3)
        col1.metric("DCF 股权价值", f"{result.dcf_result.equity_value:,.0f}")
        col2.metric("相对估值", f"{result.relative_result.implied_equity_value:,.0f}")
        col3.metric("综合评级", result.summary.rating.split("—")[0].strip())
        fig = pipeline.football_chart.plot(result.summary)
        st.plotly_chart(fig, use_container_width=True)

elif page == "1. 数据导入 (data_loader)":
    st.header("数据导入")
    st.caption("支持 A 股（AkShare）与美股（yfinance）自动抓取，失败时降级为手动输入。")

    finlight_key = st.sidebar.text_input(
        "finlight API Key",
        value=os.getenv("FINLIGHT_API_KEY", ""),
        type="password",
        help="在 https://app.finlight.me 注册获取免费 Key（约 5000 次/月）",
    )

    market = st.selectbox(
        "市场选择",
        options=[Market.A_SHARE, Market.US],
        format_func=lambda m: m.value,
        help="A 股使用 AkShare；美股使用 AkShare 东方财富（备用 yfinance）",
    )
    st.session_state.selected_market = market.value
    market_cfg = MARKET_CONFIG[market]

    tab_auto, tab_manual, tab_upload = st.tabs(["自动抓取", "手动输入", "CSV 上传"])

    with tab_auto:
        col_a, col_b = st.columns([2, 1])
        stock_code = col_a.text_input(
            market_cfg["code_label"],
            placeholder=market_cfg["code_placeholder"],
        )
        if col_b.button("开始抓取", type="primary", use_container_width=True):
            if not stock_code.strip():
                st.error("请输入股票代码")
            else:
                with st.spinner(f"正在抓取 {market.value} 财务、新闻与宏观数据…"):
                    result = pipeline.data_loader.auto_fetch(
                        stock_code=stock_code.strip(),
                        market=market,
                        finlight_api_key=finlight_key or None,
                        years=st.session_state.get("financial_years", 5),
                    )
                    st.session_state.fetch_result = result
                    if result.financials is not None:
                        st.session_state.financials_df = result.financials
                    if result.current_price and result.current_price > 0:
                        st.session_state.current_market_price = result.current_price
                        st.session_state.current_market_price_source = result.current_price_source

        result: AutoFetchResult | None = st.session_state.fetch_result
        if result is not None:
            if result.mode == "manual":
                st.warning("部分或全部 API 不可用，已切换为**手动输入模式**，请在「手动输入」或「CSV 上传」补充数据。")
            else:
                st.success("自动抓取完成")

            render_fetch_errors(result)

            if result.industry_summary:
                st.subheader("行业趋势总结")
                st.markdown(result.industry_summary)

            st.subheader("情绪仪表盘")
            if result.sentiment_stats.total > 0:
                render_sentiment_dashboard(result.sentiment_stats)
                with st.expander("新闻列表（摘要 + 情绪）"):
                    for item in result.news[:20]:
                        emoji = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(item.sentiment, "⚪")
                        st.markdown(f"{emoji} **{item.sentiment}** · {item.title}")
                        st.caption(item.summary or "（无摘要）")
            else:
                st.info("暂无新闻情绪数据（可能未配置 finlight Key 或 API 调用失败）")

            st.subheader("关键宏观指标")
            if result.macro:
                render_macro_cards(result.macro)
            else:
                st.info("宏观数据暂不可用")

            if result.reports:
                selected_years = getattr(result.reports, "selected_years", None) or []
                if selected_years:
                    st.subheader(
                        f"三大财务报表（{selected_years[0]}–{selected_years[-1]}，{len(selected_years)} 个完整财年）"
                    )
                else:
                    st.subheader("三大财务报表")
                if result.reports.year_note:
                    st.info(result.reports.year_note)
                r1, r2, r3 = st.tabs(["利润表", "资产负债表", "现金流量表"])
                with r1:
                    st.dataframe(result.reports.income_statement, use_container_width=True)
                with r2:
                    st.dataframe(result.reports.balance_sheet, use_container_width=True)
                with r3:
                    st.dataframe(result.reports.cash_flow, use_container_width=True)

            if result.financials is not None:
                st.subheader("标准化财务数据（供下游模块使用）")
                selected_years = getattr(result.reports, "selected_years", None) if result.reports else None
                if selected_years:
                    st.caption(
                        f"已加载完整财年：{'、'.join(str(y) for y in selected_years)}"
                    )
                st.dataframe(result.financials, use_container_width=True)

                if {"product_revenue", "services_revenue"}.issubset(result.financials.columns):
                    seg_cols = ["year", "product_revenue", "services_revenue", "revenue"]
                    seg_view = result.financials[seg_cols].copy()
                    if market == Market.US:
                        seg_view["product_revenue"] /= 1e9
                        seg_view["services_revenue"] /= 1e9
                        seg_view["revenue"] /= 1e9
                        unit = "十亿美元"
                    else:
                        seg_view["product_revenue"] /= 1e8
                        seg_view["services_revenue"] /= 1e8
                        seg_view["revenue"] /= 1e8
                        unit = "亿元"
                    st.subheader("分业务收入 Products / Services")
                    st.caption(
                        f"来源：yfinance 或 Apple 10-K 参考表（{result.stock_code}）· 单位 {unit}"
                    )
                    st.dataframe(seg_view, use_container_width=True, hide_index=True)

                from modules.ratio_analysis.turnover_diag import diagnose_turnover_fields

                st.caption("周转率字段检查（模块二依赖）")
                st.dataframe(diagnose_turnover_fields(result.financials), use_container_width=True, hide_index=True)

    with tab_manual:
        st.info("当自动抓取失败时，在此手动填写最近 5 年核心财务数据（单位：亿元或保持一致即可）。")
        default_years = list(range(2020, 2025))
        manual_rows = []
        for year in default_years:
            c1, c2, c3, c4, c5 = st.columns(5)
            revenue = c1.number_input(f"{year} 收入", min_value=0.0, value=0.0, key=f"rev_{year}")
            net_income = c2.number_input(f"{year} 净利润", min_value=0.0, value=0.0, key=f"ni_{year}")
            assets = c3.number_input(f"{year} 总资产", min_value=0.0, value=0.0, key=f"ta_{year}")
            equity = c4.number_input(f"{year} 净资产", min_value=0.0, value=0.0, key=f"eq_{year}")
            if revenue > 0 and net_income > 0 and assets > 0 and equity > 0:
                manual_rows.append(
                    {
                        "year": year,
                        "revenue": revenue,
                        "net_income": net_income,
                        "total_assets": assets,
                        "total_equity": equity,
                    }
                )
        if st.button("保存手动数据"):
            if len(manual_rows) < 2:
                st.error("请至少填写 2 个完整年度")
            else:
                try:
                    df = pipeline.data_loader.manual_financials_from_inputs(manual_rows)
                    st.session_state.financials_df = df
                    st.success("手动数据已保存")
                    st.dataframe(df)
                except ValueError as exc:
                    st.error(str(exc))

    with tab_upload:
        st.info(
            "必要字段: year, revenue, net_income, total_assets, total_equity\n\n"
            "周转率可选字段: operating_cost, accounts_receivable, inventory, accounts_payable"
        )
        uploaded = st.file_uploader("上传财务数据 CSV", type=["csv"])
        if uploaded:
            try:
                df = pipeline.data_loader.load_from_upload(uploaded)
                st.session_state.financials_df = df
                st.dataframe(df)
            except ValueError as exc:
                st.error(str(exc))
        elif st.session_state.financials_df is not None:
            st.dataframe(st.session_state.financials_df)
        else:
            st.dataframe(SAMPLE_FINANCIALS)

elif page == "2. 财务比率 (ratio_analysis)":
    st.header("历史体检 — 财务比率分析")
    fin_df, data_source = get_active_financials()
    st.caption(f"当前数据来源：**{data_source}**")

    from modules.ratio_analysis.charts import (
        plot_advanced_margin_trends,
        plot_advanced_turnover_trends,
        plot_dupont_trends,
        plot_margin_trends,
        plot_turnover_trends,
    )
    from modules.ratio_analysis.turnover_diag import (
        diagnose_turnover_fields,
        missing_turnover_fields,
    )

    turnover_diag = diagnose_turnover_fields(fin_df)
    ratios, insights = pipeline.ratio_analyzer.analyze(fin_df)
    _, values, coverage = pipeline.ratio_analyzer.health_check(fin_df)

    # ── 1. 杜邦 ROE ──
    st.subheader("1. ROE 杜邦拆解")
    if "roe" in ratios.columns:
        st.plotly_chart(plot_dupont_trends(ratios), use_container_width=True)
        st.markdown(insights.dupont_summary)
    else:
        st.warning("ROE 数据不足，请确认模块一已提供净利润与股东权益。")

    # ── 2. 利润率 ──
    st.subheader("2. 毛利率 / 净利率 / 研发费用率")
    if {"gross_margin", "net_margin"}.intersection(ratios.columns):
        st.plotly_chart(plot_margin_trends(ratios), use_container_width=True)
        st.markdown(insights.margin_summary)
    else:
        st.warning("利润率数据不足。")

    # ── 3. 周转率 ──
    st.subheader("3. 运营周转率")
    turnover_cols = {
        c for c in ("ar_turnover", "inventory_turnover", "ap_turnover")
        if c in ratios.columns and ratios[c].notna().any()
    }
    if turnover_cols:
        st.plotly_chart(plot_turnover_trends(ratios), use_container_width=True)
        st.markdown(insights.turnover_summary)
        for note in insights.annotations:
            st.warning(note)
    else:
        missing = missing_turnover_fields(fin_df)
        st.warning(
            "周转率数据不足。模块一缺少以下字段或未抓取成功："
            + ("、".join(missing) if missing else "营业成本 / 应收 / 存货 / 应付")
        )
        st.markdown("**字段诊断**")
        st.dataframe(turnover_diag, use_container_width=True, hide_index=True)
        st.info("请回到 **模块 1 → 自动抓取**，输入股票代码重新抓取；或在上传 CSV 中补充上述字段。")

    # ── 4. 高级杜邦（Group 9）──
    st.subheader("4. 高级杜邦 — EBIT / SG&A / NOA 周转")
    adv_margin_cols = {"ebitda_margin", "ebit_margin", "sga_margin"} & set(ratios.columns)
    adv_turn_cols = {"noa_turnover", "fixed_asset_turnover", "asset_turnover"} & set(ratios.columns)
    if adv_margin_cols:
        st.plotly_chart(plot_advanced_margin_trends(ratios), use_container_width=True)
        if insights.advanced_margin_summary:
            st.markdown(insights.advanced_margin_summary)
    if adv_turn_cols:
        st.plotly_chart(plot_advanced_turnover_trends(ratios), use_container_width=True)
        if insights.advanced_turnover_summary:
            st.markdown(insights.advanced_turnover_summary)
    if not adv_margin_cols and not adv_turn_cols:
        st.info("高级杜邦需模块一提供营业利润、现金、应收/存货/应付等完整字段。")

    with st.expander("数据覆盖与完整指标表"):
        st.dataframe(coverage, use_container_width=True, hide_index=True)
        if not values.empty:
            pivot = values.pivot_table(
                index=["category", "metric", "formula"],
                columns="year",
                values="display",
                aggfunc="first",
            )
            st.dataframe(pivot, use_container_width=True)
        st.dataframe(ratios, use_container_width=True)

elif page == "3. 财务预测 (forecasting)":
    st.header("财务预测")

    from modules.forecasting.tax_charts import plot_effective_tax_rate
    from modules.forecasting.tax_rate import STATUTORY_TAX_RATES, TaxRateCalculator

    fin_df, data_source = get_active_financials()
    st.caption(f"当前数据来源：**{data_source}**")

    market = st.session_state.get("selected_market", "A股")
    tax_calc = TaxRateCalculator()
    tax_analysis = tax_calc.analyze(fin_df, market=market)

    # 法定税率参考（角落）
    stat_col1, stat_col2 = st.columns([3, 1])
    with stat_col2:
        st.markdown("**法定税率参考**")
        for label, info in STATUTORY_TAX_RATES.items():
            marker = " ← 当前市场" if label == market else ""
            st.caption(f"{info['country']} {info['rate']:.1%}{marker}")
        st.caption("_仅供参考_")

    with stat_col1:
        st.subheader("税率输入")
        if tax_analysis.warning:
            st.warning(tax_analysis.warning)
        if not tax_analysis.yearly["effective_rate"].notna().any():
            st.info("历史利润表缺少所得税/税前利润字段，请在模块一重新抓取完整报表。")

        st.plotly_chart(
            plot_effective_tax_rate(tax_analysis.yearly, tax_analysis.default_rate),
            use_container_width=True,
        )

        source_label = "近3年历史有效税率均值" if tax_analysis.source == "historical" else "法定税率（降级）"
        tax_rate_input = st.number_input(
            f"预测期适用税率（默认={source_label}）",
            min_value=0.0,
            max_value=0.6,
            value=float(tax_analysis.default_rate),
            step=0.005,
            format="%.3f",
            help="修改后下方预测利润表将自动重算",
        )
        st.session_state.forecast_tax_rate = tax_rate_input

        with st.expander("历史有效税率明细"):
            show = tax_analysis.yearly.copy()
            if "effective_rate" in show.columns:
                show["effective_rate"] = show["effective_rate"].apply(
                    lambda x: f"{x:.2%}" if x is not None and x == x else "—"
                )
            st.dataframe(show, use_container_width=True, hide_index=True)

    from modules.forecasting.session_helpers import forecast_context_key, sync_forecast_context

    fetch_result = st.session_state.get("fetch_result")
    macro = fetch_result.macro if fetch_result is not None else None
    ctx_key = forecast_context_key(market, fin_df, fetch_result)
    if sync_forecast_context(st.session_state, ctx_key, market, macro):
        st.info("已切换股票/财务数据，假设表与 GDP 默认值已重置。")

    tab_annual, tab_quarterly = st.tabs(["6 年年度预测", "单季 Q4 预测（现金 plug）"])

    with tab_annual:
        _render_annual_forecast(
            fin_df,
            market,
            tax_rate_input,
            macro,
            ctx_key,
            pipeline,
        )

    with tab_quarterly:
        _render_quarterly_forecast(fin_df, market, tax_rate_input, pipeline)


elif page == "4. 资本成本 (cost_of_capital)":
    st.header("资本成本 — CAPM & WACC")

    from modules.cost_of_capital.capital_structure import (
        infer_capital_structure,
        infer_cost_of_debt,
    )
    from modules.forecasting.session_helpers import default_gdp_growth

    fin_df, data_source = get_active_financials()
    st.caption(f"资本结构数据来源：**{data_source}**")

    market = st.session_state.get("selected_market", "A股")
    fetch_result = st.session_state.get("fetch_result")
    macro = fetch_result.macro if fetch_result is not None else None
    gdp_cap = float(st.session_state.get("gdp_growth", default_gdp_growth(market, macro)))
    default_terminal = float(
        min(st.session_state.get("terminal_growth", min(0.03, gdp_cap)), gdp_cap)
    )

    cap_struct = infer_capital_structure(fin_df)
    default_kd = infer_cost_of_debt(fin_df)
    default_tax = float(
        st.session_state.get("forecast_tax_rate", 0.25)
    )

    capm_col, sens_col = st.columns([1.2, 1])

    with capm_col:
        st.subheader("CAPM — 股权成本")
        rf = st.number_input(
            "无风险利率 (Rf)",
            min_value=0.0,
            max_value=0.15,
            value=0.04,
            step=0.005,
            format="%.3f",
            help="默认 4%",
        )
        beta = st.number_input("Beta (β)", min_value=0.0, max_value=3.0, value=1.0, step=0.05)
        mrp = st.number_input(
            "市场风险溢价 (Rm - Rf)",
            min_value=0.0,
            max_value=0.20,
            value=0.05,
            step=0.005,
            format="%.3f",
            help="默认 5%",
        )

        coc = pipeline.cost_calculator.calculate(
            risk_free_rate=rf,
            beta=beta,
            market_risk_premium=mrp,
            has_debt=cap_struct.has_debt,
            cost_of_debt=default_kd,
            tax_rate=default_tax,
            equity_weight=cap_struct.equity_weight,
            debt_weight=cap_struct.debt_weight,
        )

        ke = coc.cost_of_equity
        st.metric(
            "权益成本 Cost of Equity (Ke)",
            f"{ke:.2%}",
            help=f"Ke = {rf:.2%} + {beta:.2f} × {mrp:.2%} = {ke:.2%}",
        )
        st.caption(f"**Ke = Rf + β × MRP** → {rf:.2%} + {beta:.2f} × {mrp:.2%} = **{ke:.2%}**")

        if cap_struct.has_debt:
            st.subheader("WACC — 含负债")
            st.caption(
                f"最近一期：有息负债 {cap_struct.total_debt:,.0f} · "
                f"股东权益 {cap_struct.total_equity:,.0f} · "
                f"E/V **{cap_struct.equity_weight:.1%}** · D/V **{cap_struct.debt_weight:.1%}**"
            )
            kd = st.number_input(
                "负债成本 (Kd，税前)",
                min_value=0.0,
                max_value=0.20,
                value=float(default_kd),
                step=0.005,
                format="%.3f",
            )
            wacc_tax = st.number_input(
                "WACC 适用税率",
                min_value=0.0,
                max_value=0.6,
                value=default_tax,
                step=0.005,
                format="%.3f",
                help="用于 Kd × (1-T) 税盾；默认取自模块三预测税率",
            )
            coc = pipeline.cost_calculator.calculate(
                risk_free_rate=rf,
                beta=beta,
                market_risk_premium=mrp,
                has_debt=True,
                cost_of_debt=kd,
                tax_rate=wacc_tax,
                equity_weight=cap_struct.equity_weight,
                debt_weight=cap_struct.debt_weight,
            )
            st.metric("WACC", f"{coc.wacc:.2%}")
            st.caption(
                f"**WACC** = {cap_struct.equity_weight:.1%}×{coc.cost_of_equity:.2%} + "
                f"{cap_struct.debt_weight:.1%}×{kd:.2%}×(1−{wacc_tax:.1%}) = **{coc.wacc:.2%}**"
            )
        else:
            st.info("当前财报未检测到有息负债，WACC = Ke（全权益）。")
            st.metric("WACC", f"{coc.wacc:.2%}")

        st.session_state.cost_of_capital = coc
        st.session_state.net_debt = cap_struct.net_debt

    with sens_col:
        st.subheader("敏感性分析参数")
        st.caption("调整以下滑块，模块五 DCF 将使用此处设定进行估值。")

        from modules.cost_of_capital.sensitivity import render_sensitivity_sliders

        discount_rate, terminal_growth = render_sensitivity_sliders(
            base_wacc=coc.wacc,
            gdp_cap=gdp_cap,
            default_terminal=default_terminal,
            session_state=st.session_state,
        )

        st.markdown("---")
        st.markdown("**当前敏感性设定**")
        st.write(f"- 折现率：**{discount_rate:.2%}**（WACC 基准 {coc.wacc:.2%}）")
        st.write(f"- 终值增长率：**{terminal_growth:.2%}**")
        if discount_rate <= terminal_growth:
            st.warning("折现率必须大于终值增长率，否则 DCF 终值公式无意义。请调高折现率或调低终值增长率。")

        forecast_df = st.session_state.get("forecast_df")
        if forecast_df is not None and discount_rate > terminal_growth:
            try:
                preview = pipeline.dcf_valuator.value_dcf(
                    forecast_df,
                    coc,
                    terminal_growth=terminal_growth,
                    net_debt=cap_struct.net_debt,
                    max_terminal_growth=gdp_cap,
                    discount_rate=discount_rate,
                )
                st.metric("快速预览 — 股权价值", f"{preview.equity_value:,.0f}")
                st.caption("基于模块三预测与当前滑块设定")
            except ValueError as exc:
                st.warning(str(exc))
        else:
            st.caption("完成模块三财务预测后，此处将显示 DCF 快速预览。")

elif page == "5. 绝对估值 (dcf_valuation)":
    st.header("绝对估值 — FTE & RI")

    from modules.cost_of_capital.calculator import CostOfCapitalCalculator
    from modules.dcf_valuation.sensitivity_chart import plot_sensitivity_heatmap
    from modules.dcf_valuation.shares import default_shares_input_value
    from modules.forecasting.session_helpers import default_gdp_growth, forecast_for_display, forecast_table_column_config

    market = st.session_state.get("selected_market", "A股")
    fetch_result = st.session_state.get("fetch_result")
    macro = fetch_result.macro if fetch_result is not None else None
    default_gdp = default_gdp_growth(market, macro)
    gdp_cap = float(st.session_state.get("gdp_growth", default_gdp))
    terminal_g = float(
        st.session_state.get(
            "sensitivity_terminal_growth",
            st.session_state.get("terminal_growth", min(0.03, gdp_cap)),
        )
    )

    fin_df_hist, data_source = get_active_financials()
    forecast_df = st.session_state.get("forecast_df")
    if forecast_df is None:
        forecast_df = pipeline.forecaster.forecast(fin_df_hist)
        st.info("未在模块三生成预测，已基于当前财务数据自动生成。建议先在「财务预测」模块编辑假设。")

    coc = st.session_state.get("cost_of_capital")
    if coc is None:
        coc = CostOfCapitalCalculator().calculate()
        st.info("未在模块四设定资本成本，使用 CAPM 默认值（Rf 4%、β 1、MRP 5%）。")

    discount_rate = float(st.session_state.get("discount_rate", coc.wacc))
    book_value = float(fin_df_hist.sort_values("year").iloc[-1]["total_equity"])

    stock_code = fetch_result.stock_code if fetch_result else None
    default_shares = default_shares_input_value(
        st.session_state.get("shares_outstanding"),
        stock_code,
        market,
    )

    st.caption(
        f"折现率 **{discount_rate:.2%}** · 终值增长率 **{terminal_g:.2%}** · "
        f"GDP 上限 {gdp_cap:.1%} · 数据来源：{data_source}"
    )

    shares_col, bv_col = st.columns(2)
    shares_outstanding = shares_col.number_input(
        "总股本（股）",
        min_value=1.0,
        value=float(default_shares),
        step=1e8,
        format="%.0f",
        help="用于计算每股价值；A 股尝试自动抓取总股本",
    )
    st.session_state.shares_outstanding = shares_outstanding
    bv_col.metric("最近一期股东权益（RI 账面价值）", f"{book_value:,.0f}")

    valuator = pipeline.dcf_valuator
    currency = "元" if market == "A股" else "USD"

    st.subheader("1. 股权自由现金流（FTE）模型")
    st.caption("预测 FCF 按模块四折现率折现 + 戈登终值 → 股权价值 → 每股价值")
    try:
        fte = valuator.value_fte(
            forecast_df,
            cost_of_equity=discount_rate,
            terminal_growth=terminal_g,
            shares_outstanding=shares_outstanding,
            max_terminal_growth=gdp_cap,
        )
        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("股权价值", f"{fte.equity_value:,.0f}")
        fc2.metric("每股价值", f"{fte.per_share_value:,.2f} {currency}" if fte.per_share_value else "—")
        fc3.metric("终值现值占比", f"{fte.pv_terminal / fte.equity_value:.1%}" if fte.equity_value else "—")
        st.caption(
            f"显性期 FCF 现值 {fte.pv_explicit:,.0f} · 终值现值 {fte.pv_terminal:,.0f} · "
            f"Ke={discount_rate:.2%} · g={terminal_g:.2%}"
        )
    except ValueError as exc:
        st.error(f"FTE 估值失败：{exc}")
        fte = None

    st.subheader("2. 剩余收益（RI）模型")
    st.caption("股权价值 = 期初账面价值 + 超额收益现值 + 终值现值")
    try:
        ri = valuator.value_ri(
            book_value,
            forecast_df,
            cost_of_equity=discount_rate,
            terminal_growth=terminal_g,
            shares_outstanding=shares_outstanding,
            max_terminal_growth=gdp_cap,
        )
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("股权价值", f"{ri.equity_value:,.0f}")
        rc2.metric("每股价值", f"{ri.per_share_value:,.2f} {currency}" if ri.per_share_value else "—")
        rc3.metric("超额收益现值", f"{ri.pv_explicit:,.0f}")
        st.caption(
            f"账面价值 {book_value:,.0f} · 终值现值 {ri.pv_terminal:,.0f} · "
            f"Ke={discount_rate:.2%} · g={terminal_g:.2%}"
        )
    except ValueError as exc:
        st.error(f"RI 估值失败：{exc}")
        ri = None

    st.subheader("3. 敏感性分析 — 5×5 热力图")
    ke_grid = valuator.group9_rate_grid(discount_rate, spread=0.01)
    tg_grid = valuator.group9_terminal_grid(terminal_g, gdp_cap, spread=0.01)
    st.caption(
        f"纵轴 Ke：{ke_grid[0]:.1%} – {ke_grid[-1]:.1%}（步长 0.5pct）· "
        f"横轴 g：{tg_grid[0]:.1%} – {tg_grid[-1]:.1%} · Football Field 取矩阵 min/max"
    )

    sens_matrix = valuator.sensitivity_matrix(
        forecast_df,
        book_value=book_value,
        base_discount=discount_rate,
        base_terminal=terminal_g,
        shares_outstanding=shares_outstanding,
        method="FTE",
        max_terminal_growth=gdp_cap,
        grid_size=5,
    )

    if fte and fte.per_share_value:
        base_label = f"{fte.per_share_value:,.2f}"
        st.session_state.fte_target_price = fte.per_share_value
        mkt = float(st.session_state.get("current_market_price", 0) or 0)
        upside = (fte.per_share_value / mkt - 1) if mkt > 0 else None
        tp_col1, tp_col2 = st.columns(2)
        tp_col1.metric("目标价（FTE 基准）", f"{base_label} {currency}", help="以 FTE 基准情景作为主要目标价")
        if upside is not None:
            tp_col2.metric("相对市价潜在空间", f"{upside:+.2%}")
        st.info(f"基准情景（Ke={discount_rate:.1%} × g={terminal_g:.1%}）FTE 每股：**{base_label} {currency}**")

    st.plotly_chart(
        plot_sensitivity_heatmap(
            sens_matrix,
            title=f"FTE 每股价值敏感性（{currency}/股）",
            color_label=f"每股价值 ({currency})",
        ),
        use_container_width=True,
    )
    st.dataframe(
        sens_matrix.style.format("{:,.2f}", na_rep="—"),
        use_container_width=True,
    )

    ri_sens_matrix = valuator.sensitivity_matrix(
        forecast_df,
        book_value=book_value,
        base_discount=discount_rate,
        base_terminal=terminal_g,
        shares_outstanding=shares_outstanding,
        method="RI",
        max_terminal_growth=gdp_cap,
        grid_size=5,
    )
    st.session_state.fte_sensitivity_matrix = sens_matrix
    st.session_state.ri_sensitivity_matrix = ri_sens_matrix
    if fte:
        st.session_state.dcf_fte = fte
    if ri:
        st.session_state.dcf_ri = ri

    with st.expander("RI 敏感性矩阵"):
        st.dataframe(
            ri_sens_matrix.style.format("{:,.2f}", na_rep="—"),
            use_container_width=True,
        )

    with st.expander("预测数据明细（模块三）"):
        st.dataframe(
            forecast_for_display(forecast_df, market),
            column_config=forecast_table_column_config(market),
            use_container_width=True,
            hide_index=True,
        )

elif page == "6. 相对估值 (relative_valuation)":
    st.header("相对估值 — 可比公司法")

    from modules.cost_of_capital.capital_structure import infer_capital_structure
    from modules.dcf_valuation.shares import default_shares_input_value
    from modules.forecasting.session_helpers import forecast_amount_scale
    from modules.relative_valuation.peers import (
        compute_peer_multiples,
        empty_peer_table,
        summarize_multiples,
        target_ltm_ntm,
        valid_peers,
    )

    market = st.session_state.get("selected_market", "A股")
    fetch_result = st.session_state.get("fetch_result")
    stock_code = fetch_result.stock_code if fetch_result else None
    scale, unit = forecast_amount_scale(market)
    currency = "元" if market == "A股" else "USD"

    # 切换目标公司时清空推荐缓存
    peer_ctx = f"{market}:{stock_code or 'manual'}"
    if st.session_state.get("peer_recommend_ctx") != peer_ctx:
        st.session_state.pop("recommended_peers_df", None)
        st.session_state.pop("target_industry_info", None)
        st.session_state.peer_recommend_ctx = peer_ctx

    fin_df, data_source = get_active_financials()
    forecast_df = st.session_state.get("forecast_df")
    (ltm_ebitda_raw, ltm_ni_raw), (ntm_ebitda_raw, ntm_ni_raw) = target_ltm_ntm(fin_df, forecast_df)
    cap_struct = infer_capital_structure(fin_df)
    net_debt = cap_struct.net_debt

    default_shares = default_shares_input_value(
        st.session_state.get("shares_outstanding"),
        stock_code,
        market,
    )

    st.caption(
        f"目标公司数据来源：**{data_source}** · 金额单位：**{unit}**（与下表输入保持一致）"
    )

    st.caption("LTM = 历史最近一期 · NTM = 预测第一期")

    target_col1, target_col2, target_col3 = st.columns(3)
    target_ebitda = target_col1.number_input(
        f"LTM EBITDA ({unit})",
        min_value=0.0,
        value=float(ltm_ebitda_raw / scale),
        step=max(ltm_ebitda_raw / scale * 0.01, 0.01),
        format="%.4f",
    )
    target_net_income = target_col2.number_input(
        f"LTM 净利润 ({unit})",
        min_value=0.0,
        value=float(ltm_ni_raw / scale),
        step=max(ltm_ni_raw / scale * 0.01, 0.01),
        format="%.4f",
    )
    shares_outstanding = target_col3.number_input(
        "总股本（股）",
        min_value=1.0,
        value=float(default_shares),
        step=1e8,
        format="%.0f",
    )
    st.session_state.shares_outstanding = shares_outstanding

    ntm_col1, ntm_col2 = st.columns(2)
    target_ebitda_ntm = ntm_col1.number_input(
        f"NTM EBITDA ({unit})",
        min_value=0.0,
        value=float(ntm_ebitda_raw / scale),
        step=max(ntm_ebitda_raw / scale * 0.01, 0.01),
        format="%.4f",
    )
    target_net_income_ntm = ntm_col2.number_input(
        f"NTM 净利润 ({unit})",
        min_value=0.0,
        value=float(ntm_ni_raw / scale),
        step=max(ntm_ni_raw / scale * 0.01, 0.01),
        format="%.4f",
    )

    if net_debt > 0:
        st.caption(f"EV/EBITDA 估值将扣除净负债 **{net_debt / scale:,.2f} {unit}** 换算为股权价值")

    # ── 自动推荐同行 ──
    st.subheader("自动推荐同行")
    from modules.relative_valuation.peer_recommender import PeerRecommender

    peer_recommender = PeerRecommender(pipeline.data_loader)
    cached_industry = None
    if fetch_result and getattr(fetch_result, "industry", None):
        cached_industry = fetch_result.industry
        st.caption(f"模块一已缓存行业：**{cached_industry}**（抓取时识别，可加速推荐）")

    if not stock_code:
        st.warning("请先在 **模块一** 输入目标公司并点击「开始抓取」，再使用自动推荐。")
    else:
        st.caption(f"目标：**{market} {stock_code}** · 将识别行业并拉取同行业公司列表")

        col_rec_a, col_rec_b, col_rec_c = st.columns(3)
        with col_rec_a:
            auto_recommend = st.button("自动推荐同行", type="primary", key="btn_recommend_peers")
        with col_rec_b:
            preset_recommend = st.button("使用内置模板（离线）", key="btn_preset_peers")
        with col_rec_c:
            apple_preset = (
                market == "美股"
                and str(stock_code).strip().upper() in ("AAPL", "APPLE")
                and st.button("Apple 标准可比模板", key="btn_apple_group9_peers")
            )

        peer_lightweight = st.checkbox(
            "轻量模式（优先 yfinance/行情倍数，不逐家拉财报）",
            value=True,
            key="peer_lightweight_mode",
            help="接口限流时建议开启；完整模式会先拉财报，失败再自动降级为轻量估算。",
        )

        def _apply_recommendation(industry_info, rec_df, success_prefix: str) -> None:
            st.session_state.target_industry_info = industry_info
            st.session_state.recommended_peers_df = rec_df
            st.session_state.pop("peer_pool_failures", None)
            st.success(
                f"{success_prefix}已识别行业：**{industry_info.name}**（来源：{industry_info.source}），"
                f"推荐 {len(rec_df)} 家同行"
            )

        if auto_recommend:
            with st.spinner("正在识别行业并拉取同行列表…"):
                try:
                    industry_info, rec_df = peer_recommender.recommend(
                        stock_code,
                        market,
                        cached_industry=cached_industry,
                    )
                    _apply_recommendation(industry_info, rec_df, "")
                except ValueError as exc:
                    st.error(str(exc))
                    st.session_state.show_manual_industry = True
                except Exception as exc:
                    st.error(f"推荐失败：{exc}")
                    st.info(
                        "在线接口可能限流。请点 **「使用内置模板（离线）」** 或下方手动选择行业。"
                    )
                    st.session_state.show_manual_industry = True

        if preset_recommend:
            with st.spinner("加载内置同行模板…"):
                try:
                    industry_info, rec_df = peer_recommender.recommend_preset(
                        stock_code,
                        market,
                        cached_industry=cached_industry,
                    )
                    _apply_recommendation(industry_info, rec_df, "【离线】")
                except Exception as exc:
                    st.error(f"内置模板加载失败：{exc}")

        if apple_preset:
            with st.spinner("加载 Apple 标准可比公司…"):
                try:
                    industry_info, rec_df = peer_recommender.recommend_preset(
                        stock_code,
                        market,
                        preset="apple_group9",
                    )
                    _apply_recommendation(industry_info, rec_df, "【标准模板】")
                except Exception as exc:
                    st.error(f"模板加载失败：{exc}")

        if market == "A股" and st.session_state.get("show_manual_industry"):
            with st.expander("手动选择行业（接口失败时使用）", expanded=True):
                try:
                    if "a_share_industry_list" not in st.session_state:
                        with st.spinner("加载行业板块列表…"):
                            st.session_state.a_share_industry_list = (
                                peer_recommender.list_a_share_industries()
                            )
                    manual_ind = st.selectbox(
                        "行业板块",
                        options=st.session_state.a_share_industry_list,
                        key="manual_industry_select",
                    )
                    mcol1, mcol2 = st.columns(2)
                    with mcol1:
                        manual_online = st.button("按所选行业推荐（在线）", key="btn_manual_recommend")
                    with mcol2:
                        manual_preset = st.button("按所选行业推荐（内置模板）", key="btn_manual_preset")
                    if manual_online:
                        with st.spinner("拉取行业成份股…"):
                            industry_info, rec_df = peer_recommender.recommend(
                                stock_code,
                                market,
                                industry_override=manual_ind,
                            )
                            st.session_state.target_industry_info = industry_info
                            st.session_state.recommended_peers_df = rec_df
                            st.session_state.show_manual_industry = False
                            st.success(f"已加载 **{manual_ind}** 成份股 {len(rec_df)} 家")
                            st.rerun()
                    if manual_preset:
                        with st.spinner("加载内置模板…"):
                            industry_info, rec_df = peer_recommender.recommend_preset(
                                stock_code,
                                market,
                                industry_override=manual_ind,
                            )
                            st.session_state.target_industry_info = industry_info
                            st.session_state.recommended_peers_df = rec_df
                            st.session_state.show_manual_industry = False
                            st.success(f"【离线】已加载 **{manual_ind}** 模板 {len(rec_df)} 家")
                            st.rerun()
                except Exception as exc:
                    st.error(f"手动推荐失败：{exc}")

        if market == "美股" and st.session_state.get("show_manual_industry"):
            from modules.relative_valuation.peer_presets import US_PEERS_BY_CATEGORY

            with st.expander("手动选择美股分类（接口失败时使用）", expanded=True):
                us_cat = st.selectbox(
                    "行业分类",
                    options=list(US_PEERS_BY_CATEGORY.keys()),
                    key="manual_us_category",
                )
                if st.button("按所选分类加载内置模板", key="btn_us_preset_category"):
                    with st.spinner("加载内置模板…"):
                        industry_info, rec_df = peer_recommender.recommend_preset(
                            stock_code,
                            market,
                            industry_override=f"{us_cat}（{us_cat}）",
                        )
                        st.session_state.target_industry_info = industry_info
                        st.session_state.recommended_peers_df = rec_df
                        st.session_state.show_manual_industry = False
                        st.success(f"【离线】已加载 **{us_cat}** 模板 {len(rec_df)} 家")
                        st.rerun()

    if st.session_state.get("recommended_peers_df") is not None:
        ind_info = st.session_state.get("target_industry_info")
        if ind_info:
            st.info(f"行业：**{ind_info.name}** · 数据来源：{ind_info.source}")

        st.caption("勾选 5–8 家同行（默认已选市值靠前的 6 家），点击「确认选择」写入可比公司池。")
        rec_col_config = {
            "selected": st.column_config.CheckboxColumn("选择", default=False),
            "code": st.column_config.TextColumn("代码", disabled=True),
            "name": st.column_config.TextColumn("公司名称", disabled=True),
        }
        if "pe_dynamic" in st.session_state.recommended_peers_df.columns:
            rec_col_config["pe_dynamic"] = st.column_config.NumberColumn(
                "市盈率(参考)", disabled=True, format="%.2f"
            )
        rec_edited = st.data_editor(
            st.session_state.recommended_peers_df,
            num_rows="fixed",
            column_config=rec_col_config,
            use_container_width=True,
            key="recommended_peers_editor",
        )
        st.session_state.recommended_peers_df = rec_edited

        selected_count = int(rec_edited["selected"].sum())
        st.caption(f"已勾选 **{selected_count}** 家（建议 5–8 家，最多 10 家进入可比池）")

        if st.button("确认选择", key="btn_confirm_peers"):
            picked = rec_edited[rec_edited["selected"].astype(bool)].copy()
            if len(picked) < 1:
                st.error("请至少勾选 1 家同行")
            elif len(picked) > 10:
                st.error("最多选择 10 家同行")
            else:
                with st.spinner(
                    f"正在拉取 {len(picked)} 家同行"
                    f"{'（轻量模式）' if peer_lightweight else ''}…"
                ):
                    try:
                        pool_result = peer_recommender.build_peer_pool(
                            picked,
                            market,
                            scale,
                            lightweight=peer_lightweight,
                        )
                        st.session_state.peer_df = pool_result.peers
                        st.session_state.peer_pool_failures = pool_result.failures
                        if pool_result.failures:
                            st.warning(
                                f"成功 **{len(pool_result.peers)}/{len(picked)}** 家；"
                                f"失败 {len(pool_result.failures)} 家（详见下方）"
                            )
                        else:
                            st.success(f"已将 {len(pool_result.peers)} 家同行写入可比公司池")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.error(f"拉取失败：{exc}")

    st.subheader("可比公司池 / 手动输入（最多 10 家）")
    st.caption("自动推荐确认后会填入下表；也可手动编辑。LTM 必填；NTM 可选（留空则沿用 LTM）。")

    if st.session_state.get("peer_pool_failures"):
        with st.expander(
            f"⚠ 上次拉取失败 {len(st.session_state.peer_pool_failures)} 家",
            expanded=True,
        ):
            st.caption("失败原因多为接口限流或该公司财报缺失，可稍后重试或手动补录下表。")
            for msg in st.session_state.peer_pool_failures:
                st.write(f"- {msg}")

    if "peer_df" not in st.session_state:
        st.session_state.peer_df = empty_peer_table(3)

    edited_peers = st.data_editor(
        st.session_state.peer_df,
        num_rows="dynamic",
        column_config={
            "company": st.column_config.TextColumn("公司名", required=True),
            "market_cap": st.column_config.NumberColumn(
                f"市值 ({unit})", min_value=0.0, format="%.4f"
            ),
            "ebitda": st.column_config.NumberColumn(
                f"EBITDA ({unit})", min_value=0.0, format="%.4f"
            ),
            "net_income": st.column_config.NumberColumn(
                f"LTM 净利润 ({unit})", min_value=0.0, format="%.4f"
            ),
            "ebitda_ntm": st.column_config.NumberColumn(
                f"NTM EBITDA ({unit})", min_value=0.0, format="%.4f"
            ),
            "net_income_ntm": st.column_config.NumberColumn(
                f"NTM 净利润 ({unit})", min_value=0.0, format="%.4f"
            ),
        },
        use_container_width=True,
        key="peer_companies_editor",
    )
    if len(edited_peers) > 10:
        st.warning("最多支持 10 家可比公司，仅前 10 家有效行将参与计算。")
    st.session_state.peer_df = edited_peers.head(10)

    peers_valid = valid_peers(st.session_state.peer_df)
    if not peers_valid.empty:
        with st.expander("同行倍数明细（自动计算）", expanded=True):
            peer_calc = compute_peer_multiples(st.session_state.peer_df)
            display = peer_calc.copy()
            display["ev_ebitda"] = display["ev_ebitda"].map(lambda x: f"{x:.2f}x")
            display["pe"] = display["pe"].map(lambda x: f"{x:.2f}x")
            display["ev_ebitda_ntm"] = display["ev_ebitda_ntm"].map(lambda x: f"{x:.2f}x")
            display["pe_ntm"] = display["pe_ntm"].map(lambda x: f"{x:.2f}x")
            st.dataframe(display, use_container_width=True, hide_index=True)

        try:
            summary = summarize_multiples(st.session_state.peer_df)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("LTM EV/EBITDA 中位", f"{summary.ev_ebitda_median:.2f}x")
            m2.metric("NTM EV/EBITDA 中位", f"{summary.ev_ebitda_ntm_median:.2f}x")
            m3.metric("LTM P/E 中位", f"{summary.pe_median:.2f}x")
            m4.metric("NTM P/E 中位", f"{summary.pe_ntm_median:.2f}x")
            st.caption(f"基于 **{summary.peer_count}** 家有效可比公司 · P75 用于 Football Field 上沿")

            target_ebitda_abs = target_ebitda * scale
            target_ni_abs = target_net_income * scale
            target_ebitda_ntm_abs = target_ebitda_ntm * scale
            target_ni_ntm_abs = target_net_income_ntm * scale

            result = pipeline.relative_valuator.value_from_peers(
                st.session_state.peer_df,
                target_ebitda=target_ebitda_abs,
                target_net_income=target_ni_abs,
                shares_outstanding=shares_outstanding,
                net_debt=net_debt,
                target_ebitda_ntm=target_ebitda_ntm_abs,
                target_net_income_ntm=target_ni_ntm_abs,
            )
            st.session_state.relative_valuation = pipeline.relative_valuator.to_legacy_result(result)
            st.session_state.peer_implied_prices = result

            st.subheader("基于市场行情的估算股价（LTM / NTM）")
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                st.metric("LTM EV/EBITDA", f"{result.price_ev_ebitda:,.2f} {currency}")
                if result.price_ev_ebitda_p75:
                    st.caption(f"P75：{result.price_ev_ebitda_p75:,.2f}")
            with p2:
                st.metric("NTM EV/EBITDA", f"{result.price_ev_ebitda_ntm:,.2f} {currency}" if result.price_ev_ebitda_ntm else "—")
                if result.price_ev_ebitda_ntm_p75:
                    st.caption(f"P75：{result.price_ev_ebitda_ntm_p75:,.2f}")
            with p3:
                st.metric("LTM P/E", f"{result.price_pe:,.2f} {currency}")
                if result.price_pe_p75:
                    st.caption(f"P75：{result.price_pe_p75:,.2f}")
            with p4:
                st.metric("NTM P/E", f"{result.price_pe_ntm:,.2f} {currency}" if result.price_pe_ntm else "—")
                if result.price_pe_ntm_p75:
                    st.caption(f"P75：{result.price_pe_ntm_p75:,.2f}")
        except ValueError as exc:
            st.error(str(exc))
    else:
        st.info("请在上方表格至少填写 1 家可比公司（公司名、市值、EBITDA、净利润均须 > 0）。")

        with st.expander("示例：白酒行业可比（演示数据，单位：亿元）"):
            st.dataframe(
                pd.DataFrame(
                    {
                        "company": ["山西汾酒", "泸州老窖", "五粮液"],
                        "market_cap": [3200, 2100, 5800],
                        "ebitda": [120, 180, 380],
                        "net_income": [95, 130, 300],
                    }
                ),
                hide_index=True,
            )

elif page == "7. 图表与评级 (football_chart)":
    st.header("Football Field — 估值汇总与评级")

    from modules.football_chart.market_price import fetch_current_price_with_meta
    from modules.football_chart.summary_builder import (
        build_football_summary,
        matrix_price_range,
        relative_price_ranges,
    )

    market = st.session_state.get("selected_market", "A股")
    fetch_result = st.session_state.get("fetch_result")
    stock_code = fetch_result.stock_code if fetch_result else None
    currency = "元" if market == "A股" else "USD"

    fte_range = matrix_price_range(
        st.session_state.get("fte_sensitivity_matrix"),
        fallback_price=getattr(st.session_state.get("dcf_fte"), "per_share_value", None),
    )
    ri_range = matrix_price_range(
        st.session_state.get("ri_sensitivity_matrix"),
        fallback_price=getattr(st.session_state.get("dcf_ri"), "per_share_value", None),
    )

    peer_prices = st.session_state.get("peer_implied_prices")
    ev_ltm_range = ev_ntm_range = pe_ltm_range = pe_ntm_range = None
    if peer_prices is not None:
        ev_ltm_range = relative_price_ranges(
            getattr(peer_prices, "price_ev_ebitda", None),
            getattr(peer_prices, "price_ev_ebitda_p75", None),
        )
        ev_ntm_range = relative_price_ranges(
            getattr(peer_prices, "price_ev_ebitda_ntm", None),
            getattr(peer_prices, "price_ev_ebitda_ntm_p75", None),
        )
        pe_ltm_range = relative_price_ranges(
            getattr(peer_prices, "price_pe", None),
            getattr(peer_prices, "price_pe_p75", None),
        )
        pe_ntm_range = relative_price_ranges(
            getattr(peer_prices, "price_pe_ntm", None),
            getattr(peer_prices, "price_pe_ntm_p75", None),
        )

    missing = []
    if not fte_range:
        missing.append("模块五 FTE 敏感性矩阵")
    if not ri_range:
        missing.append("模块五 RI 敏感性矩阵")
    if not ev_ltm_range:
        missing.append("模块六 LTM EV/EBITDA")
    if not ev_ntm_range:
        missing.append("模块六 NTM EV/EBITDA")
    if not pe_ltm_range:
        missing.append("模块六 LTM P/E")
    if not pe_ntm_range:
        missing.append("模块六 NTM P/E")
    if missing:
        st.warning("以下数据尚未就绪，请按顺序完成对应模块：" + "、".join(missing))

    cached_price = None
    cached_source = ""
    if fetch_result and getattr(fetch_result, "current_price", None):
        cached_price = fetch_result.current_price
        cached_source = getattr(fetch_result, "current_price_source", "") or "模块一缓存"
    elif st.session_state.get("current_market_price", 0) > 0:
        cached_price = st.session_state.current_market_price
        cached_source = st.session_state.get("current_market_price_source", "会话缓存")

    price_result = fetch_current_price_with_meta(
        stock_code,
        market,
        cached_price=cached_price if cached_price and cached_price > 0 else None,
    )
    if price_result.price <= 0 and cached_price and cached_price > 0:
        price_result = type(price_result)(
            price=cached_price,
            source=cached_source,
            is_stale=True,
        )

    default_price = price_result.price if price_result.price > 0 else float(
        st.session_state.get("current_market_price", 0) or 0
    )
    price_col1, price_col2, price_col3 = st.columns([3, 1, 1])
    with price_col2:
        refresh_price = st.button("重新拉取市价", use_container_width=True)
    with price_col3:
        use_reference = st.button("使用参考价", use_container_width=True, help="接口限流时填入内置参考价，可再手动修正")

    if refresh_price:
        live_result = fetch_current_price_with_meta(
            stock_code,
            market,
            cached_price=None,
            allow_reference=False,
            allow_stale_cache=True,
        )
        if live_result.price > 0:
            st.session_state.current_market_price = live_result.price
            st.session_state.current_market_price_source = live_result.source
            st.session_state.football_current_price_input = live_result.price
            default_price = live_result.price
            price_result = live_result
            suffix = "（历史缓存）" if live_result.is_stale else ""
            st.success(f"已更新市价：**{live_result.price:,.2f}** {currency}（{live_result.source}{suffix}）")
        else:
            prev = float(st.session_state.get("current_market_price", 0) or 0)
            if prev > 0:
                st.warning(
                    "在线行情暂不可用，已保留当前输入的市价。"
                    + (f" 详情：{live_result.error}" if live_result.error else "")
                )
            else:
                ref_result = fetch_current_price_with_meta(
                    stock_code, market, cached_price=None, allow_reference=True
                )
                if ref_result.price > 0:
                    st.session_state.current_market_price = ref_result.price
                    st.session_state.current_market_price_source = ref_result.source
                    st.session_state.football_current_price_input = ref_result.price
                    default_price = ref_result.price
                    price_result = ref_result
                    st.warning(f"在线拉取失败，已填入参考价 **{ref_result.price:,.2f}** {currency}（请核对后使用）")
                else:
                    st.error(
                        "行情拉取失败，请手动输入或使用「使用参考价」。"
                        + (f" 详情：{live_result.error}" if live_result.error else "")
                    )

    if use_reference and stock_code:
        ref_result = fetch_current_price_with_meta(
            stock_code, market, cached_price=None, allow_reference=True, allow_stale_cache=False
        )
        if ref_result.price > 0:
            st.session_state.current_market_price = ref_result.price
            st.session_state.current_market_price_source = ref_result.source
            st.session_state.football_current_price_input = ref_result.price
            default_price = ref_result.price
            price_result = ref_result
            st.info(f"已填入参考价 **{ref_result.price:,.2f}** {currency}（{ref_result.source}）")
        else:
            st.error("暂无该股票的参考价，请手动输入。")

    with price_col1:
        if default_price > 0:
            stale_hint = " · ⚠ 参考/缓存数据，建议核对" if price_result.is_stale else ""
            st.caption(
                f"自动获取市价：**{default_price:,.2f}** {currency}（来源：{price_result.source}）{stale_hint}"
            )
        elif fetch_result is None:
            st.caption("未检测到模块一抓取结果，请在下方手动输入市价，或点击「使用参考价」。")
        else:
            detail = f"（{price_result.error}）" if price_result.error else ""
            st.caption(f"行情接口暂不可用，请手动输入或使用参考价。{detail}")

    session_default = float(st.session_state.get("current_market_price", 0) or 0)
    input_default = session_default if session_default > 0 else (default_price or 0.0)
    current_price = st.number_input(
        f"当前市场价格 ({currency}/股)",
        min_value=0.0,
        value=input_default,
        step=0.01,
        format="%.2f",
        help="优先使用模块一/磁盘缓存；拉取失败时可使用参考价或手动修正",
        key="football_current_price_input",
    )
    st.session_state.current_market_price = current_price
    if current_price > 0 and price_result.source:
        st.session_state.current_market_price_source = price_result.source

    fte_target = st.session_state.get("fte_target_price") or getattr(
        st.session_state.get("dcf_fte"), "per_share_value", None
    )
    if fte_target and fte_target > 0 and st.session_state.get("current_market_price", 0) > 0:
        mkt = st.session_state.current_market_price
        st.metric(
            "目标价（FTE 基准 · 主目标）",
            f"{fte_target:,.2f} {currency}",
            delta=f"{fte_target / mkt - 1:+.2%}",
            delta_color="inverse",
        )

    if (
        fte_range
        and ri_range
        and ev_ltm_range
        and ev_ntm_range
        and pe_ltm_range
        and pe_ntm_range
        and current_price > 0
    ):
        try:
            ff_summary = build_football_summary(
                fte_range=fte_range,
                ri_range=ri_range,
                ev_ebitda_ltm_range=ev_ltm_range,
                ev_ebitda_ntm_range=ev_ntm_range,
                pe_ltm_range=pe_ltm_range,
                pe_ntm_range=pe_ntm_range,
                current_price=current_price,
                currency=currency,
                fte_target_price=fte_target,
            )
            st.session_state.football_summary = ff_summary

            if ff_summary.rating == "买入（低估）":
                st.success(f"综合评级：**{ff_summary.rating}**")
            elif ff_summary.rating == "卖出（高估）":
                st.error(f"综合评级：**{ff_summary.rating}**")
            else:
                st.warning(f"综合评级：**{ff_summary.rating}**")
            st.caption(
                f"市价 **{current_price:,.2f}** {currency} · "
                f"估值包络 **{ff_summary.global_low:,.2f} – {ff_summary.global_high:,.2f}** {currency}"
            )

            fig = pipeline.football_chart.plot_football_field(ff_summary)
            st.plotly_chart(fig, use_container_width=True)

            band_rows = [
                {
                    "估值方法": b.name,
                    "下限": f"{b.low:,.2f}",
                    "上限": f"{b.high:,.2f}",
                    "中点": f"{b.mid:,.2f}",
                }
                for b in ff_summary.bands
            ]
            st.dataframe(pd.DataFrame(band_rows), use_container_width=True, hide_index=True)

            st.markdown(
                """
**评级规则**
- 市价低于所有区间下限 → **买入（低估）**
- 市价高于所有区间上限 → **卖出（高估）**
- 其余 → **持有**（各方法结果分布在市价两侧时，以 FTE 目标价为主目标）
                """
            )
        except ValueError as exc:
            st.error(str(exc))
    elif current_price <= 0:
        st.info("请输入或抓取有效的当前市场价格。")
    else:
        st.info("完成模块五（DCF 敏感性）与模块六（相对估值）后，此处将自动生成 Football Field 图。")
