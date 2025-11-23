import streamlit as st
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import altair as alt
import holidays

# ページ設定
st.set_page_config(page_title="飲食店AI売上予測", layout="wide")

st.title('🍜 飲食店向け AI売上予測')
st.markdown("過去データを元に、**指定した期間の売上**を予測します。")
st.markdown("⚡ **0円の日は自動で削除。時短営業やイベント**による売上の増減（掛け率）を手動で設定可能")

# --- サイドバー ---
st.sidebar.header("1. データ入力")
st.sidebar.info("""
**【CSVデータの注意点】**
* **1列目**: 日付 (`2025/10/31` 形式推奨)
* **2列目**: 売上 (数値のみ)
* **備考**: データは2年以上推奨。また、貸切や営業変更はCSVから削除すると精度向上
""")

uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type="csv")

# サンプルCSV
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

sample_data = pd.DataFrame({
    'ds': ['2025-10-27', '2025-10-28', '2025-10-29', '2025-10-30', '2025-10-31'],
    'y': [100000, 150000, 120000, 130000, 160000]
})
csv = convert_df(sample_data)
st.sidebar.download_button("サンプルCSVをDL", csv, "sample.csv", "text/csv")

# --- メイン処理 ---
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        if len(df.columns) >= 2:
            df.columns = ['ds', 'y']
            df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
            
            # 日付エラー削除
            df = df.dropna(subset=['ds'])
            
            # 売上が0円以下の行を削除（定休日対策）
            original_len = len(df)
            df = df[df['y'] > 0]
            deleted_count = original_len - len(df)
            
            if deleted_count > 0:
                st.sidebar.warning(f"⚠️ 売上が0円（または空白）のデータ {deleted_count}件 を学習から除外しました。")

            if len(df) == 0:
                 st.error("有効なデータがありません。CSVを確認してください。")
            else:
                last_date = df['ds'].max()
                st.sidebar.success(f"📅 データ最終日: {last_date.strftime('%Y/%m/%d')}")

                # --- 2. 予測期間の設定 ---
                st.sidebar.header("2. 予測期間の設定")
                
                default_start = last_date + timedelta(days=1)
                default_end = default_start + timedelta(days=30)
                
                col_date1, col_date2 = st.sidebar.columns(2)
                with col_date1:
                    target_start_date = st.date_input("開始日", value=default_start)
                with col_date2:
                    target_end_date = st.date_input("終了日", value=default_end)
                
                target_start = pd.to_datetime(target_start_date)
                target_end = pd.to_datetime(target_end_date)

                # --- 3. 休業日・調整日の設定 ---
                st.sidebar.header("3. 休業・営業時間の設定")
                
                # (A) 毎週の定休日
                weekdays_jp = ["月", "火", "水", "木", "金", "土", "日"]
                closed_days = st.sidebar.multiselect(
                    "毎週の定休日 (売上0円)",
                    options=weekdays_jp,
                    default=[]
                )
                
                # (B) 詳細設定（臨時休業＆掛け率）
                st.sidebar.caption("臨時休業・営業時間変更")
                st.sidebar.info("""
                **書き方のルール**
                * 休業日: `2025/12/31` (日付のみ)
                * 調整日: `2025/01/01, 0.5` (日付, 倍率)
                ※ 0.5は半分、1.2は1.2倍の意味です
                """)
                
                special_settings_text = st.sidebar.text_area(
                    "日付設定入力欄",
                    height=150,
                    placeholder="2025/12/31\n2026/01/01, 0.5\n2026/01/02, 1.2"
                )

                # --- エラーチェックと実行 ---
                if target_start <= last_date:
                    st.error(f"⚠️ 開始日は、データ最終日（{last_date.strftime('%m/%d')}）よりあとの日付にしてください。")
                elif target_start > target_end:
                    st.error("⚠️ 終了日は、開始日よりあとの日付にしてください。")
                else:
                    st.success(f"**{target_start.strftime('%m/%d')} 〜 {target_end.strftime('%m/%d')}** の売上を予測します...")
                    
                    with st.spinner('AIが計算中...'):
                        m = Prophet()
                        m.add_country_holidays(country_name='JP') 
                        m.fit(df)
                        
                        days_to_predict = (target_end - last_date).days + 5
                        future = m.make_future_dataframe(periods=days_to_predict)
                        forecast = m.predict(future)

                    # 期間切り出し
                    target_mask = (forecast['ds'] >= target_start) & (forecast['ds'] <= target_end)
                    future_forecast = forecast.loc[target_mask].copy()

                    # --- ★ここで休業・調整を適用する処理 ---
                    
                    # 1. 曜日変換用マップ
                    weekday_map = {
                        "月": 0, "火": 1, "水": 2, "木": 3, "金": 4, "土": 5, "日": 6
                    }
                    target_weekdays = [weekday_map[day] for day in closed_days]
                    
                    # 2. テキストエリアの解析（日付と倍率を取り出す）
                    special_adjustments = {} # 日付: 倍率 の辞書
                    
                    if special_settings_text:
                        for line in special_settings_text.split('\n'):
                            line = line.strip()
                            if line:
                                parts = line.split(',')
                                try:
                                    dt_str = parts[0].strip()
                                    dt_obj = pd.to_datetime(dt_str).date()
                                    
                                    if len(parts) > 1:
                                        # カンマがあれば倍率指定とみなす
                                        ratio = float(parts[1].strip())
                                        special_adjustments[dt_obj] = ratio
                                    else:
                                        # カンマがなければ休業(0倍)とみなす
                                        special_adjustments[dt_obj] = 0.0
                                except:
                                    pass # 読み取れない行は無視

                    # 3. 適用関数
                    def apply_adjustments(row):
                        current_date = row['ds'].date()
                        
                        # (A) 個別設定（テキストエリア）を最優先
                        if current_date in special_adjustments:
                            ratio = special_adjustments[current_date]
                            return row['yhat'] * ratio
                        
                        # (B) 毎週の定休日
                        if row['ds'].weekday() in target_weekdays:
                            return 0
                        
                        # (C) 通常通り
                        return row['yhat']

                    # 適用実行
                    future_forecast['yhat'] = future_forecast.apply(apply_adjustments, axis=1)
                    
                    # 0円にした日は、予測の幅（最小・最大）も0にする
                    # 倍率をかけた日は、幅も倍率をかける
                    def apply_adjustments_bounds(row, col_name):
                        current_date = row['ds'].date()
                        if current_date in special_adjustments:
                            return row[col_name] * special_adjustments[current_date]
                        if row['ds'].weekday() in target_weekdays:
                            return 0
                        return row[col_name]

                    future_forecast['yhat_lower'] = future_forecast.apply(lambda r: apply_adjustments_bounds(r, 'yhat_lower'), axis=1)
                    future_forecast['yhat_upper'] = future_forecast.apply(lambda r: apply_adjustments_bounds(r, 'yhat_upper'), axis=1)

                    if len(future_forecast) == 0:
                        st.error("予測データの取得に失敗しました。")
                    else:
                        # --- 表示エリア ---
                        st.markdown(f"### 🎯 {target_start.strftime('%Y/%m/%d')} 〜 {target_end.strftime('%m/%d')} の予測結果")
                        
                        total_sales = future_forecast['yhat'].sum()
                        st.markdown(f"## 💰 期間合計予測: <span style='color:#FF4B4B'>{int(total_sales):,} 円</span>", unsafe_allow_html=True)
                        st.markdown("---")

                        col1, col2 = st.columns([1, 2])

                        with col1:
                            st.subheader("📅 日別の予測表 (円)")
                            display_df = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                            display_df.columns = ['日付', '予測売上', '最低予測', '最大予測']
                            display_df['曜日'] = display_df['日付'].dt.strftime('%a')
                            display_df = display_df[['日付', '曜日', '予測売上', '最低予測', '最大予測']]
                            display_df['日付'] = display_df['日付'].dt.date
                            display_df = display_df.round(0)
                            
                            st.dataframe(display_df, height=500)
                            
                            csv_data = display_df.to_csv(index=False).encode('utf-8')
                            csv_name = f"sales_{target_start.strftime('%Y%m%d')}_{target_end.strftime('%m%d')}.csv"
                            st.download_button("📥 CSVダウンロード", csv_data, csv_name, "text/csv")

                        with col2:
                            st.subheader("📈 売上推移グラフ (万円)")
                            
                            chart_df = future_forecast[['ds', 'yhat']].copy()
                            chart_df['売上(万円)'] = chart_df['yhat'] / 10000
                            
                            jp_holidays = holidays.Japan()
                            chart_df['is_holiday'] = chart_df['ds'].apply(
                                lambda x: x.weekday() >= 5 or x in jp_holidays
                            )
                            
                            line = alt.Chart(chart_df).mark_line(point=True, color='#2563EB').encode(
                                x=alt.X('ds', title='日付', axis=alt.Axis(format='%m/%d')),
                                y=alt.Y('売上(万円)', title='売上 (万円)'),
                                tooltip=[alt.Tooltip('ds', title='日付', format='%Y/%m/%d'), alt.Tooltip('売上(万円)', format='.1f')]
                            )
                            
                            holidays_chart = alt.Chart(chart_df).transform_filter(
                                alt.datum.is_holiday == True
                            ).mark_rule(color='red', opacity=0.1, strokeWidth=15).encode(x='ds')

                            st.altair_chart((holidays_chart + line).interactive(), use_container_width=True)
                            st.caption("🟥 赤い縦帯がついている日は「土日」または「祝日」です。")

        else:
            st.error("CSVの列が足りません。")

    except Exception as e:
        st.error(f"エラー: {e}")
