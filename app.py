import streamlit as st
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import altair as alt
import holidays

# ページ設定
st.set_page_config(page_title="飲食店AI売上予測", layout="wide")

st.title('🍜 飲食店向け AI売上予測 ')
st.markdown("過去データを元に、**指定した期間の売上**を予測します。")
st.markdown("🗑️ **過去の売上0円の日**は、自動的に学習データから除外されます（定休日対策）。")

# --- サイドバー ---
st.sidebar.header("1. データ入力")
st.sidebar.info("""
**【CSVデータの注意点】**
* **1列目**: 日付 (`2025/10/31` 形式推奨)
* **2列目**: 売上 (数値のみ)
* **備考**: 2年以上のデータ推奨。貸切や営業短縮は外れ値として除外したほうが精度は高くなる。
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
            
            # 1. 日付エラー削除
            df = df.dropna(subset=['ds'])
            
            # ★ 2. 売上が0円以下の行を削除（ここが新機能！）
            # これにより、過去の定休日が「売上減」として学習されるのを防ぎます
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

                # --- 3. 休業日の設定 ---
                st.sidebar.header("3. 未来の休業日設定")
                
                # (A) 毎週の定休日
                weekdays_jp = ["月", "火", "水", "木", "金", "土", "日"]
                closed_days = st.sidebar.multiselect(
                    "毎週の定休日を選んでください",
                    options=weekdays_jp,
                    default=[]
                )
                
                # (B) 臨時休業日
                st.sidebar.caption("臨時休業日（例: 2025/12/31）")
                special_holidays_text = st.sidebar.text_area(
                    "日付を入力（複数ある場合は改行）",
                    height=100,
                    placeholder="2025/12/31\n2026/01/01"
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

                    # --- 休業日を0円にする処理 ---
                    weekday_map = {
                        "月": 0, "火": 1, "水": 2, "木": 3, "金": 4, "土": 5, "日": 6
                    }
                    target_weekdays = [weekday_map[day] for day in closed_days]
                    
                    special_holidays_list = []
                    if special_holidays_text:
                        for line in special_holidays_text.split('\n'):
                            line = line.strip()
                            if line:
                                try:
                                    dt = pd.to_datetime(line)
                                    special_holidays_list.append(dt)
                                except:
                                    pass

                    def apply_holidays(row):
                        if row['ds'].weekday() in target_weekdays:
                            return 0
                        for holiday in special_holidays_list:
                            if row['ds'].date() == holiday.date():
                                return 0
                        return row['yhat']

                    future_forecast['yhat'] = future_forecast.apply(apply_holidays, axis=1)
                    future_forecast.loc[future_forecast['yhat'] == 0, ['yhat_lower', 'yhat_upper']] = 0

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
