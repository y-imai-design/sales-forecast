import streamlit as st
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import altair as alt
import holidays

# ページ設定
st.set_page_config(page_title="飲食店AI売上予測", layout="wide")

st.title('🍜 飲食店向け AI売上予測 (人時売上高・シフト計算機能付き)')
st.markdown("過去データを元に、指定した期間の売上と、**目標人時売上高に基づく適正労働時間**を算出します。")

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
                
                # (B) 詳細設定
                st.sidebar.caption("臨時休業・営業時間変更")
                st.sidebar.info("例: `2025/12/31` (休み), `2026/01/01, 0.5` (売上半分)")
                
                special_settings_text = st.sidebar.text_area(
                    "日付設定入力欄",
                    height=100,
                    placeholder="2025/12/31\n2026/01/01, 0.5"
                )

                # --- 4. 目標設定（人時売上高） ---
                st.sidebar.header("4. 目標設定")
                st.sidebar.markdown("目標とする **人時売上高** を入力してください。")
                
                target_productivity = st.sidebar.number_input(
                    "目標人時売上高 (円/時間)",
                    min_value=1000,
                    value=5000,
                    step=100,
                    help="従業員1人が1時間に稼ぐ売上の目標値です。一般的に4000円〜6000円程度が目安です。"
                )

                # --- エラーチェックと実行 ---
                if target_start <= last_date:
                    st.error(f"⚠️ 開始日は、データ最終日（{last_date.strftime('%m/%d')}）よりあとの日付にしてください。")
                elif target_start > target_end:
                    st.error("⚠️ 終了日は、開始日よりあとの日付にしてください。")
                else:
                    st.success(f"**{target_start.strftime('%m/%d')} 〜 {target_end.strftime('%m/%d')}** の売上と労働時間を予測します...")
                    
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

                    # --- 休業・調整を適用 ---
                    weekday_map = {"月": 0, "火": 1, "水": 2, "木": 3, "金": 4, "土": 5, "日": 6}
                    target_weekdays = [weekday_map[day] for day in closed_days]
                    
                    special_adjustments = {}
                    if special_settings_text:
                        for line in special_settings_text.split('\n'):
                            line = line.strip()
                            if line:
                                parts = line.split(',')
                                try:
                                    dt_str = parts[0].strip()
                                    dt_obj = pd.to_datetime(dt_str).date()
                                    ratio = float(parts[1].strip()) if len(parts) > 1 else 0.0
                                    special_adjustments[dt_obj] = ratio
                                except:
                                    pass

                    def apply_adjustments(row):
                        current_date = row['ds'].date()
                        if current_date in special_adjustments:
                            return row['yhat'] * special_adjustments[current_date]
                        if row['ds'].weekday() in target_weekdays:
                            return 0
                        return row['yhat']

                    future_forecast['yhat'] = future_forecast.apply(apply_adjustments, axis=1)
                    
                    # 予測の幅も調整
                    def apply_bounds(row, col):
                        d = row['ds'].date()
                        if d in special_adjustments: return row[col] * special_adjustments[d]
                        if row['ds'].weekday() in target_weekdays: return 0
                        return row[col]
                    future_forecast['yhat_lower'] = future_forecast.apply(lambda r: apply_bounds(r, 'yhat_lower'), axis=1)
                    future_forecast['yhat_upper'] = future_forecast.apply(lambda r: apply_bounds(r, 'yhat_upper'), axis=1)

                    if len(future_forecast) == 0:
                        st.error("予測データの取得に失敗しました。")
                    else:
                        # --- ★人時売上高に基づく労働時間の計算 ---
                        future_forecast['target_hours'] = future_forecast['yhat'] / target_productivity
                        
                        # --- 表示エリア ---
                        st.markdown(f"### 🎯 {target_start.strftime('%Y/%m/%d')} 〜 {target_end.strftime('%m/%d')} の予測結果")
                        
                        # 合計計算
                        total_sales = future_forecast['yhat'].sum()
                        total_hours = future_forecast['target_hours'].sum()
                        
                        # 指標の表示
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric("💰 期間予測売上", f"{int(total_sales):,} 円")
                        with col_m2:
                            st.metric("⏱️ 目標総労働時間", f"{int(total_hours):,} 時間", help="この期間で使えるスタッフの総時間枠です")
                        with col_m3:
                            st.metric("📊 設定した人時売上高", f"{int(target_productivity):,} 円/h")

                        st.markdown("---")

                        col1, col2 = st.columns([1.2, 2])

                        with col1:
                            st.subheader("📅 日別の目標労働時間")
                            
                            # 表示用データ作成
                            display_df = future_forecast[['ds', 'yhat', 'target_hours']].copy()
                            display_df.columns = ['日付', '予測売上(円)', '目標労働時間(h)']
                            
                            display_df['曜日'] = display_df['日付'].dt.strftime('%a')
                            display_df['日付'] = display_df['日付'].dt.date
                            
                            # 丸め処理
                            display_df['予測売上(円)'] = display_df['予測売上(円)'].round(0).astype(int)
                            display_df['目標労働時間(h)'] = display_df['目標労働時間(h)'].round(1) # 小数点1位まで
                            
                            # 列の並び替え
                            display_df = display_df[['日付', '曜日', '予測売上(円)', '目標労働時間(h)']]
                            
                            # 表の表示
                            st.dataframe(
                                display_df.style.format({
                                    '予測売上(円)': '{:,}',
                                    '目標労働時間(h)': '{:.1f}'
                                }), 
                                height=500
                            )
                            
                            # CSVダウンロード
                            csv_data = display_df.to_csv(index=False).encode('utf-8')
                            csv_name = f"shift_plan_{target_start.strftime('%Y%m%d')}.csv"
                            st.download_button("📥 シフト作成用CSVをDL", csv_data, csv_name, "text/csv")

                        with col2:
                            st.subheader("📈 売上と労働時間の推移")
                            
                            # グラフ用データ
                            chart_df = future_forecast[['ds', 'yhat']].copy()
                            chart_df['売上(万円)'] = chart_df['yhat'] / 10000
                            
                            # 休日フラグ
                            jp_holidays = holidays.Japan()
                            chart_df['is_holiday'] = chart_df['ds'].apply(lambda x: x.weekday() >= 5 or x in jp_holidays)
                            
                            # 売上グラフ
                            line = alt.Chart(chart_df).mark_line(point=True, color='#2563EB').encode(
                                x=alt.X('ds', title='日付', axis=alt.Axis(format='%m/%d')),
                                y=alt.Y('売上(万円)', title='売上 (万円)'),
                                tooltip=[alt.Tooltip('ds', title='日付', format='%Y/%m/%d'), alt.Tooltip('売上(万円)', format='.1f')]
                            )
                            
                            # 休日背景
                            holidays_chart = alt.Chart(chart_df).transform_filter(alt.datum.is_holiday == True).mark_rule(
                                color='red', opacity=0.1, strokeWidth=15
                            ).encode(x='ds')

                            st.altair_chart((holidays_chart + line).interactive(), use_container_width=True)
                            
                            st.info(f"💡 目標人時売上高 **{int(target_productivity):,}円** を達成するには、表の「目標労働時間」以内にシフトを収めてください。")

        else:
            st.error("CSVの列が足りません。")

    except Exception as e:
        st.error(f"エラー: {e}")
