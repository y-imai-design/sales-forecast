import streamlit as st
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import altair as alt # グラフ描画用のライブラリ
import holidays # 祝日判定用（Prophetと一緒に入っています）

# ページ設定
st.set_page_config(page_title="飲食店AI売上予測", layout="wide")

st.title('🍜 飲食店向け AI売上予測 (祝日色付)')
st.markdown("過去データを元に、**指定した月の売上**を予測します。")
st.markdown("🎌 **土日と祝日**には、グラフに赤い縦線が入ります。")

# --- サイドバー ---
st.sidebar.header("1. データ入力")
st.sidebar.info("""
**【CSVデータの注意点】**
* **1列目**: 日付 (`2025/10/31` 形式推奨)
* **2列目**: 売上 (数値のみ　,や￥をいれない)
* **備考**: 貸切や営業変更などで外れ値があると予測しずらいので、理想はその日は消す。データとしては、2年以上が推奨
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
            # 日付変換（エラーがあっても強制的に読み込む）
            df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
            # 日付として読めなかった行だけ削除
            df = df.dropna(subset=['ds'])
            
            # ★修正点：2026年以降を削除するコードを撤廃しました
            # これで何年のデータでも読み込めます

            if len(df) == 0:
                 st.error("有効なデータがありません。CSVを確認してください。")
            else:
                last_date = df['ds'].max()
                st.sidebar.success(f"📅 データ最終日: {last_date.strftime('%Y/%m/%d')}")

                # --- 予測設定 ---
                st.sidebar.header("2. 予測設定")
                default_next_month = (last_date.replace(day=1) + timedelta(days=32)).replace(day=1)
                target_date = st.sidebar.date_input("予測したい月の「1日」を選んでください", value=default_next_month)
                
                target_start = pd.to_datetime(target_date).replace(day=1)
                target_end = (target_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
                
                if target_start <= last_date:
                    st.error("⚠️ 過去の日付が含まれています。データ最終日より後の月を選んでください。")
                else:
                    st.success(f"**{target_start.strftime('%Y年%m月')}** の売上を予測します...")
                    
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
                    
                    if len(future_forecast) == 0:
                        st.error("予測データの取得に失敗しました。")
                    else:
                        # --- 表示エリア ---
                        st.markdown(f"### 🎯 {target_start.strftime('%Y年%m月')}の予測結果")
                        
                        total_sales = future_forecast['yhat'].sum()
                        st.markdown(f"## 💰 月商予測: <span style='color:#FF4B4B'>{int(total_sales):,} 円</span>", unsafe_allow_html=True)
                        st.markdown("---")

                        col1, col2 = st.columns([1, 2])

                        with col1:
                            st.subheader("📅 日別の予測表 (円)")
                            display_df = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                            display_df.columns = ['日付', '予測売上', '最低予測', '最大予測']
                            
                            # 曜日の追加
                            display_df['曜日'] = display_df['日付'].dt.strftime('%a')
                            # 順番を入れ替え
                            display_df = display_df[['日付', '曜日', '予測売上', '最低予測', '最大予測']]
                            
                            display_df['日付'] = display_df['日付'].dt.date
                            display_df = display_df.round(0)
                            
                            st.dataframe(display_df, height=500)
                            
                            csv_data = display_df.to_csv(index=False).encode('utf-8')
                            st.download_button("📥 CSVダウンロード", csv_data, f"sales_{target_start.strftime('%Y%m')}.csv", "text/csv")

                        with col2:
                            st.subheader("📈 売上推移グラフ (万円)")
                            
                            # グラフ用データ作成
                            chart_df = future_forecast[['ds', 'yhat']].copy()
                            chart_df['売上(万円)'] = chart_df['yhat'] / 10000
                            
                            # 土日祝判定ロジック
                            jp_holidays = holidays.Japan()
                            chart_df['is_holiday'] = chart_df['ds'].apply(
                                lambda x: x.weekday() >= 5 or x in jp_holidays
                            )
                            
                            # --- Altairによる高度なグラフ描画 ---
                            
                            # 1. 売上の折れ線
                            line = alt.Chart(chart_df).mark_line(
                                point=True,  # 点を表示
                                color='#2563EB' # 青色
                            ).encode(
                                x=alt.X('ds', title='日付', axis=alt.Axis(format='%m/%d')),
                                y=alt.Y('売上(万円)', title='売上 (万円)'),
                                tooltip=[alt.Tooltip('ds', title='日付', format='%Y/%m/%d'), alt.Tooltip('売上(万円)', format='.1f')]
                            )
                            
                            # 2. 土日祝の背景帯（赤い縦線）
                            holidays_chart = alt.Chart(chart_df).transform_filter(
                                alt.datum.is_holiday == True
                            ).mark_rule(
                                color='red',
                                opacity=0.1, # 透明度（薄くする）
                                strokeWidth=15 # 線の太さ
                            ).encode(
                                x='ds'
                            )

                            # グラフを重ねて表示
                            st.altair_chart((holidays_chart + line).interactive(), use_container_width=True)
                            
                            st.caption("🟥 赤い縦帯がついている日は「土日」または「祝日」です。")

        else:
            st.error("CSVの列が足りません。")

    except Exception as e:

        st.error(f"エラー: {e}")
