import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

rcParams['font.family'] = 'Noto Sans Thai'

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="พยากรณ์ค่าฝุ่น PM2.5",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------- Step 1: Load and Combine Data ------------------- #
base_url = "https://raw.githubusercontent.com/IamMegaNoobb/WeatherForecast/main/WeatherForecast/WeatherForecast/PM2.5-Files/"

years = ['2020', '2021', '2022', '2023']
train_df_list = []

for year in years:
    url = base_url + f'average_daliy_PM2.5({year}).xlsx'
    df = pd.read_excel(url, index_col=0, parse_dates=True)
    train_df_list.append(df)

train_df = pd.concat(train_df_list)

# โหลด test set ปี 2024
test_url = base_url + 'average_daliy_PM2.5(2024).xlsx'
test_df = pd.read_excel(test_url, index_col=0, parse_dates=True)

# ส่วนหัวของแอปพลิเคชัน
st.markdown("<h1 class='header'>พยากรณ์ค่าฝุ่น PM2.5</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>ระบบพยากรณ์ค่าฝุ่น PM2.5 ด้วย AIRMIND SIGHT: ระบบพยากรณ์อากาศและฝุ่น PM2.5 ร่วมกับ Predictive Model เพื่อกลยุทธ์ทางธุรกิจและความปลอดภัยของสังคม</p>", unsafe_allow_html=True)

# สร้างส่วนการเลือกวันที่และจังหวัด
col1, col2 = st.columns([1, 1])

with col1:
    selected_date = st.date_input(
        "เลือกวันที่ต้องการพยากรณ์",
        datetime.datetime.now().date(),
        min_value=datetime.datetime.now().date() - datetime.timedelta(days=365*5),  # ย้อนหลัง 5 ปี
        max_value=datetime.datetime.now().date() + datetime.timedelta(days=365*5)   # ล่วงหน้า 5 ปี
    )
    # แปลงวันที่เป็น string ในรูปแบบ วัน/เดือน/ปี
    date = selected_date.strftime("%d/%m/%Y")
    # แสดงผลลัพธ์
    # st.write(f"วันที่ที่เลือกคือ: {date}")

with col2:
    # ------------------- Step 2: Choose a Province -------------------
    # province = st.text_input("\nกรุณาใส่ชื่อจังหวัด: ", key="province_input")
    province = st.selectbox("เลือกจังหวัด", sorted(train_df.columns))
    if province not in train_df.columns:
        st.error(f"กรุณาใส่ชื่อจังหวัด {province}")
        st.stop()
    

# ปุ่มพยากรณ์
forecast_button = st.button("พยากรณ์ค่าฝุ่น", use_container_width=True)

# ------------------- Step 3: Feature Engineering -------------------
def create_features(df, province_name):
    df = df.copy()
    df['dayofyear'] = df.index.dayofyear
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday
    df['lag1'] = df[province_name].shift(1)
    df['lag2'] = df[province_name].shift(2)
    df['lag3'] = df[province_name].shift(3)
    return df

# รวม train และ test ชั่วคราว เพื่อไม่ให้ lag ตัดวันต้นปี 2024
full_df = pd.concat([train_df, test_df])
full_feat = create_features(full_df[[province]], province)

# แยกกลับออกหลังจากสร้าง lag แล้ว
train_feat = full_feat.loc[train_df.index].dropna()
test_feat = full_feat.loc[test_df.index]

# แยก X, y
X_train = train_feat.drop(columns=[province])
X_test = test_feat.drop(columns=[province])
y_train = train_feat[province]
y_test = test_feat[province]

# ------------------- Step 4: Scaling -------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------- Step 5: Train XGBoost -------------------
model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train_scaled, y_train)

# ------------------- Step 6: Predict and Evaluate -------------------
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)


# ฟังก์ชันแปลงค่า PM2.5 เป็นระดับคุณภาพอากาศ
def get_air_quality_level(pm25):
    if pm25 <= 15:
        return {
            "level": "ดีมาก",
            "description": "อากาศดี ไม่มีผลกระทบต่อสุขภาพ",
            "color" : "	#00E400"
        }
    elif pm25 <= 25:
        return {
            "level": "ดี",
            "description": "อากาศดี สามารถทำกิจกรรมกลางแจ้งได้ปกติ",
            "color" : "#A3E635"
        }
    elif pm25 <= 37:
        return {
            "level": "ปานกลาง",
            "description": "เริ่มมีผลต่อผู้ที่ไวต่อมลพิษ",
            "color" : "	#FFD700"
        }
    elif pm25 <= 50:
        return {
            "level": "เริ่มมีผลกระทบต่อสุขภาพ",
            "description": "กลุ่มเสี่ยงควรลดเวลาทำกิจกรรมกลางแจ้ง",
            "color" : "#FFA500"
        }
    elif pm25 <= 90:
        return {
            "level": "มีผลกระทบต่อสุขภาพ",
            "description": "ประชาชนทั่วไปควรหลีกเลี่ยงกิจกรรมกลางแจ้ง",
            "color" : "#FF4500"
        }
    else:
        return {
            "level": "อันตราย",
            "description": "ทุกคนควรอยู่ภายในอาคาร ปิดประตูหน้าต่าง",
            "color" : "#FF0000"
        }
        

def forecast_pm25():
    # แสดงหัวข้อผลการพยากรณ์
    st.markdown(f"<h2 class='header'>ผลการพยากรณ์ค่าฝุ่น PM2.5 จังหวัด{province}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p>วันที่ {date}</p>", unsafe_allow_html=True)
    
    df_select = pd.DataFrame(y_pred)
    df_select.columns = [province]
    df_select.index = X_test.index
    
    pm25 = df_select.loc[[selected_date]].values[0][0]

    result = get_air_quality_level(pm25)

    # แสดงค่าและระดับคุณภาพอากาศ
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"<div class='pm25-value' style='color:#FF9800'>{pm25:.1f}</div>", unsafe_allow_html=True)
        st.markdown("<p>µg/m³</p>", unsafe_allow_html=True)       
    
    with col2:
        st.markdown(f"<div class='pm25-level' style='color:{result['color']};'>{result["level"]}</div>", unsafe_allow_html=True)
        st.markdown(f"<p class='pm25-description'>{result["description"]}</p>", unsafe_allow_html=True)
    
    
def plot_1year():
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(y_test.index, y_test, label='Actual', linewidth=2)
    ax.plot(y_test.index, y_pred, label='Predicted', linewidth=2)
    ax.set_title(f'PM2.5 Forecasting in {province} (2024)', fontsize=18)
    ax.set_xlabel('Date')
    ax.set_ylabel('PM2.5 (µg/m³)')
    ax.legend()
    ax.grid(True)

    # ใส่ค่า MSE ในมุมล่างซ้าย
    ax.text(0.01, 0.01, f'MSE: {mse:.2f}', transform=ax.transAxes,
            fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    # ปรับ layout และแสดงผลใน Streamlit
    fig.tight_layout()
    st.pyplot(fig)

    
def plot_7day():
    # สร้าง DataFrame จากค่าที่พยากรณ์
    df_date = pd.DataFrame(y_pred)
    df_date.columns = [province]
    df_date["date"] = X_test.index

    # เลือกแถวที่ตรงกับวันที่ต้องการ
    df_7day_idx = df_date[df_date['date'] == date].index

    # ดึงข้อมูล 7 วันจากวันที่ที่เลือก
    df_7day = [df_date.iloc[i:i+7] for i in df_7day_idx]
    df_7day = pd.concat(df_7day)

    # สร้าง DataFrame จากค่าที่พยากรณ์
    df_date = pd.DataFrame(y_pred)
    df_date.columns = [province]
    df_date["date"] = X_test.index

    # เลือกแถวที่ตรงกับวันที่ต้องการ
    df_7day_idx = df_date[df_date['date'] == date].index

    # ดึงข้อมูล 7 วันจากวันที่ที่เลือก
    df_7day = [df_date.iloc[i:i+7] for i in df_7day_idx]
    df_7day = pd.concat(df_7day)
    
    df_7day_index = df_7day.set_index("date")
    
    # fig, ax = plt.subplots(figsize=(15, 6))
    # ax.plot(df_7day["date"].to_numpy(), df_7day[province].to_numpy(), label='PM2.5 in 7 days', linewidth=2)
    # ax.set_title(f'PM2.5 Forecasting in {province} 2024 (7 days)', fontsize=16)
    # ax.set_xlabel('Date')
    # ax.set_ylabel('PM2.5 µg/m³')
    # ax.legend()
    # ax.grid(True)
    # st.pyplot(fig)
    
    # #แสดงตารางค่าฝุ่น
    # st.dataframe(df_7day_index)
    
    # ตั้งค่าสำหรับ fig.update_layout()
    y_min = df_7day[province].min()
    y_max = df_7day[province].max()
    y_range = [
        max(0, y_min - 5),
        min(100, y_max + 5)
    ]
    
    fig = px.line(df_7day, x='date', y=[province], markers=True)
    
    fig.update_traces(
        line=dict(color='#1976D2', width=3),
        marker=dict(size=10, color='#1976D2')
    )
    fig.update_layout(
        title='แนวโน้มค่าฝุ่น PM2.5 (7 วัน)',
        xaxis_title='Date',
        yaxis_title='PM2.5 (µg/m³)',
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', range=y_range),
        plot_bgcolor='rgba(255, 255, 255, 0.05)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=50, b=20),
        height=400
    )
    
    # เพิ่มเส้นแบ่งระดับคุณภาพอากาศ
    fig.add_shape(
        type="line", line=dict(color="#FFC107", width=1, dash="dash"),
        y0=15, y1=15, x0=0, x1=1, xref="paper"
    )
    fig.add_shape(
        type="line", line=dict(color="#FFD700", width=1, dash="dash"),
        y0=25, y1=25, x0=0, x1=1, xref="paper"
    )
    fig.add_shape(
        type="line", line=dict(color="#FFA500", width=1, dash="dash"),
        y0=37, y1=37, x0=0, x1=1, xref="paper"
    )
    fig.add_shape(
        type="line", line=dict(color="#FF4500", width=1, dash="dash"),
        y0=50, y1=50, x0=0, x1=1, xref="paper"
    )
    fig.add_shape(
        type="line", line=dict(color="#FF0000", width=1, dash="dash"),
        y0=90, y1=90, x0=0, x1=1, xref="paper"
    )
    
    # Plot
    st.plotly_chart(fig)


# แสดงผลการพยากรณ์เมื่อกดปุ่ม
if forecast_button or 'last_forecast' in st.session_state:
    forecast_pm25()
    plot_7day()
    
    st.markdown(f"""
    **ข้อมูลเพิ่มเติมเกี่ยวกับโมเดล**
    - Model: XGBoost Regressor
    - MAE: {mae:.2f}
    - MSE: {mse:.2f}
    - RMSE: {rmse:.2f}
    """)

# ฟังก์ชันจำลองการแนะนำโรงแรมปลอดฝุ่น
def get_dust_free_hotels(province):
    """
    จำลองการแนะนำโรงแรมปลอดฝุ่นใกล้กับจังหวัดที่เลือก
    ในสถานการณ์จริง ฟังก์ชันนี้จะเชื่อมต่อกับฐานข้อมูลโรงแรม
    """
    # ข้อมูลจำลองโรงแรมปลอดฝุ่น
    hotels_data = {
        "กรุงเทพมหานคร": [
            {"name": "Bangkok Clean Air Hotel", "description": "โรงแรมระบบฟอกอากาศมาตรฐานสูง ใจกลางกรุงเทพฯ", "rating": 4.8},
            {"name": "Sukhumvit Fresh Resort", "description": "รีสอร์ทสไตล์โมเดิร์นพร้อมระบบกรองอากาศทุกห้อง", "rating": 4.7},
            {"name": "Silom Breeze Hotel", "description": "โรงแรมธุรกิจที่ใส่ใจคุณภาพอากาศ ย่านสีลม", "rating": 4.6},
            {"name": "Riverside Pure Air", "description": "วิวแม่น้ำเจ้าพระยาพร้อมระบบฟอกอากาศทั้งโรงแรม", "rating": 4.9},
            {"name": "Central Embassy Clean Living", "description": "ห้องพักหรูพร้อมเทคโนโลยีฟอกอากาศล่าสุด", "rating": 4.7}
        ],
        "เชียงใหม่": [
            {"name": "Nimman Clean Air Residence", "description": "ที่พักย่านนิมมานพร้อมระบบฟอกอากาศทุกห้อง", "rating": 4.7},
            {"name": "Ping River Fresh Hotel", "description": "โรงแรมริมแม่น้ำปิงที่รับรองคุณภาพอากาศสะอาด", "rating": 4.8},
            {"name": "Mountain Breeze Resort", "description": "รีสอร์ทบนดอยที่มีค่าฝุ่นต่ำกว่าในเมือง", "rating": 4.9},
            {"name": "Old City Purified Stay", "description": "เกสต์เฮาส์ในเมืองเก่าพร้อมระบบกรองอากาศ", "rating": 4.6},
            {"name": "Mae Rim Fresh Air Villa", "description": "วิลล่าส่วนตัวในแม่ริมพร้อมเครื่องฟอกอากาศ", "rating": 4.8}
        ]
    }
    
    # สร้างข้อมูลสำหรับจังหวัดที่ไม่มีในฐานข้อมูล
    if province not in hotels_data:
        return [
            {"name": f"{province} Clean Air Resort", "description": f"รีสอร์ทปลอดฝุ่นชั้นนำใน{province}", "rating": round(random.uniform(4.5, 4.9), 1)},
            {"name": f"{province} Fresh Breeze Hotel", "description": f"โรงแรมพร้อมระบบฟอกอากาศมาตรฐานสูงใน{province}", "rating": round(random.uniform(4.5, 4.9), 1)},
            {"name": f"{province} Pure Air Villa", "description": f"วิลล่าส่วนตัวพร้อมระบบกรองอากาศใน{province}", "rating": round(random.uniform(4.5, 4.9), 1)},
            {"name": f"Central {province} Clean Living", "description": f"ที่พักใจกลาง{province}พร้อมเทคโนโลยีฟอกอากาศ", "rating": round(random.uniform(4.5, 4.9), 1)},
            {"name": f"{province} Healthy Stay", "description": f"โรงแรมเพื่อสุขภาพที่ใส่ใจคุณภาพอากาศใน{province}", "rating": round(random.uniform(4.5, 4.9), 1)}
        ]
    
    return hotels_data[province]

# ฟังก์ชันสร้างกราฟแนวโน้มค่าฝุ่น 7 วัน
# def create_forecast_trend(province, selected_date):
    # # สร้างข้อมูลสำหรับ 7 วัน (3 วันก่อนหน้า วันที่เลือก และ 3 วันถัดไป)
    # dates = [selected_date + datetime.timedelta(days=i-3) for i in range(7)]
    # values = [predict_pm25(province, date) for date in dates]
    
    # # สร้าง DataFrame
    # df = pd.DataFrame({
    #     'วันที่': dates,
    #     'PM2.5': values
    # })
    
    # # สร้างกราฟ
    # fig = px.line(df, x='วันที่', y='PM2.5', markers=True)
    
    # # ปรับแต่งกราฟ
    # fig.update_traces(
    #     line=dict(color='#1976D2', width=3),
    #     marker=dict(size=10, color='#1976D2')
    # )
    # fig.update_layout(
    #     title='แนวโน้มค่าฝุ่น PM2.5 (7 วัน)',
    #     xaxis_title='วันที่',
    #     yaxis_title='PM2.5 (µg/m³)',
    #     plot_bgcolor='rgba(240, 249, 255, 0.5)',
    #     paper_bgcolor='rgba(0,0,0,0)',
    #     margin=dict(l=20, r=20, t=50, b=20),
    #     height=300
    # )
    
    # เพิ่มเส้นแบ่งระดับคุณภาพอากาศ
    # fig.add_shape(
    #     type="line", line=dict(color="#4CAF50", width=1, dash="dash"),
    #     y0=25, y1=25, x0=0, x1=1, xref="paper"
    # )
    # fig.add_shape(
    #     type="line", line=dict(color="#FFC107", width=1, dash="dash"),
    #     y0=50, y1=50, x0=0, x1=1, xref="paper"
    # )
    # fig.add_shape(
    #     type="line", line=dict(color="#FF9800", width=1, dash="dash"),
    #     y0=100, y1=100, x0=0, x1=1, xref="paper"
    # )
    
    # return fig

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .forecast-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    .hotel-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        height: 100%;
    }
    .hotel-name {
        font-weight: bold;
        color: #1976D2;
        margin-bottom: 5px;
    }
    .hotel-description {
        color: #555;
        font-size: 0.9em;
    }
    .hotel-rating {
        color: #FF9800;
        font-weight: bold;
    }
    .header {
        color: #1976D2;
        font-weight: bold;
    }
    .subheader {
        color: #555;
        font-size: 1.1em;
    }
    .pm25-value {
        font-size: 3em;
        font-weight: bold;
        margin: 10px 0;
    }
    .pm25-level {
        font-size: 1.5em;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .pm25-description {
        color: #555;
    }
    .footer {
        text-align: center;
        color: #777;
        font-size: 0.8em;
        margin-top: 30px;
        padding: 10px;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# # แสดงผลการพยากรณ์เมื่อกดปุ่ม
# if forecast_button or 'last_forecast' in st.session_state:
    
#     forecast_pm25()
    
    # # บันทึกการพยากรณ์ล่าสุดใน session state
    # if forecast_button:
    #     st.session_state.last_forecast = {
    #         'date': selected_date,
    #         'province': selected_province
    #     }
    
    # # ใช้ค่าจาก session state ถ้ามี
    # forecast_date = st.session_state.last_forecast['date'] if 'last_forecast' in st.session_state else selected_date
    # forecast_province = st.session_state.last_forecast['province'] if 'last_forecast' in st.session_state else selected_province
    
    # พยากรณ์ค่าฝุ่น
    # pm25_value = predict_pm25(forecast_province, forecast_date)
    # level, color, description = get_air_quality_level(pm25_value)
    
    
    # # แสดงผลการพยากรณ์
    # st.markdown("<div class='forecast-card'>", unsafe_allow_html=True)
    
    # # แสดงหัวข้อผลการพยากรณ์
    # st.markdown(f"<h2 class='header'>ผลการพยากรณ์ค่าฝุ่น PM2.5 จังหวัด{forecast_province}</h2>", unsafe_allow_html=True)
    # st.markdown(f"<p>วันที่ {forecast_date.strftime('%d/%m/%Y')}</p>", unsafe_allow_html=True)
    
    # # แสดงค่าและระดับคุณภาพอากาศ
    # col1, col2 = st.columns([1, 2])
    
    # with col1:
    #     st.markdown(f"<div class='pm25-value' style='color:{color}'>{pm25_value}</div>", unsafe_allow_html=True)
    #     st.markdown("<p>µg/m³</p>", unsafe_allow_html=True)
    
    # with col2:
    #     st.markdown(f"<div class='pm25-level' style='color:{color}'>{level}</div>", unsafe_allow_html=True)
    #     st.markdown(f"<p class='pm25-description'>{description}</p>", unsafe_allow_html=True)
    
    # # แสดงกราฟแนวโน้ม
    # st.plotly_chart(create_forecast_trend(forecast_province, forecast_date), use_container_width=True)
    
    # st.markdown("</div>", unsafe_allow_html=True)
    
    # # แสดงโรงแรมปลอดฝุ่นที่แนะนำ
    # st.markdown(f"<h2 class='header'>โรงแรมปลอดฝุ่นที่แนะนำใกล้{forecast_province}</h2>", unsafe_allow_html=True)
    
    # # ดึงข้อมูลโรงแรม
    # hotels = get_dust_free_hotels(forecast_province)
    
    # # แสดงโรงแรมในรูปแบบกริด
    # cols = st.columns(5)
    # for i, hotel in enumerate(hotels):
    #     with cols[i]:
    #         st.markdown(f"""
    #         <div class='hotel-card'>
    #             <div class='hotel-name'>{hotel['name']}</div>
    #             <div class='hotel-rating'>★ {hotel['rating']}</div>
    #             <div class='hotel-description'>{hotel['description']}</div>
    #         </div>
    #         """, unsafe_allow_html=True)


# ส่วนท้ายของแอปพลิเคชัน
st.markdown("""
<div class='footer'>
    <p>© 2025 ระบบพยากรณ์ค่าฝุ่น PM2.5 | ข้อมูลนี้เป็นการพยากรณ์เพื่อเป็นแนวทางเท่านั้น</p>
</div>
""", unsafe_allow_html=True)
