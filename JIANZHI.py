import streamlit as st
import requests
import pandas as pd
from PIL import Image
from transformers import pipeline
from deep_translator import GoogleTranslator

# ==========================================
# 0. 全局配置 (已修复 layout 错误)
# ==========================================
st.set_page_config(
    page_title="NeuroScale Pro: 智能身材管理",
    page_icon="🥑",
    layout="centered",  # 必须是 centered 或 wide，不能是 mobile
    initial_sidebar_state="collapsed"
)

# ==========================================
# 1. 工具函数
# ==========================================
def translate_to_chinese(text):
    """把AI识别的英文单词翻译成中文"""
    try:
        translated = GoogleTranslator(source='auto', target='zh-CN').translate(text)
        return translated
    except:
        return text

def safe_float(val):
    """防弹衣：把任何垃圾数据强行转为数字"""
    try:
        if val is None: return 0.0
        return float(val)
    except:
        return 0.0

# ==========================================
# 2. 核心引擎
# ==========================================
class MetabolicEngine:
    ACTIVITY_LEVELS = {
        "久坐 (办公室工作)": 1.2,
        "轻度 (每周运动1-3次)": 1.375,
        "中度 (每周运动3-5次)": 1.55,
        "高度 (每周运动6-7次)": 1.725,
        "极度 (体力劳动/双练)": 1.9
    }

    GOAL_MODIFIERS = {
        "精瘦增肌 (+10% 热量)": 1.10,
        "保持当前状态": 1.0,
        "强力减脂 (-15% 热量)": 0.85
    }

    @staticmethod
    def calculate_targets(weight, height, age, gender, activity, goal):
        base = (10 * weight) + (6.25 * height) - (5 * age)
        bmr = base + 5 if gender == "男" else base - 161
        
        tdee = bmr * MetabolicEngine.ACTIVITY_LEVELS[activity]
        target_kcal = tdee * MetabolicEngine.GOAL_MODIFIERS[goal]
        
        p_g = weight * 2.0
        f_kcal = target_kcal * 0.25
        f_g = f_kcal / 9
        c_kcal = target_kcal - (p_g * 4) - f_kcal
        c_g = max(0, c_kcal / 4)
        
        return {
            "target_kcal": int(target_kcal),
            "p_g": int(p_g),
            "f_g": int(f_g),
            "c_g": int(c_g)
        }

class DataGateway:
    @staticmethod
    def search_food(query):
        url = "https://world.openfoodfacts.org/cgi/search.pl"
        params = {
            "search_terms": query,
            "search_simple": 1,
            "action": "process",
            "json": 1,
            "page_size": 8,
            "fields": "product_name,nutriments,code"
        }
        try:
            r = requests.get(url, params=params, timeout=5)
            data = r.json().get("products", [])
            results = []
            for item in data:
                nuts = item.get('nutriments', {})
                if 'energy-kcal_100g' in nuts:
                    name = item.get('product_name_zh', item.get('product_name', '未知食物'))
                    results.append({
                        "name": name,
                        "kcal": safe_float(nuts.get('energy-kcal_100g')),
                        "protein": safe_float(nuts.get('proteins_100g')),
                        "fat": safe_float(nuts.get('fat_100g')),
                        "carbs": safe_float(nuts.get('carbohydrates_100g'))
                    })
            return results
        except:
            return []

# ==========================================
# 3. 界面逻辑 (修复重点在这里)
# ==========================================
def main():
    # 🚨【关键修复】确保在使用 food_log 之前，它一定已经被创建了
    if 'food_log' not in st.session_state:
        st.session_state.food_log = []

    # --- 侧边栏 ---
    with st.sidebar:
        st.header("⚙️ 身体参数设置")
        gender = st.radio("性别", ["男", "女"], horizontal=True)
        age = st.number_input("年龄", 18, 60, 25)
        height = st.number_input("身高 (cm)", 150, 200, 175)
        weight = st.number_input("体重 (kg)", 40, 150, 70)
        act = st.selectbox("活动量", list(MetabolicEngine.ACTIVITY_LEVELS.keys()))
        goal = st.selectbox("目标", list(MetabolicEngine.GOAL_MODIFIERS.keys()))

    # --- 顶部仪表盘 ---
    targets = MetabolicEngine.calculate_targets(weight, height, age, gender, act, goal)
    
    # 现在这里绝对不会报错了，因为上面已经强制初始化了
    eaten_kcal = sum([x['kcal'] for x in st.session_state.food_log])
    eaten_p = sum([x['protein'] for x in st.session_state.food_log])
    
    remain_kcal = targets['target_kcal'] - eaten_kcal

    st.markdown("### 📊 今日热量余额")
    
    col_main, col_detail = st.columns([2, 1])
    with col_main:
        st.metric("还可以吃 (Kcal)", f"{int(remain_kcal)}", f"目标: {targets['target_kcal']}")
        if targets['target_kcal'] > 0:
            progress = min(1.0, eaten_kcal / targets['target_kcal'])
            st.progress(progress, text=f"已摄入 {int(progress*100)}%")
    
    with col_detail:
        st.caption("蛋白质进度")
        if targets['p_g'] > 0:
            st.progress(min(1.0, eaten_p / targets['p_g']), text=f"{int(eaten_p)}/{targets['p_g']}g")

    st.markdown("---")

    # --- 功能区 ---
    tab_manual, tab_ai = st.tabs(["🔍 手动搜索 (推荐)", "📷 AI 拍照识别"])

    # === 手动搜索 ===
    with tab_manual:
        st.caption("输入食物名称，例如：米饭、香蕉、全麦面包")
        search_query = st.text_input("搜索食物", placeholder="请输入...")
        
        if search_query:
            results = DataGateway.search_food(search_query)
            if results:
                st.success(f"找到 {len(results)} 个结果")
                food_options = [f"{r['name']} ({int(r['kcal'])}大卡/100g)" for r in results]
                selected_idx = st.selectbox("选择具体食物", range(len(food_options)), format_func=lambda x: food_options[x])
                selected_food = results[selected_idx]
                
                col_g, col_btn = st.columns([2, 1])
                with col_g:
                    portion = st.number_input("吃了多少克?", 10, 500, 100, step=10, key="manual_portion")
                
                with col_btn:
                    st.write("") 
                    st.write("") 
                    if st.button("➕ 加入记录", type="primary", key="btn_manual_add"):
                        ratio = portion / 100.0
                        item = {
                            "name": selected_food['name'],
                            "kcal": int(selected_food['kcal'] * ratio),
                            "protein": round(selected_food['protein'] * ratio, 1),
                            "carbs": round(selected_food['carbs'] * ratio, 1),
                            "fat": round(selected_food['fat'] * ratio, 1),
                            "portion": portion
                        }
                        st.session_state.food_log.append(item)
                        st.rerun()
            else:
                st.info("没搜到？试试换个词，比如用英文 'Rice' 搜搜看。")

    # === AI 拍照 ===
    with tab_ai:
        img_file = st.file_uploader("拍摄或上传图片", type=['jpg', 'jpeg'])
        if img_file:
            image = Image.open(img_file)
            st.image(image, caption="已上传", width=200)
            
            with st.spinner("正在分析并翻译..."):
                try:
                    classifier = pipeline("image-classification", model="nateraw/food")
                    pred = classifier(image)[0]
                    en_label = pred['label'].replace("_", " ")
                    confidence = pred['score']
                    
                    cn_label = translate_to_chinese(en_label)
                    
                    st.markdown(f"### 识别结果: **{cn_label}**")
                    st.caption(f"原始结果: {en_label} (置信度 {int(confidence*100)}%)")
                    
                    db_results = DataGateway.search_food(en_label)
                    
                    if db_results:
                        selected_food = db_results[0]
                        st.info(f"匹配到: {selected_food['name']}")
                        
                        portion_ai = st.number_input("吃了多少克?", 10, 500, 100, step=10, key="ai_portion")
                        
                        if st.button("➕ 确认并加入记录", key="btn_ai_add"):
                            ratio = portion_ai / 100.0
                            item = {
                                "name": cn_label,
                                "kcal": int(selected_food['kcal'] * ratio),
                                "protein": round(selected_food['protein'] * ratio, 1),
                                "carbs": round(selected_food['carbs'] * ratio, 1),
                                "fat": round(selected_food['fat'] * ratio, 1),
                                "portion": portion_ai
                            }
                            st.session_state.food_log.append(item)
                            st.rerun()
                    else:
                        st.warning("AI 识别出了名字，但数据库没数据。建议用手动搜索。")
                        
                except Exception as e:
                    st.error(f"分析出错: {str(e)}")

    # --- 记录列表 ---
    st.markdown("---")
    st.subheader(f"🍽️ 今日记录 ({len(st.session_state.food_log)} 项)")
    
    if st.session_state.food_log:
        for i, item in enumerate(reversed(st.session_state.food_log)):
            with st.container():
                c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
                c1.markdown(f"**{item['name']}**")
                c1.caption(f"{item['portion']}克")
                c2.write(f"🔥 {item['kcal']}")
                c3.write(f"🥩 P:{item['protein']}")
                
                if c4.button("❌", key=f"del_{i}"):
                    st.session_state.food_log.pop(len(st.session_state.food_log)-1-i)
                    st.rerun()
    else:
        st.info("还没有吃东西？快去添加吧！")

if __name__ == "__main__":
    main()
