import streamlit as st
import requests
import pandas as pd
from PIL import Image
from transformers import pipeline
import time

# ==========================================
# 0. 全局配置与状态管理
# ==========================================
st.set_page_config(
    page_title="NeuroScale: 严谨代谢校准系统",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 使用缓存加载AI模型
@st.cache_resource
def load_vision_model():
    try:
        model_id = "nateraw/food" 
        classifier = pipeline("image-classification", model=model_id)
        return classifier
    except Exception as e:
        return None

# ==========================================
# 1. 核心算法模块
# ==========================================
class MetabolicEngine:
    ACTIVITY_LEVELS = {
        "久坐 (Sedentary)": 1.2,
        "轻度活跃 (Lightly Active)": 1.375,
        "中度活跃 (Moderately Active)": 1.55,
        "高度活跃 (Very Active)": 1.725,
        "极度活跃 (Extra Active)": 1.9
    }

    GOAL_MODIFIERS = {
        "精瘦增肌 (Lean Bulk, +10%)": 1.10,
        "身体重组 (Recomposition, 0%)": 1.0,
        "减脂 (Cutting, -15%)": 0.85
    }

    @staticmethod
    def calculate_bmr(weight_kg, height_cm, age, gender):
        base = (10 * weight_kg) + (6.25 * height_cm) - (5 * age)
        if gender == "男":
            return base + 5
        else:
            return base - 161

    @staticmethod
    def partition_macros(tdee, weight_kg):
        protein_g = weight_kg * 2.0
        protein_kcal = protein_g * 4
        fat_kcal = tdee * 0.25
        fat_g = fat_kcal / 9
        remaining_kcal = max(0, tdee - protein_kcal - fat_kcal)
        carb_g = remaining_kcal / 4
        carb_kcal = remaining_kcal

        return {
            "Protein": {"g": int(protein_g), "kcal": int(protein_kcal)},
            "Fat": {"g": int(fat_g), "kcal": int(fat_kcal)},
            "Carbs": {"g": int(carb_g), "kcal": int(carb_kcal)},
            "Total": int(tdee)
        }

# ==========================================
# 2. 数据工程模块 (重点修复)
# ==========================================
class DataGateway:
    SEARCH_URL = "https://world.openfoodfacts.org/cgi/search.pl"

    # 【新增】防弹函数：不管来的是什么，必须变成数字
    @staticmethod
    def safe_float(val):
        try:
            if val is None:
                return 0.0
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def search_product(query_text):
        params = {
            "search_terms": query_text,
            "search_simple": 1,
            "action": "process",
            "json": 1,
            "page_size": 10,
            "fields": "product_name,nutriments,code,serving_size"
        }
        headers = {"User-Agent": "NeuroScaleApp/1.0 (Research Project)"}

        try:
            response = requests.get(DataGateway.SEARCH_URL, params=params, headers=headers, timeout=10)
            if response.status_code != 200:
                return []
            
            data = response.json()
            products = data.get("products", [])
            clean_results = []
            
            for p in products:
                nutrients = p.get("nutriments", {})
                # 只有当包含热量数据时才处理
                if "energy-kcal_100g" in nutrients:
                    # 使用 safe_float 强制转换所有数据
                    kcal = DataGateway.safe_float(nutrients.get("energy-kcal_100g"))
                    prot = DataGateway.safe_float(nutrients.get("proteins_100g"))
                    carb = DataGateway.safe_float(nutrients.get("carbohydrates_100g"))
                    fat = DataGateway.safe_float(nutrients.get("fat_100g"))

                    clean_results.append({
                        "name": p.get("product_name", "未知商品"),
                        "kcal": kcal,
                        "protein": prot,
                        "carbs": carb,
                        "fat": fat,
                        "id": p.get("code")
                    })
            return clean_results
        except Exception as e:
            return []

# ==========================================
# 3. 用户交互层
# ==========================================
def main():
    st.title("🧬 NeuroScale | 智能代谢校准系统")
    st.markdown("---")

    with st.sidebar:
        st.header("1. 生理参数校准")
        gender = st.radio("性别", ["男", "女"], horizontal=True)
        age = st.slider("年龄 (岁)", 18, 80, 25)
        height = st.number_input("身高 (cm)", 140, 220, 175)
        weight = st.number_input("体重 (kg)", 40, 150, 70)
        
        st.markdown("---")
        st.header("2. 能量消耗设定")
        activity_key = st.selectbox("日常活动水平", list(MetabolicEngine.ACTIVITY_LEVELS.keys()))
        goal_key = st.selectbox("身体重组目标", list(MetabolicEngine.GOAL_MODIFIERS.keys()))

    tab_calc, tab_vision, tab_analysis = st.tabs(["📊 核心代谢计算", "📷 AI 视觉识别", "📈 数据洞察"])

    # --- Tab 1 ---
    with tab_calc:
        st.subheader("个性化营养处方")
        if st.button("生成计算结果", type="primary"):
            bmr = MetabolicEngine.calculate_bmr(weight, height, age, gender)
            af = MetabolicEngine.ACTIVITY_LEVELS[activity_key]
            tdee_maintenance = bmr * af
            goal_mod = MetabolicEngine.GOAL_MODIFIERS[goal_key]
            target_calories = tdee_maintenance * goal_mod
            macros = MetabolicEngine.partition_macros(target_calories, weight)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("基础代谢", f"{int(bmr)}")
            col2.metric("维持热量", f"{int(tdee_maintenance)}")
            col3.metric("目标摄入", f"{int(target_calories)}", f"{int(target_calories - tdee_maintenance)} kcal")

            st.markdown("### 宏量营养素目标")
            c1, c2, c3 = st.columns(3)
            c1.info(f"蛋白质: {macros['Protein']['g']}g")
            c2.warning(f"碳水: {macros['Carbs']['g']}g")
            c3.error(f"脂肪: {macros['Fat']['g']}g")

    # --- Tab 2 ---
    with tab_vision:
        st.subheader("智能食品识别")
        img_file = st.file_uploader("上传食物照片", type=['jpg', 'png', 'jpeg'])
        
        if img_file:
            # 这里保持双列布局
            c_img, c_data = st.columns(2)
            
            with c_img:
                image = Image.open(img_file)
                st.image(image, use_column_width=True, caption="分析对象")
            
            with c_data:
                with st.spinner("AI 识别中..."):
                    classifier = load_vision_model()
                    if classifier:
                        predictions = classifier(image)
                        top_pred = predictions[0]
                        label_en = top_pred['label'].replace("_", " ")
                        conf = top_pred['score']
                        
                        st.success(f"识别结果: **{label_en.title()}**")
                        st.progress(conf, text=f"置信度: {conf:.1%}")
                        
                        db_results = DataGateway.search_product(label_en)
                        if db_results:
                            selected_item_name = st.selectbox("选择匹配项:", [item['name'] for item in db_results])
                            selected_food = next(item for item in db_results if item['name'] == selected_item_name)
                            
                            portion = st.number_input("份量 (g)", value=100, step=10)
                            ratio = portion / 100.0
                            
                            # 即使数据是0.0，这里也不会报错了
                            result_df = pd.DataFrame({
                                "营养素": ["热量", "蛋白质", "碳水", "脂肪"],
                                "总量": [
                                    int(selected_food['kcal'] * ratio),
                                    round(selected_food['protein'] * ratio, 1),
                                    round(selected_food['carbs'] * ratio, 1),
                                    round(selected_food['fat'] * ratio, 1)
                                ]
                            })
                            st.table(result_df)
                        else:
                            st.warning("未找到详细营养数据")
                    else:
                        st.error("模型加载失败")

    with tab_analysis:
        st.info("数据模块开发中...")

if __name__ == "__main__":
    main()
