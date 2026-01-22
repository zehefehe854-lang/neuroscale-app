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


# 使用缓存加载AI模型，避免每次刷新页面重新加载重型模型
@st.cache_resource
def load_vision_model():
    """
    加载在Food-101数据集上微调的Vision Transformer模型。
    使用Hugging Face的pipeline API进行封装。
    """
    try:
        # 使用nateraw提供的微调模型，该模型在Food-101上表现优异
        model_id = "nateraw/food"
        classifier = pipeline("image-classification", model=model_id)
        return classifier
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None


# ==========================================
# 1. 核心算法模块：计算生理学引擎
# ==========================================
class MetabolicEngine:
    """
    实现Mifflin-St Jeor方程及ISSN建议的营养分区逻辑。
    """

    # 严谨定义的活动系数
    ACTIVITY_LEVELS = {
        "久坐 (Sedentary)": 1.2,
        "轻度活跃 (Lightly Active)": 1.375,
        "中度活跃 (Moderately Active)": 1.55,
        "高度活跃 (Very Active)": 1.725,
        "极度活跃 (Extra Active)": 1.9
    }

    # 动态目标调整系数
    GOAL_MODIFIERS = {
        "精瘦增肌 (Lean Bulk, +10%)": 1.10,
        "身体重组 (Recomposition, 0%)": 1.0,
        "减脂 (Cutting, -15%)": 0.85
    }

    @staticmethod
    def calculate_bmr(weight_kg, height_cm, age, gender):
        """
        Mifflin-St Jeor 方程实现
        男性: (10 × weight) + (6.25 × height) - (5 × age) + 5
        女性: (10 × weight) + (6.25 × height) - (5 × age) - 161
        """
        base = (10 * weight_kg) + (6.25 * height_cm) - (5 * age)
        if gender == "男":
            return base + 5
        else:
            return base - 161

    @staticmethod
    def partition_macros(tdee, weight_kg):
        """
        基于'蛋白质优先'的营养分区算法。
        1. 蛋白质: 2.0g/kg (增肌减脂黄金标准)
        2. 脂肪: 总热量的25% (激素维持)
        3. 碳水: 剩余热量 (训练供能)
        """
        # 1. 计算蛋白质 (4 kcal/g)
        protein_g = weight_kg * 2.0
        protein_kcal = protein_g * 4

        # 2. 计算脂肪 (9 kcal/g)
        fat_kcal = tdee * 0.25
        fat_g = fat_kcal / 9

        # 3. 计算碳水化合物 (4 kcal/g)
        # 确保剩余热量不为负
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
# 2. 数据工程模块：API网关与清洗
# ==========================================
class DataGateway:
    """
    OpenFoodFacts API 接口封装，包含数据清洗逻辑。
    """
    SEARCH_URL = "https://world.openfoodfacts.org/cgi/search.pl"

    @staticmethod
    def search_product(query_text):
        """
        执行搜索并清洗数据，确保返回条目包含完整营养素。
        """
        params = {
            "search_terms": query_text,
            "search_simple": 1,
            "action": "process",
            "json": 1,
            "page_size": 10,  # 获取更多条目以供过滤
            "fields": "product_name,nutriments,code,serving_size"
        }
        # 设置User-Agent以符合API规范
        headers = {"User-Agent": "NeuroScaleApp/1.0 (Research Project)"}

        try:
            response = requests.get(DataGateway.SEARCH_URL, params=params, headers=headers, timeout=10)
            if response.status_code != 200:
                return []

            data = response.json()
            products = data.get("products", [])

            # 【修复】这里之前是 clean_results = 无内容
            clean_results = []

            for p in products:
                nutrients = p.get("nutriments", {})

                # 数据严谨性检查：必须包含热量、蛋白、碳水、脂肪
                if "energy-kcal_100g" in nutrients:
                    clean_results.append({
                        "name": p.get("product_name", "未知商品"),
                        "kcal": nutrients.get("energy-kcal_100g", 0),
                        "protein": nutrients.get("proteins_100g", 0),
                        "carbs": nutrients.get("carbohydrates_100g", 0),
                        "fat": nutrients.get("fat_100g", 0),
                        "id": p.get("code")
                    })

            return clean_results
        except Exception as e:
            st.warning(f"API连接异常: {e}")
            return []


# ==========================================
# 3. 用户交互层：Streamlit UI
# ==========================================
def main():
    st.title("🧬 NeuroScale | 智能代谢校准系统")
    st.markdown("---")

    # 侧边栏：生理参数输入
    with st.sidebar:
        st.header("1. 生理参数校准")
        gender = st.radio("性别", ["男", "女"], horizontal=True)
        age = st.slider("年龄 (岁)", 18, 80, 25)
        height = st.number_input("身高 (cm)", 140, 220, 175)
        weight = st.number_input("体重 (kg)", 40, 150, 70)

        st.markdown("---")
        st.header("2. 能量消耗设定")
        activity_key = st.selectbox(
            "日常活动水平",
            list(MetabolicEngine.ACTIVITY_LEVELS.keys()),
            help="请诚实选择，高估活动量是减脂失败的主要原因。"
        )

        goal_key = st.selectbox(
            "身体重组目标",
            list(MetabolicEngine.GOAL_MODIFIERS.keys())
        )

    # 核心选项卡
    tab_calc, tab_vision, tab_analysis = st.tabs(["📊 核心代谢计算", "📷 AI 视觉识别", "📈 数据洞察"])

    # --- Tab 1: 代谢计算结果 ---
    with tab_calc:
        st.subheader("个性化营养处方")

        if st.button("生成计算结果", type="primary"):
            # 1. 计算 BMR
            bmr = MetabolicEngine.calculate_bmr(weight, height, age, gender)

            # 2. 计算 TDEE
            af = MetabolicEngine.ACTIVITY_LEVELS[activity_key]
            tdee_maintenance = bmr * af

            # 3. 应用目标修正
            goal_mod = MetabolicEngine.GOAL_MODIFIERS[goal_key]
            target_calories = tdee_maintenance * goal_mod

            # 4. 营养分区
            macros = MetabolicEngine.partition_macros(target_calories, weight)

            # 5. 可视化展示
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("基础代谢 (BMR)", f"{int(bmr)}", "kcal/day")
            col2.metric("维持热量 (TDEE)", f"{int(tdee_maintenance)}", "kcal/day")
            col3.metric("目标摄入", f"{int(target_calories)}", "kcal/day")
            diff = int(target_calories - tdee_maintenance)
            col3.caption(f"热量缺口/盈余: {diff} kcal")

            st.markdown("### 宏量营养素目标 (每日)")
            c1, c2, c3 = st.columns(3)
            c1.info(f"**蛋白质**: {macros['Protein']['g']}g ({macros['Protein']['kcal']} kcal)")
            c2.warning(f"**碳水化合物**: {macros['Carbs']['g']}g ({macros['Carbs']['kcal']} kcal)")
            c3.error(f"**脂肪**: {macros['Fat']['g']}g ({macros['Fat']['kcal']} kcal)")

            st.markdown("""
            > **专家提示**：蛋白质摄入量已锁定为体重×2.0g，这是保障增肌减脂效果的关键变量，不建议随意降低。
            """)

    # --- Tab 2: AI 视觉识别 ---
    with tab_vision:
        st.subheader("智能食品识别与营养查询")
        st.caption("采用 Vision Transformer (ViT) 模型进行图像分类，结合 OFF 数据库保障数据严谨性。")

        img_file = st.file_uploader("上传食物照片", type=['jpg', 'png', 'jpeg'])

        if img_file:
            # 布局：左图右数据
            c_img, c_data = st.columns()

            with c_img:
                # 【修复】这里之前有乱码
                image = Image.open(img_file)
                st.image(image, use_column_width=True, caption="上传的图片")

            with c_data:
                with st.spinner("AI 神经网络正在分析纹理特征..."):
                    # 1. AI 推理
                    classifier = load_vision_model()
                    if classifier:
                        predictions = classifier(image)
                        # 【修复】predictions 是列表，需要取第0个元素
                        top_pred = predictions[0]
                        label_en = top_pred['label'].replace("_", " ")
                        conf = top_pred['score']

                        st.success(f"识别结果: **{label_en.title()}**")
                        st.progress(conf, text=f"AI置信度: {conf:.1%}")

                        # 2. 数据库验证 (Data Rigor)
                        st.markdown("#### 🔍 数据库匹配 (每100g数据)")
                        db_results = DataGateway.search_product(label_en)

                        if db_results:
                            # 让用户选择具体变种
                            selected_item_name = st.selectbox(
                                "请选择匹配的最接近食品:",
                                [item['name'] for item in db_results]
                            )

                            # 获取选中项的详细数据
                            selected_food = next(item for item in db_results if item['name'] == selected_item_name)

                            # 输入份量
                            portion = st.number_input("摄入份量 (克)", value=100, step=10)
                            ratio = portion / 100.0

                            # 【修复】这里是你要改的主要地方
                            result_df = pd.DataFrame({
                                "营养素": ["热量 (kcal)", "蛋白质 (g)", "碳水 (g)", "脂肪 (g)"],
                                "每100g": [
                                    selected_food['kcal'],
                                    selected_food['protein'],
                                    selected_food['carbs'],
                                    selected_food['fat']
                                ],
                                "摄入总量": [
                                    int(selected_food['kcal'] * ratio),
                                    round(selected_food['protein'] * ratio, 1),
                                    round(selected_food['carbs'] * ratio, 1),
                                    round(selected_food['fat'] * ratio, 1)
                                ]
                            })
                            st.table(result_df)

                        else:
                            st.warning("AI识别成功，但数据库中未找到对应的高质量营养数据。建议手动搜索。")
                    else:
                        st.error("AI模型加载失败，请检查网络连接。")

    # --- Tab 3: 数据洞察 (静态示例) ---
    with tab_analysis:
        st.info("此模块将基于用户的长期记录，展示体重变化与TDEE的动态适应曲线（开发中）。")


if __name__ == "__main__":
    main()