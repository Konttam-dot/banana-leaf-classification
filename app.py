import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision import transforms
from PIL import Image


# =========================================================
# ตั้งค่าหน้าเว็บ
# =========================================================

st.set_page_config(
    page_title="Banana Leaf AI",
    page_icon="🍌",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CSS - ตกแต่งเว็บไซต์
# =========================================================

st.markdown("""
<style>

    /* พื้นหลัง */
    .stApp {
        background-color: #ffffff;
    }

    /* ซ่อนเมนู Streamlit */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* ระยะขอบ */
    .block-container {
        max-width: 1000px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* หัวเว็บ */
    .hero {
        text-align: center;
        padding: 30px 20px 25px 20px;
    }

    .banana-icon {
        font-size: 65px;
        margin-bottom: 5px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 700;
        color: #202124;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #5f6368;
        margin-bottom: 20px;
    }

    /* Card */
    .card {
        background: #ffffff;
        border: 1px solid #e8eaed;
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 3px 12px rgba(60,64,67,0.10);
    }

    .card-title {
        font-size: 23px;
        font-weight: 700;
        color: #202124;
        margin-bottom: 8px;
    }

    .card-text {
        font-size: 16px;
        color: #5f6368;
        line-height: 1.7;
    }

    /* Feature cards */
    .feature-card {
        background: #f8fafd;
        border-radius: 18px;
        padding: 22px;
        text-align: center;
        border: 1px solid #e8eaed;
        height: 100%;
    }

    .feature-icon {
        font-size: 38px;
        margin-bottom: 10px;
    }

    .feature-title {
        font-size: 18px;
        font-weight: 700;
        color: #202124;
    }

    .feature-text {
        color: #5f6368;
        font-size: 14px;
        margin-top: 7px;
    }

    /* Result */
    .result-card {
        background: #f1f8e9;
        border: 1px solid #c5e1a5;
        border-radius: 20px;
        padding: 25px;
        margin-top: 20px;
    }

    .result-title {
        color: #2e7d32;
        font-size: 28px;
        font-weight: 700;
    }

    .result-confidence {
        font-size: 19px;
        color: #33691e;
        margin-top: 8px;
    }

    /* Disease */
    .disease-card {
        background: #fff8e1;
        border: 1px solid #ffe082;
        border-radius: 20px;
        padding: 25px;
        margin-top: 20px;
    }

    .disease-title {
        color: #e65100;
        font-size: 25px;
        font-weight: 700;
    }

    /* Healthy */
    .healthy-card {
        background: #e8f5e9;
        border: 1px solid #a5d6a7;
        border-radius: 20px;
        padding: 25px;
        margin-top: 20px;
    }

    .healthy-title {
        color: #2e7d32;
        font-size: 25px;
        font-weight: 700;
    }

    /* Section */
    .section-title {
        font-size: 25px;
        font-weight: 700;
        color: #202124;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    /* Info box */
    .info-box {
        background: #f8f9fa;
        border-radius: 16px;
        padding: 20px;
        margin: 12px 0;
        border-left: 5px solid #1a73e8;
    }

    .info-title {
        font-size: 18px;
        font-weight: 700;
        color: #202124;
    }

    .info-text {
        color: #5f6368;
        line-height: 1.7;
        margin-top: 5px;
    }

    /* Footer */
    .custom-footer {
        text-align: center;
        color: #9aa0a6;
        font-size: 14px;
        padding: 35px 0 10px 0;
    }

    /* ปุ่ม */
    .stButton > button {
        border-radius: 12px;
        height: 50px;
        font-size: 17px;
        font-weight: 600;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# ชื่อคลาส
# =========================================================

CLASS_NAMES = [
    "Banana Skipper Damage",
    "Black and Yellow Sigatoka",
    "Chewing insect damage on banana leaf",
    "Healthy Banana leaf",
    "Panama Wilt Disease"
]


# =========================================================
# ข้อมูลโรค
# =========================================================

DISEASE_INFO = {

    "Banana Skipper Damage": {

        "name": "หนอนม้วนใบกล้วย",
        "icon": "🐛",

        "short":
            "ความเสียหายของใบกล้วยที่เกิดจากหนอนหรือแมลง "
            "ซึ่งอาจทำให้ใบถูกกัดกินหรือม้วนตัว",

        "cause":
            "เกิดจากหนอนหรือแมลงที่เข้าทำลายใบกล้วย "
            "โดยตัวหนอนอาจกัดกินเนื้อใบและทำให้ใบเกิดความเสียหาย",

        "symptoms": [
            "ใบมีรอยกัดหรือถูกกิน",
            "ใบอาจถูกม้วนหรือพับ",
            "พบความเสียหายบริเวณขอบหรือแผ่นใบ",
            "อาจพบตัวหนอนหรือร่องรอยของแมลง"
        ],

        "management": [
            "ตรวจสอบใบกล้วยอย่างสม่ำเสมอ",
            "กำจัดหนอนหรือแมลงที่พบ",
            "ตัดส่วนของใบที่เสียหายมากออก",
            "ตรวจสอบต้นกล้วยบริเวณใกล้เคียง"
        ],

        "prevention": [
            "สำรวจแปลงปลูกเป็นประจำ",
            "กำจัดวัชพืชและเศษใบที่สะสม",
            "เฝ้าระวังการระบาดของแมลง"
        ]
    },


    "Black and Yellow Sigatoka": {

        "name": "โรคใบจุดดำและเหลือง",
        "icon": "🍂",

        "short":
            "โรคที่ทำให้ใบกล้วยเกิดจุดหรือแผลสีเหลือง "
            "และอาจพัฒนาเป็นสีน้ำตาลหรือดำ",

        "cause":
            "เกิดจากเชื้อรากลุ่มที่ทำให้เกิดโรค Sigatoka "
            "โดยสภาพอากาศชื้นและใบมีความชื้นเป็นเวลานาน "
            "สามารถเอื้อต่อการเกิดโรค",

        "symptoms": [
            "เริ่มพบจุดหรือขีดสีเหลืองบนใบ",
            "แผลอาจขยายและเปลี่ยนเป็นสีน้ำตาลหรือดำ",
            "ใบอาจมีพื้นที่สีเขียวลดลง",
            "หากรุนแรงใบอาจแห้งและเสื่อมสภาพ"
        ],

        "management": [
            "ตัดใบที่มีอาการรุนแรงออก",
            "เก็บใบที่ร่วงและนำออกจากบริเวณแปลง",
            "ลดความชื้นสะสมในแปลง",
            "เพิ่มการระบายอากาศระหว่างต้น"
        ],

        "prevention": [
            "ตรวจสอบใบกล้วยเป็นประจำ",
            "ไม่ปล่อยให้ใบที่เป็นโรคสะสมในแปลง",
            "จัดระยะปลูกให้เหมาะสม",
            "ดูแลพื้นที่ให้มีการระบายอากาศ"
        ]
    },


    "Chewing insect damage on banana leaf": {

        "name": "ความเสียหายจากแมลงกัดกินใบ",
        "icon": "🦗",

        "short":
            "ลักษณะความเสียหายของใบกล้วยที่เกิดจากแมลงกัดกิน "
            "ทำให้เกิดรูหรือรอยแหว่งบนใบ",

        "cause":
            "เกิดจากแมลงที่กัดกินเนื้อเยื่อของใบกล้วย "
            "ทำให้เกิดรู รอยแหว่ง หรือพื้นที่ใบที่ถูกทำลาย",

        "symptoms": [
            "พบรูบนใบ",
            "ขอบใบมีรอยแหว่ง",
            "ใบมีพื้นที่ถูกกัดกิน",
            "อาจพบร่องรอยหรือแมลงบริเวณใบ"
        ],

        "management": [
            "ตรวจสอบใบกล้วยและกำจัดแมลงที่พบ",
            "ตัดใบที่เสียหายมากออก",
            "ตรวจสอบใบอื่นของต้น",
            "ติดตามการระบาดอย่างต่อเนื่อง"
        ],

        "prevention": [
            "ตรวจแปลงเป็นประจำ",
            "กำจัดวัชพืชและแหล่งอาศัยของแมลง",
            "เฝ้าระวังแมลงในช่วงที่พบการระบาด"
        ]
    },


    "Healthy Banana leaf": {

        "name": "ใบกล้วยสุขภาพดี",
        "icon": "🌿",

        "short":
            "ใบกล้วยที่มีลักษณะโดยรวมปกติ "
            "และไม่พบลักษณะของโรคหรือความเสียหาย "
            "ในกลุ่มที่โมเดลสามารถจำแนกได้",

        "cause":
            "ไม่มีสาเหตุของโรคในกลุ่มที่ตรวจพบจากภาพ "
            "แต่สุขภาพของต้นยังขึ้นอยู่กับน้ำ ธาตุอาหาร "
            "สภาพแวดล้อม และการดูแล",

        "symptoms": [
            "ใบมีสีเขียวตามปกติ",
            "ไม่พบจุดโรคที่เด่นชัด",
            "ไม่พบรอยกัดกินจำนวนมาก",
            "ใบยังมีสภาพสมบูรณ์"
        ],

        "management": [
            "ดูแลน้ำให้เหมาะสม",
            "ดูแลธาตุอาหารของต้น",
            "ตรวจสอบใบอย่างสม่ำเสมอ",
            "เฝ้าระวังโรคและแมลง"
        ],

        "prevention": [
            "ดูแลแปลงให้สะอาด",
            "ตรวจใบและลำต้นเป็นประจำ",
            "ดูแลการให้น้ำและธาตุอาหารอย่างเหมาะสม",
            "สังเกตความผิดปกติตั้งแต่ระยะแรก"
        ]
    },


    "Panama Wilt Disease": {

        "name": "โรคตายพราย",
        "icon": "🦠",

        "short":
            "โรคที่เกี่ยวข้องกับเชื้อราที่อาศัยอยู่ในดิน "
            "และสามารถส่งผลต่อระบบท่อลำเลียงของต้นกล้วย",

        "cause":
            "เกี่ยวข้องกับเชื้อรา Fusarium oxysporum f. sp. cubense "
            "ซึ่งสามารถอาศัยอยู่ในดินและแพร่กระจายผ่านดิน "
            "น้ำ หรือวัสดุที่ปนเปื้อน",

        "symptoms": [
            "ใบล่างอาจเริ่มเหลือง",
            "ใบอาจเหี่ยวและแห้ง",
            "ต้นอาจเจริญเติบโตผิดปกติ",
            "อาการสามารถพัฒนาจากใบไปสู่ทั้งต้น"
        ],

        "management": [
            "แยกต้นที่สงสัยว่าเป็นโรค",
            "หลีกเลี่ยงการเคลื่อนย้ายดินจากบริเวณที่พบโรค",
            "ทำความสะอาดอุปกรณ์ที่สัมผัสดิน",
            "ปรึกษาผู้เชี่ยวชาญเพื่อยืนยันโรค"
        ],

        "prevention": [
            "ใช้วัสดุปลูกหรือหน่อพันธุ์ที่มีแหล่งที่มาน่าเชื่อถือ",
            "หลีกเลี่ยงการนำดินจากพื้นที่ที่พบโรคไปยังพื้นที่อื่น",
            "รักษาความสะอาดของเครื่องมือ",
            "ตรวจต้นกล้วยอย่างสม่ำเสมอ"
        ]
    }
}


# =========================================================
# โหลดโมเดล
# =========================================================

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = mobilenet_v2(weights=None)

    model.classifier[1] = nn.Linear(
        model.last_channel,
        len(CLASS_NAMES)
    )

    checkpoint = torch.load(
        "best_mobilenetv2_banana.pth",
        map_location=device,
        weights_only=False
    )

    # รองรับ checkpoint แบบที่ใช้ตอน train
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(device)

    model.eval()

    return model, device


# =========================================================
# เตรียมรูป
# =========================================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# โหลดโมเดล
# =========================================================

try:

    model, device = load_model()

except Exception as e:

    st.error("❌ ไม่สามารถโหลดโมเดลได้")

    st.info(
        "กรุณาตรวจสอบว่าไฟล์ "
        "`best_mobilenetv2_banana.pth` "
        "อยู่ในโฟลเดอร์เดียวกับ `app.py`"
    )

    st.code(str(e))

    st.stop()


# =========================================================
# เมนู
# =========================================================

page = st.radio(

    "เมนู",

    [
        "🏠 หน้าหลัก",
        "🔍 ทำนายโรค",
        "📚 ความรู้โรคใบกล้วย"
    ],

    horizontal=True

)

st.divider()


# =========================================================
# หน้าหลัก
# =========================================================

if page == "🏠 หน้าหลัก":

    # -----------------------------
    # Hero
    # -----------------------------

    st.markdown("""
    <div class="hero">

        <div class="banana-icon">
            🍌
        </div>

        <div class="hero-title">
            Banana Leaf AI
        </div>

        <div class="hero-subtitle">
            ระบบ AI สำหรับจำแนกโรคและความผิดปกติของใบกล้วย
        </div>

    </div>
    """, unsafe_allow_html=True)


    # -----------------------------
    # อัปโหลดรูป
    # -----------------------------

    st.markdown("""
    <div class="card">

        <div class="card-title">
            🔍 ตรวจสอบใบกล้วย
        </div>

        <div class="card-text">
            อัปโหลดภาพใบกล้วย แล้วให้ AI วิเคราะห์ลักษณะของใบ
            เพื่อช่วยจำแนกประเภทที่โมเดลเรียนรู้
        </div>

    </div>
    """, unsafe_allow_html=True)


    uploaded_home = st.file_uploader(

        "📷 เลือกรูปใบกล้วย",

        type=[
            "jpg",
            "jpeg",
            "png"
        ],

        key="home_uploader"

    )


    if uploaded_home is not None:

        home_image = Image.open(
            uploaded_home
        ).convert("RGB")


        st.image(
            home_image,
            caption="ภาพที่เลือก",
            use_container_width=True
        )


        if st.button(
            "🔮 ทำนายโรคใบกล้วย",
            use_container_width=True,
            key="home_predict"
        ):

            with st.spinner(
                "🍃 AI กำลังวิเคราะห์ใบกล้วย..."
            ):

                home_tensor = transform(
                    home_image
                ).unsqueeze(0).to(device)


                with torch.no_grad():

                    home_output = model(
                        home_tensor
                    )

                    home_probabilities = torch.softmax(
                        home_output,
                        dim=1
                    )

                    home_confidence, home_predicted = torch.max(
                        home_probabilities,
                        dim=1
                    )


            home_index = home_predicted.item()

            home_result = CLASS_NAMES[
                home_index
            ]

            home_confidence_percent = (
                home_confidence.item() * 100
            )

            home_info = DISEASE_INFO[
                home_result
            ]


            # -------------------------
            # ผลลัพธ์
            # -------------------------

            if home_result == "Healthy Banana leaf":

                st.markdown(f"""
                <div class="healthy-card">

                    <div class="healthy-title">
                        {home_info["icon"]}
                        ใบกล้วยสุขภาพดี
                    </div>

                    <p>
                        <b>ผลการทำนาย:</b>
                        {home_info["name"]}
                    </p>

                    <p>
                        <b>ความมั่นใจ:</b>
                        {home_confidence_percent:.2f}%
                    </p>

                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown(f"""
                <div class="disease-card">

                    <div class="disease-title">
                        {home_info["icon"]}
                        {home_info["name"]}
                    </div>

                    <p>
                        <b>ประเภท:</b>
                        {home_result}
                    </p>

                    <p>
                        <b>ความมั่นใจ:</b>
                        {home_confidence_percent:.2f}%
                    </p>

                </div>
                """, unsafe_allow_html=True)


            # -------------------------
            # รายละเอียด
            # -------------------------

            st.markdown(
                '<div class="section-title">📋 รายละเอียด</div>',
                unsafe_allow_html=True
            )

            st.write(
                home_info["short"]
            )


            # -------------------------
            # สาเหตุ
            # -------------------------

            st.markdown(
                '<div class="section-title">🔬 สาเหตุ</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="info-box">

                    <div class="info-title">
                        🔬 สาเหตุ
                    </div>

                    <div class="info-text">
                        {home_info["cause"]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            # -------------------------
            # อาการ
            # -------------------------

            st.markdown(
                '<div class="section-title">👀 ลักษณะอาการ</div>',
                unsafe_allow_html=True
            )

            for symptom in home_info["symptoms"]:

                st.write(
                    f"• {symptom}"
                )


            # -------------------------
            # วิธีดูแล
            # -------------------------

            st.markdown(
                '<div class="section-title">🩺 แนวทางการดูแล</div>',
                unsafe_allow_html=True
            )

            for management in home_info["management"]:

                st.write(
                    f"• {management}"
                )


            # -------------------------
            # การป้องกัน
            # -------------------------

            st.markdown(
                '<div class="section-title">🛡️ การป้องกัน</div>',
                unsafe_allow_html=True
            )

            for prevention in home_info["prevention"]:

                st.write(
                    f"• {prevention}"
                )


            # -------------------------
            # คะแนนทุกคลาส
            # -------------------------

            st.markdown(
                '<div class="section-title">📊 คะแนนการทำนายแต่ละประเภท</div>',
                unsafe_allow_html=True
            )


            for i, class_name in enumerate(
                CLASS_NAMES
            ):

                score = (
                    home_probabilities[0][i].item()
                    * 100
                )

                st.write(
                    f"**{class_name}** — "
                    f"{score:.2f}%"
                )

                st.progress(
                    min(score / 100, 1.0)
                )


    else:

        st.info(
            "🍃 กรุณาอัปโหลดรูปใบกล้วยเพื่อเริ่มการทำนาย"
        )


    # -----------------------------
    # Feature
    # -----------------------------

    st.markdown(
        '<div class="section-title">✨ ฟังก์ชันของระบบ</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown("""
        <div class="feature-card">

            <div class="feature-icon">
                🤖
            </div>

            <div class="feature-title">
                AI วิเคราะห์
            </div>

            <div class="feature-text">
                ใช้ MobileNetV2
                สำหรับจำแนกภาพใบกล้วย
            </div>

        </div>
        """, unsafe_allow_html=True)


    with col2:

        st.markdown("""
        <div class="feature-card">

            <div class="feature-icon">
                🔬
            </div>

            <div class="feature-title">
                วิเคราะห์โรค
            </div>

            <div class="feature-text">
                แสดงประเภทและ
                ระดับความมั่นใจ
            </div>

        </div>
        """, unsafe_allow_html=True)


    with col3:

        st.markdown("""
        <div class="feature-card">

            <div class="feature-icon">
                📚
            </div>

            <div class="feature-title">
                ความรู้โรค
            </div>

            <div class="feature-text">
                ดูสาเหตุ อาการ
                และแนวทางการดูแล
            </div>

        </div>
        """, unsafe_allow_html=True)


# =========================================================
# หน้าทำนาย
# =========================================================

elif page == "🔍 ทำนายโรค":

    st.markdown("""
    <div class="hero">

        <div class="banana-icon">
            🔍
        </div>

        <div class="hero-title">
            ทำนายโรคใบกล้วย
        </div>

        <div class="hero-subtitle">
            อัปโหลดภาพเพื่อให้ AI วิเคราะห์
        </div>

    </div>
    """, unsafe_allow_html=True)


    uploaded_file = st.file_uploader(

        "📷 เลือกรูปใบกล้วย",

        type=[
            "jpg",
            "jpeg",
            "png"
        ],

        key="prediction_uploader"

    )


    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        st.image(
            image,
            caption="รูปที่เลือก",
            use_container_width=True
        )


        if st.button(
            "🔮 เริ่มทำนาย",
            use_container_width=True,
            key="prediction_button"
        ):

            with st.spinner(
                "🤖 AI กำลังวิเคราะห์..."
            ):

                image_tensor = transform(
                    image
                ).unsqueeze(0).to(device)


                with torch.no_grad():

                    output = model(
                        image_tensor
                    )

                    probabilities = torch.softmax(
                        output,
                        dim=1
                    )

                    confidence, predicted = torch.max(
                        probabilities,
                        dim=1
                    )


            index = predicted.item()

            result = CLASS_NAMES[
                index
            ]

            confidence_percent = (
                confidence.item() * 100
            )

            info = DISEASE_INFO[
                result
            ]


            # -------------------------
            # ผลลัพธ์
            # -------------------------

            if result == "Healthy Banana leaf":

                st.success(
                    "🌿 ผลการทำนาย: ใบกล้วยสุขภาพดี"
                )

            else:

                st.warning(
                    "⚠️ พบลักษณะที่โมเดลจำแนกเป็นความผิดปกติ"
                )


            st.markdown(f"""
            <div class="result-card">

                <div class="result-title">
                    {info["icon"]}
                    {info["name"]}
                </div>

                <div class="result-confidence">
                    ความมั่นใจ
                    {confidence_percent:.2f}%
                </div>

            </div>
            """, unsafe_allow_html=True)


            # -------------------------
            # รายละเอียด
            # -------------------------

            st.subheader(
                "📋 รายละเอียด"
            )

            st.write(
                info["short"]
            )


            # -------------------------
            # สาเหตุ
            # -------------------------

            st.subheader(
                "🔬 สาเหตุ"
            )

            st.write(
                info["cause"]
            )


            # -------------------------
            # อาการ
            # -------------------------

            st.subheader(
                "👀 อาการที่พบ"
            )

            for symptom in info["symptoms"]:

                st.write(
                    f"• {symptom}"
                )


            # -------------------------
            # ดูแล
            # -------------------------

            st.subheader(
                "🩺 แนวทางการดูแล"
            )

            for item in info["management"]:

                st.write(
                    f"• {item}"
                )


            # -------------------------
            # ป้องกัน
            # -------------------------

            st.subheader(
                "🛡️ การป้องกัน"
            )

            for item in info["prevention"]:

                st.write(
                    f"• {item}"
                )


            # -------------------------
            # คะแนนทุกคลาส
            # -------------------------

            st.subheader(
                "📊 คะแนนแต่ละประเภท"
            )


            for i, class_name in enumerate(
                CLASS_NAMES
            ):

                score = (
                    probabilities[0][i].item()
                    * 100
                )

                st.write(
                    f"**{class_name}**: "
                    f"{score:.2f}%"
                )

                st.progress(
                    min(score / 100, 1.0)
                )


    else:

        st.info(
            "🍌 กรุณาอัปโหลดภาพใบกล้วย"
        )


# =========================================================
# หน้าความรู้
# =========================================================

elif page == "📚 ความรู้โรคใบกล้วย":

    st.markdown("""
    <div class="hero">

        <div class="banana-icon">
            📚
        </div>

        <div class="hero-title">
            ความรู้โรคใบกล้วย
        </div>

        <div class="hero-subtitle">
            ข้อมูลเกี่ยวกับสาเหตุ อาการ การดูแล และการป้องกัน
        </div>

    </div>
    """, unsafe_allow_html=True)


    # -----------------------------------------------------
    # เลือกโรค
    # -----------------------------------------------------

    selected_disease = st.selectbox(

        "🍃 เลือกประเภท",

        CLASS_NAMES

    )


    info = DISEASE_INFO[
        selected_disease
    ]


    # -----------------------------------------------------
    # ชื่อ
    # -----------------------------------------------------

    st.markdown(f"""
    <div class="result-card">

        <div class="result-title">
            {info["icon"]}
            {info["name"]}
        </div>

        <p>
            <b>ชื่อประเภท:</b>
            {selected_disease}
        </p>

    </div>
    """, unsafe_allow_html=True)


    # -----------------------------------------------------
    # รายละเอียด
    # -----------------------------------------------------

    st.subheader(
        "📋 รายละเอียด"
    )

    st.write(
        info["short"]
    )


    # -----------------------------------------------------
    # สาเหตุ
    # -----------------------------------------------------

    st.subheader(
        "🔬 สาเหตุ"
    )

    st.write(
        info["cause"]
    )


    # -----------------------------------------------------
    # อาการ
    # -----------------------------------------------------

    st.subheader(
        "👀 อาการที่พบ"
    )

    for symptom in info["symptoms"]:

        st.write(
            f"• {symptom}"
        )


    # -----------------------------------------------------
    # การดูแล
    # -----------------------------------------------------

    st.subheader(
        "🩺 แนวทางการดูแล"
    )

    for item in info["management"]:

        st.write(
            f"• {item}"
        )


    # -----------------------------------------------------
    # การป้องกัน
    # -----------------------------------------------------

    st.subheader(
        "🛡️ การป้องกัน"
    )

    for item in info["prevention"]:

        st.write(
            f"• {item}"
        )


# =========================================================
# Footer
# =========================================================

st.markdown("""
<div class="custom-footer">

    🍌 Banana Leaf AI
    <br>
    ระบบจำแนกโรคใบกล้วยด้วยปัญญาประดิษฐ์
    <br><br>
    Powered by MobileNetV2

</div>
""", unsafe_allow_html=True)
