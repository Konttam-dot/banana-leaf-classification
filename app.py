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
# CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: #ffffff;
}

.main .block-container {
    max-width: 900px;
    padding-top: 25px;
    padding-bottom: 50px;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}


/* HEADER */

.logo {
    text-align: center;
    font-size: 55px;
}

.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
    color: #202124;
}

.subtitle {
    text-align: center;
    color: #5f6368;
    font-size: 16px;
    margin-bottom: 25px;
}


/* CARD */

.card {
    background: white;
    border: 1px solid #dadce0;
    border-radius: 22px;
    padding: 24px;
    margin: 15px 0;
    box-shadow: 0 2px 8px rgba(60,64,67,.10);
}

.card-title {
    font-size: 22px;
    font-weight: 700;
    color: #202124;
    margin-bottom: 10px;
}

.card-text {
    color: #5f6368;
    line-height: 1.7;
    font-size: 16px;
}


/* DISEASE CARD */

.disease-card {
    background: #ffffff;
    border: 1px solid #dadce0;
    border-radius: 20px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 2px 6px rgba(60,64,67,.08);
}

.disease-icon {
    font-size: 38px;
}

.disease-name {
    font-size: 20px;
    font-weight: 700;
    color: #202124;
}

.disease-desc {
    color: #5f6368;
    font-size: 14px;
}


/* RESULT */

.result-card {
    background: #f6fbf7;
    border: 1px solid #cde5d2;
    border-radius: 24px;
    padding: 28px;
    text-align: center;
    margin-top: 20px;
}

.result-icon {
    font-size: 50px;
}

.result-title {
    color: #188038;
    font-size: 30px;
    font-weight: 700;
}


/* CONFIDENCE */

.confidence-card {
    background: white;
    border: 1px solid #dadce0;
    border-radius: 20px;
    padding: 20px;
    text-align: center;
    margin-top: 15px;
}

.confidence-value {
    color: #1a73e8;
    font-size: 36px;
    font-weight: 700;
}


/* BUTTON */

.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 25px;
    border: none;
    background: #1a73e8;
    color: white;
    font-size: 16px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #1557b0;
    color: white;
}


/* FOOTER */

.footer {
    text-align: center;
    color: #9aa0a6;
    font-size: 13px;
    margin-top: 45px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# CLASS
# =========================================================

CLASS_NAMES = [
    "Banana Skipper Damage",
    "Black and Yellow Sigatoka",
    "Chewing insect damage on banana leaf",
    "Healthy Banana leaf",
    "Panama Wilt Disease"
]


# =========================================================
# ฐานข้อมูลความรู้
# =========================================================

DISEASE_INFO = {

    "Banana Skipper Damage": {

        "name": "หนอนม้วนใบกล้วย",

        "icon": "🐛",

        "short": "ความเสียหายจากหนอนที่ทำลายใบกล้วย",

        "cause":
        "เกิดจากหนอนของผีเสื้อหนอนม้วนใบกล้วย "
        "ที่เข้าทำลายใบและอาจทำให้ใบเสียหาย",

        "symptoms": [
            "ใบมีรอยกัดหรือถูกทำลาย",
            "ใบอาจมีลักษณะม้วนหรือพับ",
            "พบส่วนของใบที่ถูกกัดกิน"
        ],

        "management": [
            "ตรวจสอบใบกล้วยอย่างสม่ำเสมอ",
            "เก็บหรือกำจัดส่วนของใบที่เสียหายมาก",
            "ตรวจสอบบริเวณต้นกล้วยโดยรอบ",
            "ติดตามการกลับมาของแมลงอย่างต่อเนื่อง"
        ],

        "prevention": [
            "สำรวจใบเป็นประจำ",
            "รักษาความสะอาดบริเวณแปลง",
            "กำจัดใบที่เสียหายมากอย่างเหมาะสม"
        ]
    },


    "Black and Yellow Sigatoka": {

        "name": "โรคใบจุดดำและเหลือง",

        "icon": "🍂",

        "short": "โรคใบจุดที่ทำให้เกิดแผลสีเหลืองถึงดำบนใบ",

        "cause":
        "โรค Sigatoka เกี่ยวข้องกับเชื้อราที่เข้าทำลายใบกล้วย "
        "และสามารถแพร่กระจายได้ดีในสภาพที่มีความชื้น",

        "symptoms": [
            "เริ่มพบจุดหรือรอยแผลบนใบ",
            "แผลอาจเปลี่ยนเป็นสีน้ำตาลหรือดำ",
            "บริเวณใบที่เป็นโรคอาจเสื่อมสภาพมากขึ้น"
        ],

        "management": [
            "ตัดใบที่มีอาการมากออก",
            "เก็บใบที่เป็นโรคออกจากบริเวณแปลง",
            "ลดสภาพแวดล้อมที่มีความชื้นสะสม",
            "ดูแลการระบายน้ำและการถ่ายเทอากาศ"
        ],

        "prevention": [
            "ตรวจสอบใบอย่างสม่ำเสมอ",
            "หลีกเลี่ยงการปลูกที่หนาแน่นเกินไป",
            "รักษาความสะอาดของแปลง",
            "ใช้แนวทางควบคุมโรคตามคำแนะนำของผู้เชี่ยวชาญ"
        ]
    },


    "Chewing insect damage on banana leaf": {

        "name": "ความเสียหายจากแมลงกัดกินใบ",

        "icon": "🐜",

        "short": "ใบมีรอยกัดหรือรูจากแมลง",

        "cause":
        "เกิดจากแมลงที่เข้ากัดกินเนื้อเยื่อของใบ "
        "ทำให้เกิดรูหรือรอยแหว่ง",

        "symptoms": [
            "ใบมีรู",
            "ขอบใบมีรอยแหว่ง",
            "พบร่องรอยการกัดกินบนใบ"
        ],

        "management": [
            "ตรวจสอบใต้ใบและบริเวณรอบต้น",
            "กำจัดส่วนของใบที่เสียหายมาก",
            "ติดตามจำนวนแมลง",
            "ใช้วิธีควบคุมแมลงตามคำแนะนำที่เหมาะสม"
        ],

        "prevention": [
            "ตรวจใบเป็นประจำ",
            "กำจัดเศษใบและวัสดุที่เป็นแหล่งสะสมของแมลง",
            "ดูแลต้นกล้วยให้แข็งแรง"
        ]
    },


    "Healthy Banana leaf": {

        "name": "ใบกล้วยสุขภาพดี",

        "icon": "🌿",

        "short": "ไม่พบลักษณะของโรคในกลุ่มที่โมเดลจำแนก",

        "cause":
        "เป็นลักษณะของใบที่ไม่พบความผิดปกติที่อยู่ในประเภทที่ระบบสามารถจำแนกได้",

        "symptoms": [
            "ใบมีสีเขียวตามลักษณะปกติ",
            "ไม่พบจุดโรคที่เด่นชัด",
            "ไม่พบร่องรอยความเสียหายรุนแรง"
        ],

        "management": [
            "ดูแลน้ำให้เหมาะสม",
            "ดูแลธาตุอาหารของต้นกล้วย",
            "ตรวจสอบใบอย่างสม่ำเสมอ"
        ],

        "prevention": [
            "สำรวจใบเป็นประจำ",
            "รักษาความสะอาดของแปลง",
            "เฝ้าระวังโรคและแมลง"
        ]
    },


    "Panama Wilt Disease": {

        "name": "โรคตายพราย",

        "icon": "⚠️",

        "short": "โรคเหี่ยวที่เกี่ยวข้องกับเชื้อ Fusarium",

        "cause":
        "โรคตายพรายหรือ Fusarium wilt เกิดจากเชื้อในกลุ่ม "
        "Fusarium oxysporum f. sp. cubense "
        "ซึ่งสามารถอยู่ในดินและส่งผลต่อระบบท่อลำเลียงของพืช",

        "symptoms": [
            "ใบอาจเหลืองและเหี่ยว",
            "ใบแก่สามารถแสดงอาการก่อน",
            "ต้นอาจทรุดโทรมและแสดงอาการเหี่ยว"
        ],

        "management": [
            "แยกต้นที่สงสัยว่าเป็นโรค",
            "หลีกเลี่ยงการเคลื่อนย้ายดินจากพื้นที่ที่พบโรค",
            "ใช้วัสดุปลูกที่สะอาด",
            "พิจารณาพันธุ์ที่มีความต้านทานตามพื้นที่",
            "ปรึกษาผู้เชี่ยวชาญเพื่อยืนยันโรค"
        ],

        "prevention": [
            "ใช้ต้นพันธุ์หรือวัสดุปลูกที่สะอาด",
            "ระวังการเคลื่อนย้ายดินจากพื้นที่ปนเปื้อน",
            "จัดการแปลงอย่างเป็นระบบ",
            "เลือกพันธุ์ที่เหมาะสมกับพื้นที่"
        ]
    }
}


# =========================================================
# LOAD MODEL
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

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, device


# =========================================================
# IMAGE TRANSFORM
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
# LOAD MODEL
# =========================================================

try:

    model, device = load_model()

except Exception:

    st.error("ไม่สามารถโหลดโมเดลได้")

    st.info(
        "ตรวจสอบว่าไฟล์ "
        "best_mobilenetv2_banana.pth "
        "อยู่ใน Repository เดียวกับ app.py"
    )

    st.stop()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="logo">🍌</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">Banana Leaf AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'ผู้ช่วยตรวจสอบและเรียนรู้เกี่ยวกับโรคใบกล้วย'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# MENU
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


# =========================================================
# HOME
# =========================================================

if page == "🏠 หน้าหลัก":

    # -----------------------------------------
    # Hero
    # -----------------------------------------

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:25px 10px 15px 10px;
        ">

            <div style="
                font-size:65px;
                margin-bottom:5px;
            ">
                🍌
            </div>

            <div style="
                font-size:38px;
                font-weight:700;
                color:#202124;
            ">
                Banana Leaf AI
            </div>

            <div style="
                font-size:17px;
                color:#5f6368;
                margin-top:8px;
            ">
                ผู้ช่วยตรวจสอบและเรียนรู้เกี่ยวกับโรคใบกล้วย
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # -----------------------------------------
    # Main prediction card
    # -----------------------------------------

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                🔍 ตรวจสอบใบกล้วย
            </div>

            <div class="card-text">
                อัปโหลดรูปใบกล้วยเพื่อให้ AI
                วิเคราะห์และจำแนกประเภทของภาพ
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    uploaded_home = st.file_uploader(
        "เลือกรูปใบกล้วยจากอุปกรณ์",
        type=["jpg", "jpeg", "png"],
        key="home_uploader"
    )


    # -----------------------------------------
    # ถ้ามีรูป
    # -----------------------------------------

    if uploaded_home is not None:

        home_image = Image.open(
            uploaded_home
        ).convert("RGB")


        st.markdown(
            """
            <div style="
                text-align:center;
                color:#5f6368;
                margin:15px 0 8px 0;
            ">
                📷 รูปภาพที่เลือก
            </div>
            """,
            unsafe_allow_html=True
        )


        st.image(
            home_image,
            use_container_width=True
        )


        # -----------------------------------------
        # ปุ่มทำนาย
        # -----------------------------------------

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


            # -----------------------------------------
            # Result
            # -----------------------------------------

            st.markdown(
                f"""
                <div class="result-card">

                    <div class="result-icon">
                        {home_info["icon"]}
                    </div>

                    <div class="result-title">
                        {home_info["name"]}
                    </div>

                    <div class="result-type">
                        ผลการวิเคราะห์จาก Banana Leaf AI
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            # -----------------------------------------
            # Confidence
            # -----------------------------------------

            st.markdown(
                f"""
                <div class="confidence-card">

                    <div style="
                        color:#5f6368;
                        font-size:15px;
                    ">
                        ความมั่นใจของโมเดล
                    </div>

                    <div class="confidence-value">
                        {home_confidence_percent:.2f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            # -----------------------------------------
            # Status
            # -----------------------------------------

            if home_result == "Healthy Banana leaf":

                st.success(
                    "🌿 ใบกล้วยมีลักษณะสุขภาพดี "
                    "ตามประเภทที่โมเดลสามารถจำแนกได้"
                )

            else:

                st.warning(
                    "⚠️ ระบบพบลักษณะที่อาจเกี่ยวข้องกับ "
                    "โรคหรือความเสียหาย"
                )


            # -----------------------------------------
            # Information
            # -----------------------------------------

            st.markdown(
                """
                <div class="card">

                    <div class="card-title">
                        📖 ข้อมูลเบื้องต้น
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.write(
                home_info["short"]
            )


            # -----------------------------------------
            # Cause
            # -----------------------------------------

            st.markdown(
                """
                <div class="card">

                    <div class="card-title">
                        🔬 สาเหตุ
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.write(
                home_info["cause"]
            )


            # -----------------------------------------
            # Symptoms
            # -----------------------------------------

            st.markdown(
                """
                <div class="card">

                    <div class="card-title">
                        👀 ลักษณะอาการ
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            for item in home_info["symptoms"]:

                st.write(
                    f"✓ {item}"
                )


            # -----------------------------------------
            # Management
            # -----------------------------------------

            st.markdown(
                """
                <div class="card">

                    <div class="card-title">
                        🛠️ วิธีแก้ไข / จัดการ
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            for item in home_info["management"]:

                st.write(
                    f"✓ {item}"
                )


            # -----------------------------------------
            # Prevention
            # -----------------------------------------

            st.markdown(
                """
                <div class="card">

                    <div class="card-title">
                        🛡️ วิธีป้องกัน
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            for item in home_info["prevention"]:

                st.write(
                    f"✓ {item}"
                )


            # -----------------------------------------
            # All scores
            # -----------------------------------------

            st.markdown(
                """
                <div class="section-title">
                    📊 คะแนนการจำแนก
                </div>
                """,
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
                    f"**{DISEASE_INFO[class_name]['name']}** "
                    f"— {score:.2f}%"
                )

                st.progress(
                    min(score / 100, 1.0)
                )


    # -----------------------------------------
    # ยังไม่มีรูป
    # -----------------------------------------

    else:

        st.markdown(
            """
            <div style="
                text-align:center;
                color:#9aa0a6;
                padding:20px;
            ">
                📷 เลือกรูปใบกล้วยเพื่อเริ่มการวิเคราะห์
            </div>
            """,
            unsafe_allow_html=True
        )


    # -----------------------------------------
    # Features
    # -----------------------------------------

    st.markdown(
        """
        <div class="section-title">
            ✨ ฟังก์ชันของระบบ
        </div>
        """,
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    🔍 AI Prediction
                </div>

                <div class="card-text">
                    วิเคราะห์รูปใบกล้วยด้วย
                    MobileNetV2
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    📚 Knowledge
                </div>

                <div class="card-text">
                    อ่านข้อมูลโรค สาเหตุ
                    อาการ วิธีแก้ไข และการป้องกัน
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # -----------------------------------------
    # Footer
    # -----------------------------------------

    st.markdown(
        """
        <div class="footer">
            🍌 Banana Leaf AI
            <br>
            ระบบจำแนกโรคใบกล้วยด้วยปัญญาประดิษฐ์
        </div>
        """,
        unsafe_allow_html=True
    )

elif page == "🔍 ทำนายโรค":

    st.markdown(
        '<div class="section-title">'
        '🔍 ตรวจสอบใบกล้วย'
        '</div>',
        unsafe_allow_html=True
    )


    uploaded_file = st.file_uploader(
        "เลือกรูปใบกล้วย",
        type=["jpg", "jpeg", "png"]
    )


    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        st.image(
            image,
            caption="รูปภาพที่เลือก",
            use_container_width=True
        )


        if st.button(
            "🔮 ทำนายโรคใบกล้วย",
            use_container_width=True
        ):

            with st.spinner(
                "AI กำลังวิเคราะห์..."
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

            result = CLASS_NAMES[index]

            confidence_percent = (
                confidence.item() * 100
            )

            info = DISEASE_INFO[result]


            # =========================
            # RESULT
            # =========================

            st.markdown(
                f"""
                <div class="result-card">

                <div class="result-icon">
                {info["icon"]}
                </div>

                <div class="result-title">
                {info["name"]}
                </div>

                <div>
                {result}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                f"""
                <div class="confidence-card">

                ความมั่นใจของโมเดล

                <div class="confidence-value">
                {confidence_percent:.2f}%
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            # =========================
            # QUICK INFO
            # =========================

            st.markdown("""
            <div class="card">

            <div class="card-title">
            📖 ข้อมูลเบื้องต้น
            </div>

            """, unsafe_allow_html=True)

            st.write(info["short"])

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


            # =========================
            # CAUSE
            # =========================

            st.markdown("""
            <div class="card">

            <div class="card-title">
            🔬 สาเหตุ
            </div>

            """, unsafe_allow_html=True)

            st.write(info["cause"])

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


            # =========================
            # SYMPTOMS
            # =========================

            st.markdown("""
            <div class="card">

            <div class="card-title">
            👀 ลักษณะอาการ
            </div>

            """, unsafe_allow_html=True)

            for item in info["symptoms"]:

                st.write(
                    f"✓ {item}"
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


            # =========================
            # MANAGEMENT
            # =========================

            st.markdown("""
            <div class="card">

            <div class="card-title">
            🛠️ วิธีแก้ไข / จัดการ
            </div>

            """, unsafe_allow_html=True)

            for item in info["management"]:

                st.write(
                    f"✓ {item}"
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


            # =========================
            # PREVENTION
            # =========================

            st.markdown("""
            <div class="card">

            <div class="card-title">
            🛡️ วิธีป้องกัน
            </div>

            """, unsafe_allow_html=True)

            for item in info["prevention"]:

                st.write(
                    f"✓ {item}"
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


            # =========================
            # ALL SCORES
            # =========================

            st.markdown(
                '<div class="section-title">'
                '📊 คะแนนแต่ละประเภท'
                '</div>',
                unsafe_allow_html=True
            )


            for i, class_name in enumerate(
                CLASS_NAMES
            ):

                score = (
                    probabilities[0][i].item()
                    * 100
                )

                st.write(
                    f"**{DISEASE_INFO[class_name]['name']}** "
                    f"— {score:.2f}%"
                )

                st.progress(
                    min(score / 100, 1.0)
                )


# =========================================================
# KNOWLEDGE PAGE
# =========================================================

elif page == "📚 ความรู้โรคใบกล้วย":

    st.markdown(
        '<div class="section-title">'
        '📚 ศูนย์ความรู้โรคใบกล้วย'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "เลือกหัวข้อที่ต้องการอ่านข้อมูล"
    )


    selected = st.selectbox(
        "เลือกโรค",
        CLASS_NAMES,
        format_func=lambda x:
            f"{DISEASE_INFO[x]['icon']} "
            f"{DISEASE_INFO[x]['name']}"
    )


    info = DISEASE_INFO[selected]


    # =========================
    # TITLE
    # =========================

    st.markdown(
        f"""
        <div class="result-card">

        <div class="result-icon">
        {info["icon"]}
        </div>

        <div class="result-title">
        {info["name"]}
        </div>

        <div>
        {info["short"]}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # =========================
    # CAUSE
    # =========================

    with st.expander(
        "🔬 สาเหตุ",
        expanded=True
    ):

        st.write(
            info["cause"]
        )


    # =========================
    # SYMPTOMS
    # =========================

    with st.expander(
        "👀 ลักษณะอาการ",
        expanded=True
    ):

        for item in info["symptoms"]:

            st.write(
                f"✓ {item}"
            )


    # =========================
    # MANAGEMENT
    # =========================

    with st.expander(
        "🛠️ วิธีแก้ไข / จัดการ",
        expanded=True
    ):

        for item in info["management"]:

            st.write(
                f"✓ {item}"
            )


    # =========================
    # PREVENTION
    # =========================

    with st.expander(
        "🛡️ วิธีป้องกัน"
    ):

        for item in info["prevention"]:

            st.write(
                f"✓ {item}"
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
    🍌 Banana Leaf AI
    • ระบบจำแนกโรคใบกล้วยด้วย AI
    </div>
    """,
    unsafe_allow_html=True
)
