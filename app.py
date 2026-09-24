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
# CSS ปรับแต่งหน้าตา
# =========================================================

st.markdown("""
<style>

    /* =========================
       พื้นฐาน
       ========================= */

    .stApp {
        background-color: #ffffff;
    }

    .main .block-container {
        max-width: 900px;
        padding-top: 35px;
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


    /* =========================
       Header
       ========================= */

    .logo {
        text-align: center;
        font-size: 58px;
        margin-bottom: 0px;
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        color: #202124;
        letter-spacing: -1px;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #5f6368;
        margin-bottom: 35px;
    }


    /* =========================
       กล่องหัวข้อ
       ========================= */

    .section-title {
        font-size: 23px;
        font-weight: 600;
        color: #202124;
        margin-top: 20px;
        margin-bottom: 14px;
    }


    /* =========================
       Upload Card
       ========================= */

    .upload-card {
        background: #ffffff;
        border: 1px solid #dadce0;
        border-radius: 24px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(60,64,67,0.12);
        margin-bottom: 20px;
    }


    /* =========================
       Image Card
       ========================= */

    .image-card {
        background: #f8f9fa;
        border-radius: 20px;
        padding: 15px;
        margin-top: 15px;
        margin-bottom: 20px;
    }


    /* =========================
       Result
       ========================= */

    .result-card {
        background: #f8fbf8;
        border: 1px solid #d7ead9;
        border-radius: 24px;
        padding: 28px;
        margin-top: 25px;
        margin-bottom: 20px;
        text-align: center;
    }

    .result-icon {
        font-size: 48px;
        margin-bottom: 8px;
    }

    .result-title {
        font-size: 30px;
        font-weight: 700;
        color: #188038;
        margin-bottom: 8px;
    }

    .result-type {
        font-size: 15px;
        color: #5f6368;
    }


    /* =========================
       Confidence
       ========================= */

    .confidence-card {
        background: #ffffff;
        border: 1px solid #dadce0;
        border-radius: 20px;
        padding: 22px;
        margin-top: 15px;
        margin-bottom: 20px;
        text-align: center;
    }

    .confidence-label {
        font-size: 15px;
        color: #5f6368;
    }

    .confidence-value {
        font-size: 36px;
        font-weight: 700;
        color: #1a73e8;
        margin-top: 5px;
    }


    /* =========================
       Info Card
       ========================= */

    .info-card {
        background: #ffffff;
        border: 1px solid #dadce0;
        border-radius: 20px;
        padding: 22px;
        margin-top: 18px;
        box-shadow: 0 1px 5px rgba(60,64,67,0.08);
    }

    .info-title {
        font-size: 20px;
        font-weight: 600;
        color: #202124;
        margin-bottom: 10px;
    }

    .info-text {
        color: #5f6368;
        font-size: 16px;
        line-height: 1.7;
    }


    /* =========================
       ปุ่ม
       ========================= */

    .stButton > button {
        width: 100%;
        height: 52px;
        border-radius: 26px;
        border: none;
        background: #1a73e8;
        color: white;
        font-size: 17px;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(26,115,232,0.25);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: #1557b0;
        color: white;
        transform: translateY(-1px);
    }


    /* =========================
       File uploader
       ========================= */

    [data-testid="stFileUploader"] {
        background: #f8f9fa;
        border-radius: 18px;
        padding: 10px;
    }


    /* =========================
       Footer
       ========================= */

    .footer {
        text-align: center;
        color: #9aa0a6;
        font-size: 13px;
        margin-top: 45px;
        padding-top: 20px;
        border-top: 1px solid #eeeeee;
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
        "detail": "พบความเสียหายบริเวณใบจากแมลงหรือหนอน",
        "care": [
            "ตรวจสอบใบกล้วยอย่างสม่ำเสมอ",
            "กำจัดส่วนของใบที่เสียหายมาก",
            "ตรวจสอบต้นกล้วยบริเวณใกล้เคียง"
        ]
    },

    "Black and Yellow Sigatoka": {
        "name": "โรคใบจุดดำและเหลือง",
        "icon": "🍂",
        "detail": "พบจุดหรือแผลสีดำและเหลืองบนใบ",
        "care": [
            "ตัดใบที่มีอาการออก",
            "เก็บใบที่ร่วงออกจากบริเวณแปลง",
            "ดูแลบริเวณปลูกให้มีการระบายอากาศ"
        ]
    },

    "Chewing insect damage on banana leaf": {
        "name": "ความเสียหายจากแมลงกัดกินใบ",
        "icon": "🐜",
        "detail": "พบรอยกัดหรือรูบนใบกล้วย",
        "care": [
            "ตรวจสอบใบและกำจัดแมลงที่พบ",
            "ตัดใบที่เสียหายมากออก",
            "ตรวจสอบใบอื่นอย่างสม่ำเสมอ"
        ]
    },

    "Healthy Banana leaf": {
        "name": "ใบกล้วยสุขภาพดี",
        "icon": "🌿",
        "detail": "ไม่พบลักษณะของโรคในกลุ่มที่โมเดลจำแนก",
        "care": [
            "ดูแลน้ำและธาตุอาหารให้เหมาะสม",
            "ตรวจสอบใบอย่างสม่ำเสมอ",
            "เฝ้าระวังโรคและแมลง"
        ]
    },

    "Panama Wilt Disease": {
        "name": "โรคตายพราย",
        "icon": "⚠️",
        "detail": "พบลักษณะที่อาจเกี่ยวข้องกับโรคตายพราย",
        "care": [
            "แยกต้นที่สงสัยว่าเป็นโรค",
            "หลีกเลี่ยงการเคลื่อนย้ายดินจากบริเวณที่พบโรค",
            "ปรึกษาผู้เชี่ยวชาญเพื่อยืนยันโรค"
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

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

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

    st.error("ไม่สามารถโหลดโมเดลได้")

    st.write(
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
    'ระบบจำแนกโรคใบกล้วยด้วยปัญญาประดิษฐ์'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# UPLOAD
# =========================================================

st.markdown(
    '<div class="section-title">🔍 ตรวจสอบใบกล้วย</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="upload-card">',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "เลือกรูปใบกล้วย",
    type=["jpg", "jpeg", "png"]
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# ทำนาย
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.markdown(
        '<div class="image-card">',
        unsafe_allow_html=True
    )

    st.image(
        image,
        caption="รูปภาพที่เลือก",
        use_container_width=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # =========================
    # ปุ่มทำนาย
    # =========================

    if st.button(
        "🔮  ทำนายโรคใบกล้วย",
        use_container_width=True
    ):

        with st.spinner("กำลังวิเคราะห์รูปภาพ..."):

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


        # =========================
        # ผลลัพธ์
        # =========================

        index = predicted.item()

        result = CLASS_NAMES[index]

        confidence_percent = (
            confidence.item() * 100
        )

        info = DISEASE_INFO[result]


        # =========================
        # Result Card
        # =========================

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="result-icon">'
            f'{info["icon"]}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="result-title">'
            f'{info["name"]}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="result-type">'
            f'{result}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # =========================
        # สถานะ
        # =========================

        if result == "Healthy Banana leaf":

            st.success(
                "🌿 ไม่พบลักษณะของโรคจากประเภทที่โมเดลจำแนก"
            )

        else:

            st.warning(
                "⚠️ พบลักษณะที่อาจเกี่ยวข้องกับโรคหรือความเสียหาย"
            )


        # =========================
        # Confidence
        # =========================

        st.markdown(
            '<div class="confidence-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="confidence-label">'
            'ความมั่นใจของโมเดล'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="confidence-value">'
            f'{confidence_percent:.2f}%'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # =========================
        # รายละเอียด
        # =========================

        st.markdown(
            '<div class="info-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="info-title">'
            '📋 รายละเอียด'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="info-text">'
            f'{info["detail"]}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # =========================
        # แนวทางการดูแล
        # =========================

        st.markdown(
            '<div class="info-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="info-title">'
            '🩺 แนวทางการดูแล'
            '</div>',
            unsafe_allow_html=True
        )

        for item in info["care"]:

            st.markdown(
                f'<div class="info-text">'
                f'✓ {item}'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # =========================
        # คะแนนทุกคลาส
        # =========================

        st.markdown(
            '<div class="section-title">'
            '📊 คะแนนการจำแนก'
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
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    '🍌 Banana Leaf AI '
    '• ระบบจำแนกโรคใบกล้วยด้วย AI'
    '</div>',
    unsafe_allow_html=True
)
