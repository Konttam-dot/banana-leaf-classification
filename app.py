import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision import transforms
from PIL import Image


# =========================
# ตั้งค่าหน้าเว็บ
# =========================

st.set_page_config(
    page_title="Banana Leaf Disease Detection",
    page_icon="🍌",
    layout="centered"
)

st.title("🍌 Banana Leaf Disease Detection")
st.write("ระบบจำแนกโรคใบกล้วยด้วย AI")

st.divider()


# =========================
# ชื่อคลาส
# =========================

CLASS_NAMES = [
    "Banana Skipper Damage",
    "Black and Yellow Sigatoka",
    "Chewing insect damage on banana leaf",
    "Healthy Banana leaf",
    "Panama Wilt Disease"
]


# =========================
# ข้อมูลการดูแล
# =========================

DISEASE_INFO = {

    "Banana Skipper Damage": {
        "name": "หนอนม้วนใบกล้วย",
        "detail": "พบความเสียหายบริเวณใบจากแมลงหรือหนอน",
        "care": [
            "ตรวจสอบใบกล้วยอย่างสม่ำเสมอ",
            "กำจัดส่วนของใบที่เสียหายมาก",
            "ตรวจสอบต้นกล้วยบริเวณใกล้เคียง"
        ]
    },

    "Black and Yellow Sigatoka": {
        "name": "โรคใบจุดดำและเหลือง",
        "detail": "พบจุดหรือแผลสีดำและเหลืองบนใบ",
        "care": [
            "ตัดใบที่มีอาการออก",
            "เก็บใบที่ร่วงออกจากบริเวณแปลง",
            "ดูแลบริเวณปลูกให้มีการระบายอากาศ"
        ]
    },

    "Chewing insect damage on banana leaf": {
        "name": "ความเสียหายจากแมลงกัดกินใบ",
        "detail": "พบรอยกัดหรือรูบนใบกล้วย",
        "care": [
            "ตรวจสอบใบและกำจัดแมลงที่พบ",
            "ตัดใบที่เสียหายมากออก",
            "ตรวจสอบใบอื่นอย่างสม่ำเสมอ"
        ]
    },

    "Healthy Banana leaf": {
        "name": "ใบกล้วยสุขภาพดี",
        "detail": "ไม่พบลักษณะของโรคในกลุ่มที่โมเดลจำแนก",
        "care": [
            "ดูแลน้ำและธาตุอาหารให้เหมาะสม",
            "ตรวจสอบใบอย่างสม่ำเสมอ",
            "เฝ้าระวังโรคและแมลง"
        ]
    },

    "Panama Wilt Disease": {
        "name": "โรคตายพราย",
        "detail": "พบลักษณะที่อาจเกี่ยวข้องกับโรคตายพราย",
        "care": [
            "แยกต้นที่สงสัยว่าเป็นโรค",
            "หลีกเลี่ยงการเคลื่อนย้ายดินจากบริเวณที่พบโรค",
            "ปรึกษาผู้เชี่ยวชาญเพื่อยืนยันโรค"
        ]
    }
}


# =========================
# โหลดโมเดล
# =========================

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


# =========================
# เตรียมรูป
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# โหลดโมเดล
# =========================

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


# =========================
# เลือกรูป
# =========================

st.subheader("🔍 ทำนายใบกล้วย")

uploaded_file = st.file_uploader(
    "เลือกรูปใบกล้วย",
    type=["jpg", "jpeg", "png"]
)


# =========================
# ทำนาย
# =========================

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
        "🔮 ทำนาย",
        use_container_width=True
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

        st.divider()

        # =====================
        # ผลลัพธ์
        # =====================

        if result == "Healthy Banana leaf":

            st.success("🌿 ใบกล้วยสุขภาพดี")

        else:

            st.warning(
                "⚠️ พบลักษณะที่อาจเกี่ยวข้องกับโรค"
            )

        st.subheader(
            f"ผลการทำนาย: {info['name']}"
        )

        st.write(
            f"**ประเภท:** {result}"
        )

        st.write(
            f"**ความมั่นใจ:** "
            f"{confidence_percent:.2f}%"
        )

        st.write(
            f"**รายละเอียด:** {info['detail']}"
        )

        # =====================
        # วิธีดูแล
        # =====================

        st.subheader("🩺 แนวทางการดูแล")

        for item in info["care"]:

            st.write(f"• {item}")

        # =====================
        # คะแนนทุกคลาส
        # =====================

        st.subheader("📊 คะแนนแต่ละประเภท")

        for i, class_name in enumerate(
            CLASS_NAMES
        ):

            score = (
                probabilities[0][i].item()
                * 100
            )

            st.write(
                f"{class_name}: {score:.2f}%"
            )

            st.progress(
                min(score / 100, 1.0)
            )
