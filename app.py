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
    layout="centered"
)


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
        "short": (
            "ความเสียหายของใบกล้วยที่เกิดจากหนอนหรือแมลง "
            "ซึ่งอาจทำให้ใบถูกกัดกินหรือม้วนตัว"
        ),
        "cause": (
            "เกิดจากหนอนหรือแมลงที่เข้าทำลายใบกล้วย "
            "โดยตัวหนอนอาจกัดกินเนื้อใบและทำให้ใบเกิดความเสียหาย"
        ),
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
        "short": (
            "โรคที่ทำให้ใบกล้วยเกิดจุดหรือแผลสีเหลือง "
            "และอาจพัฒนาเป็นสีน้ำตาลหรือดำ"
        ),
        "cause": (
            "เกิดจากเชื้อรากลุ่มที่ทำให้เกิดโรค Sigatoka "
            "โดยสภาพอากาศชื้นและใบมีความชื้นเป็นเวลานาน "
            "สามารถเอื้อต่อการเกิดโรค"
        ),
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
        "short": (
            "ลักษณะความเสียหายของใบกล้วยที่เกิดจากแมลงกัดกิน "
            "ทำให้เกิดรูหรือรอยแหว่งบนใบ"
        ),
        "cause": (
            "เกิดจากแมลงที่กัดกินเนื้อเยื่อของใบกล้วย "
            "ทำให้เกิดรู รอยแหว่ง หรือพื้นที่ใบที่ถูกทำลาย"
        ),
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
        "short": (
            "ใบกล้วยที่มีลักษณะโดยรวมปกติ "
            "และไม่พบลักษณะของโรคหรือความเสียหาย "
            "ในกลุ่มที่โมเดลสามารถจำแนกได้"
        ),
        "cause": (
            "ไม่มีสาเหตุของโรคในกลุ่มที่ตรวจพบจากภาพ "
            "แต่สุขภาพของต้นยังขึ้นอยู่กับน้ำ ธาตุอาหาร "
            "สภาพแวดล้อม และการดูแล"
        ),
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
        "short": (
            "โรคที่เกี่ยวข้องกับเชื้อราที่อาศัยอยู่ในดิน "
            "และสามารถส่งผลต่อระบบท่อลำเลียงของต้นกล้วย"
        ),
        "cause": (
            "เกี่ยวข้องกับเชื้อรา Fusarium oxysporum f. sp. cubense "
            "ซึ่งสามารถอาศัยอยู่ในดินและแพร่กระจายผ่านดิน "
            "น้ำ หรือวัสดุที่ปนเปื้อน"
        ),
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
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = mobilenet_v2(weights=None)

    model.classifier[1] = nn.Linear(
        model.last_channel,
        len(CLASS_NAMES)
    )

    checkpoint = torch.load(
        "best_mobilenetv2_banana.pth",
        map_location=device
    )

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
# ฟังก์ชันทำนาย
# =========================================================

def predict_image(image, model, device):

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    index = predicted.item()
    result = CLASS_NAMES[index]
    confidence_percent = confidence.item() * 100

    return result, confidence_percent, probabilities


# =========================================================
# โหลดโมเดล
# =========================================================

try:

    model, device = load_model()

except Exception as e:

    st.error("❌ ไม่สามารถโหลดโมเดลได้")

    st.info(
        "ตรวจสอบว่าไฟล์ best_mobilenetv2_banana.pth "
        "อยู่ใน Repository เดียวกับ app.py"
    )

    st.code(str(e))
    st.stop()


# =========================================================
# หัวเว็บ
# =========================================================

st.title("🍌 Banana Leaf AI")
st.caption(
    "ระบบ AI สำหรับจำแนกโรคและความผิดปกติของใบกล้วย"
)

st.divider()


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

    st.header("🍌 ยินดีต้อนรับสู่ Banana Leaf AI")

    st.write(
        "อัปโหลดภาพใบกล้วยเพื่อให้โมเดล MobileNetV2 "
        "ช่วยจำแนกประเภทของใบกล้วย"
    )

    st.info(
        "💡 แนะนำให้ใช้ภาพที่เห็นใบกล้วยชัดเจน "
        "และมีแสงเพียงพอ"
    )

    uploaded_home = st.file_uploader(
        "📷 เลือกรูปใบกล้วย",
        type=["jpg", "jpeg", "png"],
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

                result, confidence_percent, probabilities = predict_image(
                    home_image,
                    model,
                    device
                )

            info = DISEASE_INFO[result]

            st.divider()

            if result == "Healthy Banana leaf":

                st.success(
                    f"🌿 ผลการทำนาย: {info['name']}"
                )

            else:

                st.warning(
                    f"⚠️ ผลการทำนาย: {info['name']}"
                )

            st.subheader(
                f"{info['icon']} {info['name']}"
            )

            st.write(
                f"**ประเภท:** {result}"
            )

            st.write(
                f"**ความมั่นใจ:** "
                f"{confidence_percent:.2f}%"
            )

            st.subheader("📋 รายละเอียด")
            st.write(info["short"])

            st.subheader("🔬 สาเหตุ")
            st.write(info["cause"])

            st.subheader("👀 ลักษณะอาการ")

            for item in info["symptoms"]:
                st.write(f"• {item}")

            st.subheader("🩺 แนวทางการดูแล")

            for item in info["management"]:
                st.write(f"• {item}")

            st.subheader("🛡️ การป้องกัน")

            for item in info["prevention"]:
                st.write(f"• {item}")

            st.subheader("📊 คะแนนการทำนายแต่ละประเภท")

            for i, class_name in enumerate(CLASS_NAMES):

                score = probabilities[0][i].item() * 100

                st.write(
                    f"**{class_name}** — {score:.2f}%"
                )

                st.progress(
                    min(score / 100, 1.0)
                )

    else:

        st.info(
            "🍃 กรุณาอัปโหลดรูปใบกล้วยเพื่อเริ่มการทำนาย"
        )

    st.divider()

    st.subheader("✨ ฟังก์ชันของระบบ")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🤖 AI วิเคราะห์")
        st.write(
            "ใช้ MobileNetV2 สำหรับจำแนกภาพใบกล้วย"
        )

    with col2:
        st.markdown("### 🔬 วิเคราะห์โรค")
        st.write(
            "แสดงประเภทและระดับความมั่นใจของการทำนาย"
        )

    with col3:
        st.markdown("### 📚 ความรู้")
        st.write(
            "ดูสาเหตุ อาการ การดูแล และการป้องกัน"
        )


# =========================================================
# หน้าทำนาย
# =========================================================

elif page == "🔍 ทำนายโรค":

    st.header("🔍 ทำนายโรคใบกล้วย")

    st.write(
        "อัปโหลดภาพใบกล้วยเพื่อให้ AI วิเคราะห์"
    )

    uploaded_file = st.file_uploader(
        "📷 เลือกรูปใบกล้วย",
        type=["jpg", "jpeg", "png"],
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

                result, confidence_percent, probabilities = predict_image(
                    image,
                    model,
                    device
                )

            info = DISEASE_INFO[result]

            st.divider()

            if result == "Healthy Banana leaf":
                st.success(
                    "🌿 ผลการทำนาย: ใบกล้วยสุขภาพดี"
                )
            else:
                st.warning(
                    "⚠️ พบลักษณะที่โมเดลจำแนกเป็นความผิดปกติ"
                )

            st.subheader(
                f"{info['icon']} {info['name']}"
            )

            st.write(
                f"**ประเภท:** {result}"
            )

            st.write(
                f"**ความมั่นใจ:** "
                f"{confidence_percent:.2f}%"
            )

            st.subheader("📋 รายละเอียด")
            st.write(info["short"])

            st.subheader("🔬 สาเหตุ")
            st.write(info["cause"])

            st.subheader("👀 อาการที่พบ")

            for item in info["symptoms"]:
                st.write(f"• {item}")

            st.subheader("🩺 แนวทางการดูแล")

            for item in info["management"]:
                st.write(f"• {item}")

            st.subheader("🛡️ การป้องกัน")

            for item in info["prevention"]:
                st.write(f"• {item}")

            st.subheader("📊 คะแนนแต่ละประเภท")

            for i, class_name in enumerate(CLASS_NAMES):

                score = probabilities[0][i].item() * 100

                st.write(
                    f"**{class_name}**: {score:.2f}%"
                )

                st.progress(
                    min(score / 100, 1.0)
                )

    else:

        st.info("🍌 กรุณาอัปโหลดภาพใบกล้วย")


# =========================================================
# หน้าความรู้
# =========================================================

elif page == "📚 ความรู้โรคใบกล้วย":

    st.header("📚 ความรู้โรคใบกล้วย")

    st.write(
        "เลือกประเภทเพื่อดูรายละเอียด สาเหตุ อาการ "
        "แนวทางการดูแล และการป้องกัน"
    )

    selected_disease = st.selectbox(
        "🍃 เลือกประเภท",
        CLASS_NAMES
    )

    info = DISEASE_INFO[selected_disease]

    st.divider()

    st.subheader(
        f"{info['icon']} {info['name']}"
    )

    st.write(
        f"**ชื่อประเภท:** {selected_disease}"
    )

    st.subheader("📋 รายละเอียด")
    st.write(info["short"])

    st.subheader("🔬 สาเหตุ")
    st.write(info["cause"])

    st.subheader("👀 อาการที่พบ")

    for item in info["symptoms"]:
        st.write(f"• {item}")

    st.subheader("🩺 แนวทางการดูแล")

    for item in info["management"]:
        st.write(f"• {item}")

    st.subheader("🛡️ การป้องกัน")

    for item in info["prevention"]:
        st.write(f"• {item}")


# =========================================================
# Footer
# =========================================================

st.divider()

st.caption(
    "🍌 Banana Leaf AI | ระบบจำแนกโรคใบกล้วยด้วย MobileNetV2"
)
