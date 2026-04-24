import streamlit as st

from core import (
    AdGenerator,
    DEFAULT_PROMPT_TEMPLATE_TEXT,
    DEFAULT_SYSTEM_MSG,
    image_to_png_bytes,
    parse_ad_sections,
    render_png_ad,
)


st.set_page_config(page_title="Ad Generator", layout="wide")
st.title("Ad Generator")
st.caption("TinyLlama + your validation pipeline + theme-aware visuals")


@st.cache_resource(show_spinner=False)
def get_generator(model_name: str, device: str | None) -> AdGenerator:
    return AdGenerator(model_name=model_name, device=device)


with st.sidebar:
    st.header("Model Settings")
    model_name = st.text_input("Hugging Face model", value="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    device_choice = st.selectbox("Device", options=["auto", "cpu", "cuda"], index=0)
    retries = st.slider("Retries", min_value=1, max_value=8, value=4)
    selected_device = None if device_choice == "auto" else device_choice

    st.divider()
    st.header("Visual Settings")
    size_label = st.selectbox(
        "Ad Size",
        options=["Portrait 1080x1350", "Square 1080x1080", "Story 1080x1920"],
        index=0,
    )
    layout_mode = st.selectbox("Layout", options=["Spacious", "Balanced", "Compact"], index=0)
    font_scale = st.slider("Font Scale", min_value=0.80, max_value=1.20, value=0.96, step=0.02)

    st.divider()
    st.header("Prompt Settings")
    system_msg = st.text_area("System Message", value=DEFAULT_SYSTEM_MSG, height=240)
    prompt_template_text = st.text_area(
        "Prompt Template",
        value=DEFAULT_PROMPT_TEMPLATE_TEXT,
        height=320,
        help="Required placeholders: {user_name}, {product_specs}, {discount}, {theme}, {channel}, {tone}",
    )

size_map = {
    "Square 1080x1080": (1080, 1080),
    "Portrait 1080x1350": (1080, 1350),
    "Story 1080x1920": (1080, 1920),
}

with st.form("ad_form"):
    left, right = st.columns(2)
    with left:
        user_name = st.text_input("User Name", value="Nina")
        product_specs = st.text_area(
            "Product Specs",
            value="Electric city bike; pedal assist; up to 100 km range; integrated LED lights; helmet included",
            height=140,
        )
        discount = st.text_input("Discount / Promo", value="20% off this week for online bookings")
    with right:
        theme = st.text_input("Theme", value="Eco-friendly commuting")
        channel = st.selectbox("Channel", options=["Instagram", "TikTok", "Facebook", "Email", "Web"], index=0)
        tone = st.selectbox("Tone", options=["friendly", "energetic", "premium", "playful", "urgent"], index=0)
    submit = st.form_submit_button("Generate Ad")


if submit:
    try:
        with st.spinner("Loading model and generating ad..."):
            generator = get_generator(model_name, selected_device)
            result = generator.generate_ad(
                product_specs=product_specs,
                discount=discount,
                theme=theme,
                user_name=user_name,
                channel=channel,
                tone=tone,
                retries=retries,
                system_msg=system_msg,
                prompt_template_text=prompt_template_text,
            )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    ad_text = result["text"]
    sections = parse_ad_sections(ad_text)
    image = render_png_ad(
        sections,
        size=size_map[size_label],
        theme=theme,
        tone=tone,
        layout_mode=layout_mode.lower(),
        font_scale=font_scale,
    )
    png_bytes = image_to_png_bytes(image)

    text_col, image_col = st.columns([1.0, 1.05])
    with text_col:
        st.subheader("Generated Ad Copy")
        st.code(ad_text)
        st.subheader("Validation")
        st.json(result["checks"])
        st.download_button(
            label="Download Ad Text",
            data=ad_text,
            file_name="generated_ad.txt",
            mime="text/plain",
        )
    with image_col:
        st.subheader("Generated Visual")
        st.image(image, use_container_width=True)
        st.download_button(
            label="Download PNG",
            data=png_bytes,
            file_name="generated_ad.png",
            mime="image/png",
        )
