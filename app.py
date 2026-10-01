import os
import sys
import tempfile

import streamlit as st


# Add src to Python path
sys.path.append(
    os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "src"
    )
)

from caption import generate_caption


# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)


# ==============================
# TITLE
# ==============================

st.title(
    "🖼️ AI Image Caption Generator"
)

st.write(
    "Upload an image and let the AI generate a textual caption."
)


# ==============================
# UPLOAD
# ==============================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ==============================
# PROCESS
# ==============================

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )


    if st.button(
        "✨ Generate Caption"
    ):

        with st.spinner(
            "Generating caption..."
        ):

            suffix = os.path.splitext(
                uploaded_file.name
            )[1]


            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                temp_path = temp_file.name


            caption = generate_caption(
                temp_path
            )


        st.success(
            "Generated Caption:"
        )

        st.markdown(
            f"### 📝 {caption}"
        )


        try:
            os.remove(
                temp_path
            )
        except:
            pass