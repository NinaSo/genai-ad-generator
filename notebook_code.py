# Install packages and import modules

!pip install -q langchain langchain-community



import re

import json

import torch



from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline

from langchain_core.prompts import PromptTemplate

from langchain_community.llms import HuggingFacePipeline
# Try using a GPU if available, else use a CPU

device = "cuda" if torch.cuda.is_available() else "cpu"



MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"



# Load model/tokenizer

tok = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(        # if google-flan-t5 then use AutoModelForSeq2SeqLM

    MODEL_NAME,

    torch_dtype=torch.float16 if device == "cuda" else torch.float32

).to(device).eval()
# Instructions for the model to generate the right ad



SYSTEM_MSG = ("""

You are an advertising specialist for BikeEase, a bike rental company.

Your job is to generate marketing ads that are clear, persuasive, and consistent with BikeEase’s brand.



STRICT RULES:

- Use ONLY the provided bike specs, discount, and theme. Do NOT invent prices, emissions numbers, distances, or extra features.

- Do NOT add sections other than the required ones.

- Output MUST match the format exactly. Use the exact labels.

- Voice: Friendly, upbeat, and benefit-driven (focus on what the customer gets).

- Simple language, short sentences.



OUTPUT FORMAT (exactly this, in this order):



[HEADLINE]

(one line, include the user name exactly)



[SUBHEADLINE]

(one sentence, must mention the discount)



[BODY]

(2–4 short sentences, mention exactly 2 features from specs)



[CTA]

(2–6 words)



[HASHTAGS]

(3–6 hashtags, include #BikeEase)





""".strip())



SYSTEM_MSG = """

You are an advertising specialist for BikeEase (bike rental).

Write ONE short ad.



Hard rules:

- Use ONLY the provided bike specs, discount, and theme.

- Do NOT invent prices, engines, brands, ranges, or extra features.

- Output MUST be ONLY what's inside <AD> ... </AD>.

- Must contain exactly these 5 sections in this order:

  [HEADLINE], [SUBHEADLINE], [BODY], [CTA], [HASHTAGS]

- HEADLINE: one line, MUST include the exact user name.

- SUBHEADLINE: one sentence, MUST mention the discount.

- BODY: 2–4 short sentences, mention EXACTLY 2 features from the specs.

- CTA: 2–6 words.

- HASHTAGS: 3–6 hashtags, MUST include #BikeEase.

No extra commentary.

""".strip()
# Generate prompt template

prompt_tmpl = PromptTemplate(

    input_variables=["bike_specs", "discount", "theme", "channel", "tone", "user_name"],

    template=(

        "Campaign settings:\n"

        "- Channel: {channel}\n"

        "- Tone: {tone}\n\n"

        "Inputs:\n"

        "- User name: {user_name}\n"

        "- Bike specs: {bike_specs}\n"

        "- Discount/Promo: {discount}\n"

        "- Theme: {theme}\n\n"

        "Return ONLY this format between <AD> and </AD>.\n\n"

        "<AD>\n"

        "[HEADLINE]\n"

        "<write one line here>\n\n"

        "[SUBHEADLINE]\n"

        "<write one sentence here>\n\n"

        "[BODY]\n"

        "<write 2-4 short sentences here>\n\n"

        "[CTA]\n"

        "<write 2-6 words here>\n\n"

        "[HASHTAGS]\n"

        "<write 3-6 hashtags here>\n"

        "</AD>\n"

        "[END]"

    )

)


# Helper Functions for ad generation



def _build_chat_prompt(user_content: str) -> str:



    messages = [

        {"role": "system", "content": SYSTEM_MSG},

        {"role": "user", "content": user_content}

    ]

    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)



def _generate_text(prompt: str) -> str:

    # Tokenize prompt (words --> numeric IDs)

    # return_tensors="pt" (PyTorch)/"tf" (TensorFlow)

    inputs = tok(prompt, return_tensors="pt").to(device) # .to(device) = move tensors to GPU if available



    with torch.no_grad():            # No training so no gradient tracking needed

        # Generate new text from the model based on the prompt

        out = model.generate(

            **inputs,                # The tokenized prompt

            max_new_tokens=260,      # Cap on how many new words to generate

            do_sample=True,          # Enable sampling for variety, if False, replace temperature variable with num_beams=3,

            temperature=0.3,         # Randomness: lower = focused, higher = creative

            top_p=0.9,               # Sample from top 90% of probable words

            repetition_penalty=1.15,  # Avoid repeating the same phrase

            no_repeat_ngram_size=4,   # Block repeating any 4-word sequence

            eos_token_id=tok.eos_token_id,

            pad_token_id=tok.pad_token_id,

        )



    # For causal models (TinyLlama), slice off prompt tokens, for T5 remove slicing

    gen_ids = out[0][inputs["input_ids"].shape[1]:]



    # Decode the token IDs back into readable text

    return tok.decode(gen_ids, skip_special_tokens=True).strip() # For T5: return tok.decode(out[0], skip_special_tokens=True).strip()



# Ad checker functions



# Define required labels globally for validation

required_labels = ["[HEADLINE]", "[SUBHEADLINE]", "[BODY]", "[CTA]", "[HASHTAGS]"]



if tok.pad_token_id is None and tok.eos_token_id is not None:

    tok.pad_token = tok.eos_token



# Normalize labels text

def normalize_labels(text: str) -> str:

    t = text

    # Convert bare labels to bracketed labels

    t = re.sub(r"^\s*Subheadline\s*$", "[SUBHEADLINE]", t, flags=re.M|re.I)

    t = re.sub(r"^\s*Headline\s*$", "[HEADLINE]", t, flags=re.M|re.I)

    t = re.sub(r"^\s*Body\s*$", "[BODY]", t, flags=re.M|re.I)

    t = re.sub(r"^\s*CTA\s*$", "[CTA]", t, flags=re.M|re.I)

    t = re.sub(r"^\s*Hashtags\s*$", "[HASHTAGS]", t, flags=re.M|re.I)



    # Also handle bracketed-but-wrong-case

    t = re.sub(r"\[(headline)\]", "[HEADLINE]", t, flags=re.I)

    t = re.sub(r"\[(subheadline)\]", "[SUBHEADLINE]", t, flags=re.I)

    t = re.sub(r"\[(body)\]", "[BODY]", t, flags=re.I)

    t = re.sub(r"\[(cta)\]", "[CTA]", t, flags=re.I)

    t = re.sub(r"\[(hashtags)\]", "[HASHTAGS]", t, flags=re.I)



    return t.strip()



# Check if ad has required labels (defined above)

def _has_required_labels(text: str) -> bool:

    t = _normalize_labels(text)

    return all(lbl in t for lbl in required_labels)



# Check if user name is in the ad (to increase personal tone)

def _name_in_headline(text: str, user_name: str) -> bool:

    t = _normalize_labels(text)

    m = re.search(r"\[HEADLINE\]\s*(.+)", t)

    return bool(m) and (user_name.lower() in m.group(1).lower())



# Combine all checkers into a validation function

def validate_ad(text: str, user_name: str) -> dict:

    t = _normalize_labels(text)

    checks = {

        "has_labels": _has_required_labels(t),

        "name_in_headline": _name_in_headline(t, user_name),

        "has_bikeease_hashtag": re.search(r"#bikeease\b", t, flags=re.I) is not None

    }

    checks["score"] = sum(checks.values())

    return checks





def extract_ad_block(text: str) -> str:

    text = text.strip()

    m = re.search(r"<AD>\s*(.*?)\s*</AD>", text, flags=re.S|re.I)

    if m:

        return m.group(1).strip()



    # fallback: cut from first label

    m2 = re.search(r"(\[HEADLINE\].*)", text, flags=re.S)

    return m2.group(1).strip() if m2 else text

def generate_ad(bike_specs, discount, theme, user_name, channel="Instagram", tone="friendly", retries=4):



    '''The function builds the text prompt that will be fed to the model using the system message (brand rules) and user input:'''



    # Build the user content

    user_content = prompt_tmpl.format(

        user_name=user_name,

        bike_specs=bike_specs,

        discount=discount,

        theme=theme,

        channel=channel,

        tone=tone

    )

    prompt = _build_chat_prompt(user_content)



    best = None

    for _ in range(retries):

        raw = _generate_text(prompt)

        # HARD STOP CUT

        if "[END]" in raw:

            raw = raw.split("[END]")[0] + "[END]"



        raw = extract_ad_block(raw)

        text = normalize_labels(raw)

        checks = validate_ad(text, user_name)



        cand = {"text": text, "checks": checks}

        if best is None or cand["checks"]["score"] > best["checks"]["score"]:

            best = cand

        if checks["score"] == 3:  # perfect score if all rules are verified

            break



    return best

# Define some user inputs here. You should modify these...

bike_specs = "Electric city bike; pedal assist; up to 70km range; integrated lights; helmet included"

discount = "20% off this week for online bookings"

theme = "Eco-friendly commuting"

user = "Nina"



ad = generate_ad(bike_specs, discount, theme, user, "Instagram", "energetic")

print(ad["text"])

bike_specs = "Full-suspension mountain bikes"

discount = "50% off this week for online bookings"

theme = "Enjoying-nature"

user = "Bobby"



ad2 = generate_ad(bike_specs, discount, theme, user, "Instagram", "energetic")

print(ad2["text"])
!pip install pillow

from PIL import Image, ImageDraw, ImageFont

import textwrap

import os



def render_png_ad(sections, filename="bikeease_ad.png"):



    # Canvas

    W, H = 1080, 1080

    img = Image.new("RGB", (W, H), "#111827")

    draw = ImageDraw.Draw(img)



    # Gradient background

    for y in range(H):

        r = int(15 + (y / H) * 20)

        g = int(23 + (y / H) * 30)

        b = int(42 + (y / H) * 40)

        draw.line([(0, y), (W, y)], fill=(r, g, b))



    # Try to load system fonts (fallback to default)

    try:

        headline_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 90)

        sub_font = ImageFont.truetype("DejaVuSans.ttf", 48)

        body_font = ImageFont.truetype("DejaVuSans.ttf", 42)

        cta_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 52)

        tag_font = ImageFont.truetype("DejaVuSans.ttf", 36)

    except:

        headline_font = ImageFont.load_default()

        sub_font = ImageFont.load_default()

        body_font = ImageFont.load_default()

        cta_font = ImageFont.load_default()

        tag_font = ImageFont.load_default()



    # Extract sections

    headline = sections["headline"]

    sub = sections["subheadline"]

    body = sections["body"]

    cta = sections["cta"]

    hashtags = sections["hashtags"]



    # Helper for wrapped text

    def draw_multiline(text, font, x, y, max_width, fill):

        lines = []

        words = text.split()

        current = ""

        for word in words:

            test = current + " " + word if current else word

            w, h = draw.textbbox((0,0), test, font=font)[2:]

            if w <= max_width:

                current = test

            else:

                lines.append(current)

                current = word

        if current:

            lines.append(current)



        for line in lines:

            draw.text((x, y), line, font=font, fill=fill)

            y += draw.textbbox((0,0), line, font=font)[3] + 10

        return y



    # Draw content

    margin = 120

    y = 180



    # Headline

    y = draw_multiline(headline, headline_font, margin, y, W - margin*2, "white")

    y += 30



    # Subheadline

    y = draw_multiline(sub, sub_font, margin, y, W - margin*2, "#cbd5e1")

    y += 60



    # Body

    y = draw_multiline(body, body_font, margin, y, W - margin*2, "#e5e7eb")

    y += 80



    # CTA Button

    btn_w = 520

    btn_h = 110

    btn_x = (W - btn_w) // 2

    btn_y = y



    draw.rounded_rectangle(

        [btn_x, btn_y, btn_x+btn_w, btn_y+btn_h],

        radius=30,

        fill="#f59e0b"

    )



    text_w, text_h = draw.textbbox((0,0), cta, font=cta_font)[2:]

    draw.text(

        (btn_x + (btn_w-text_w)//2, btn_y + (btn_h-text_h)//2),

        cta,

        font=cta_font,

        fill="black"

    )



    # Hashtags footer

    draw.text(

        (margin, H - 140),

        hashtags,

        font=tag_font,

        fill="#9ca3af"

    )



    img.save(filename)

    return img

sections = parse_ad(res["text"])



img = render_png_ad(sections)

img
