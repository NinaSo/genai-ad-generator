import re
from typing import Dict, Optional

import torch
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer

REQUIRED_LABELS = ["[HEADLINE]", "[SUBHEADLINE]", "[BODY]", "[CTA]", "[HASHTAGS]"]
PROMPT_INPUT_VARIABLES = ["product_specs", "discount", "theme", "channel", "tone", "user_name"]

DEFAULT_SYSTEM_MSG = """
You are an advertising specialist for BikeEase (bike rental).
Write ONE short ad.

Hard rules:
- Use ONLY the provided product specs, discount, and theme.
- Do NOT invent prices, engines, brands, ranges, or extra features.
- Output MUST be ONLY what's inside <AD> ... </AD>.
- Must contain exactly these 5 sections in this order:
  [HEADLINE], [SUBHEADLINE], [BODY], [CTA], [HASHTAGS]
- HEADLINE: one line, MUST include the exact user name.
- SUBHEADLINE: one sentence, MUST mention the discount.
- BODY: 2-4 short sentences, mention EXACTLY 2 features from the specs.
- CTA: 2-6 words.
- HASHTAGS: 3-6 hashtags, MUST include #BikeEase.
No extra commentary.
""".strip()

DEFAULT_PROMPT_TEMPLATE_TEXT = (
    "Campaign settings:\n"
    "- Channel: {channel}\n"
    "- Tone: {tone}\n\n"
    "Inputs:\n"
    "- User name: {user_name}\n"
    "- Product specs: {product_specs}\n"
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


class AdGenerator:
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: Optional[str] = None,
        system_msg: Optional[str] = None,
        prompt_template_text: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.system_msg = system_msg or DEFAULT_SYSTEM_MSG
        self.prompt_template_text = prompt_template_text or DEFAULT_PROMPT_TEMPLATE_TEXT
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()

        if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
            self.tok.pad_token = self.tok.eos_token

    def _build_chat_prompt(self, user_content: str, system_msg: str) -> str:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]
        if hasattr(self.tok, "apply_chat_template") and self.tok.chat_template:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"{system_msg}\n\n{user_content}"

    def _generate_text(self, prompt: str) -> str:
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=260,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                eos_token_id=self.tok.eos_token_id,
                pad_token_id=self.tok.pad_token_id,
            )

        gen_ids = out[0][inputs["input_ids"].shape[1] :]
        return self.tok.decode(gen_ids, skip_special_tokens=True).strip()

    def generate_ad(
        self,
        product_specs: str,
        discount: str,
        theme: str,
        user_name: str,
        channel: str = "Instagram",
        tone: str = "friendly",
        retries: int = 4,
        system_msg: Optional[str] = None,
        prompt_template_text: Optional[str] = None,
    ) -> Dict:
        active_system_msg = system_msg or self.system_msg
        active_prompt_template_text = prompt_template_text or self.prompt_template_text

        prompt_template = PromptTemplate(
            input_variables=PROMPT_INPUT_VARIABLES,
            template=active_prompt_template_text,
        )
        try:
            user_content = prompt_template.format(
                user_name=user_name,
                product_specs=product_specs,
                discount=discount,
                theme=theme,
                channel=channel,
                tone=tone,
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "Prompt template is invalid or missing required placeholders. "
                f"Use placeholders: {', '.join('{' + v + '}' for v in PROMPT_INPUT_VARIABLES)}."
            ) from exc

        prompt = self._build_chat_prompt(user_content, active_system_msg)

        best = None
        for _ in range(max(1, retries)):
            raw = self._generate_text(prompt)
            if "[END]" in raw:
                raw = raw.split("[END]")[0] + "[END]"

            extracted = extract_ad_block(raw)
            text = normalize_labels(extracted)
            checks = validate_ad(text, user_name)
            candidate = {"text": text, "checks": checks, "raw": raw}

            if best is None or candidate["checks"]["score"] > best["checks"]["score"]:
                best = candidate
            if checks["score"] == 3:
                break

        return best


def normalize_labels(text: str) -> str:
    normalized = text
    normalized = re.sub(r"^\s*Subheadline\s*$", "[SUBHEADLINE]", normalized, flags=re.M | re.I)
    normalized = re.sub(r"^\s*Headline\s*$", "[HEADLINE]", normalized, flags=re.M | re.I)
    normalized = re.sub(r"^\s*Body\s*$", "[BODY]", normalized, flags=re.M | re.I)
    normalized = re.sub(r"^\s*CTA\s*$", "[CTA]", normalized, flags=re.M | re.I)
    normalized = re.sub(r"^\s*Hashtags\s*$", "[HASHTAGS]", normalized, flags=re.M | re.I)
    normalized = re.sub(r"\[(headline)\]", "[HEADLINE]", normalized, flags=re.I)
    normalized = re.sub(r"\[(subheadline)\]", "[SUBHEADLINE]", normalized, flags=re.I)
    normalized = re.sub(r"\[(body)\]", "[BODY]", normalized, flags=re.I)
    normalized = re.sub(r"\[(cta)\]", "[CTA]", normalized, flags=re.I)
    normalized = re.sub(r"\[(hashtags)\]", "[HASHTAGS]", normalized, flags=re.I)
    return normalized.strip()


def _has_required_labels(text: str) -> bool:
    normalized = normalize_labels(text)
    return all(label in normalized for label in REQUIRED_LABELS)


def _name_in_headline(text: str, user_name: str) -> bool:
    normalized = normalize_labels(text)
    match = re.search(r"\[HEADLINE\]\s*(.+)", normalized)
    return bool(match) and user_name.lower() in match.group(1).lower()


def validate_ad(text: str, user_name: str) -> Dict[str, int]:
    normalized = normalize_labels(text)
    checks = {
        "has_labels": _has_required_labels(normalized),
        "name_in_headline": _name_in_headline(normalized, user_name),
        "has_hashtag": re.search(r"#\w+", normalized) is not None,
    }
    checks["score"] = int(sum(checks.values()))
    return checks


def extract_ad_block(text: str) -> str:
    stripped = text.strip()
    match = re.search(r"<AD>\s*(.*?)\s*</AD>", stripped, flags=re.S | re.I)
    if match:
        return match.group(1).strip()

    fallback = re.search(r"(\[HEADLINE\].*)", stripped, flags=re.S)
    return fallback.group(1).strip() if fallback else stripped


def parse_ad_sections(text: str) -> Dict[str, str]:
    normalized = normalize_labels(extract_ad_block(text))
    keys = {
        "headline": "HEADLINE",
        "subheadline": "SUBHEADLINE",
        "body": "BODY",
        "cta": "CTA",
        "hashtags": "HASHTAGS",
    }

    sections: Dict[str, str] = {}
    for out_key, in_key in keys.items():
        pattern = rf"\[{in_key}\]\s*(.*?)(?=\n\[[A-Z]+\]|\Z)"
        match = re.search(pattern, normalized, flags=re.S)
        sections[out_key] = match.group(1).strip() if match else ""

    return sections
