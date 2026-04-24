"""
Standalone demo: generate a BikeEase ad using the AdGenerator pipeline.

This script mirrors the notebook exploration and is intended as a
minimal runnable example. The production app lives in app.py.

Usage:
    python notebook_code.py
"""

from core.ad_generator import AdGenerator, parse_ad_sections


def main() -> None:
    print("Loading model (first run will download weights ~2.2 GB)...")
    generator = AdGenerator()

    examples = [
        {
            "product_specs": "Electric city bike; pedal assist; up to 70km range; integrated lights; helmet included",
            "discount": "20% off this week for online bookings",
            "theme": "Eco-friendly commuting",
            "user_name": "Nina",
            "channel": "Instagram",
            "tone": "energetic",
        },
        {
            "product_specs": "Full-suspension mountain bike; front and rear suspension; hydraulic disc brakes",
            "discount": "50% off this week for online bookings",
            "theme": "Enjoying nature",
            "user_name": "Bobby",
            "channel": "Instagram",
            "tone": "energetic",
        },
    ]

    for i, ex in enumerate(examples, start=1):
        print(f"\n{'=' * 60}")
        print(f"Example {i}: {ex['user_name']} — {ex['theme']}")
        print("=" * 60)

        result = generator.generate_ad(**ex)
        if result is None:
            print("Generation failed — no valid ad produced.")
            continue

        print(result["text"])
        print(f"\nValidation: {result['checks']}")

        sections = parse_ad_sections(result["text"])
        print(f"\nParsed sections: {list(sections.keys())}")


if __name__ == "__main__":
    main()
