"""Create small, versioned multilingual seed data for synthetic generation."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import PROJECT_ROOT, save_jsonl, setup_logging

LOGGER = setup_logging("seed_data", "data_download.log")

SEEDS = [
    {"text": "प्रकाश संश्लेषण में पौधे सूर्य के प्रकाश से भोजन बनाते हैं।", "language": "hi", "domain": "education", "topic": "biology"},
    {"text": "পিথাগোরাসের উপপাদ্য সমকোণী ত্রিভুজের তিনটি বাহুর সম্পর্ক ব্যাখ্যা করে।", "language": "bn", "domain": "education", "topic": "mathematics"},
    {"text": "டெங்கு காய்ச்சல் ஏடிஸ் கொசு கடிப்பால் ஏற்படலாம்; அதிக காய்ச்சல் இருந்தால் மருத்துவரை அணுகவும்.", "language": "ta", "domain": "healthcare", "topic": "dengue"},
    {"text": "భూసార పరీక్ష ద్వారా నేలలో పోషకాల లోపాన్ని తెలుసుకుని ఎరువులను ఎంచుకోవచ్చు.", "language": "te", "domain": "agriculture", "topic": "soil_testing"},
    {"text": "आधार कार्ड भारत सरकार की 12 अंकों की पहचान संख्या है। केवल आधिकारिक UIDAI चैनलों से बदलाव करें।", "language": "hi", "domain": "public_services", "topic": "aadhaar"},
    {"text": "Bhai, Aadhaar update ke liye official UIDAI website ya authorised centre hi use karo.", "language": "hi-en", "domain": "public_services", "topic": "aadhaar"},
]


def main() -> None:
    seed_dir = PROJECT_ROOT / "data/raw/seeds"
    save_jsonl(SEEDS, seed_dir / "all_seeds.jsonl")
    for domain in sorted({row["domain"] for row in SEEDS}):
        save_jsonl([row for row in SEEDS if row["domain"] == domain], seed_dir / f"seeds_{domain}.jsonl")
    LOGGER.info("Wrote %s seed records to %s", len(SEEDS), seed_dir)


if __name__ == "__main__":
    main()
