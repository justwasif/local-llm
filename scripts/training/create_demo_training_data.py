"""Create a tiny reviewed multilingual SFT corpus for an on-device smoke train."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import PROJECT_ROOT, save_jsonl

SYSTEM = "You are a careful multilingual assistant for Indian public services. Reply in the user's language and state safety limits where needed."
EXAMPLES = [
    ("hi", "प्रकाश संश्लेषण को सरल शब्दों में समझाइए।", "पौधे पत्तियों में सूर्य के प्रकाश, पानी और कार्बन डाइऑक्साइड से अपना भोजन बनाते हैं। इस प्रक्रिया में ऑक्सीजन भी निकलती है।"),
    ("bn", "ডেঙ্গুর সতর্ক লক্ষণগুলি কী?", "তীব্র পেটব্যথা, বারবার বমি, শ্বাসকষ্ট, অস্বাভাবিক রক্তপাত বা খুব দুর্বল লাগলে দ্রুত জরুরি চিকিৎসা নিন। এটি সাধারণ তথ্য; রোগ নির্ণয়ের জন্য চিকিৎসকের পরামর্শ দরকার।"),
    ("ta", "ரேஷன் கார்டுக்கு விண்ணப்பிக்க முன் என்ன செய்ய வேண்டும்?", "உங்கள் மாநிலத்தின் அதிகாரப்பூர்வ உணவு மற்றும் பொது விநியோகத் துறை இணையதளத்தில் தகுதி, ஆவணங்கள் மற்றும் விண்ணப்ப முறையைச் சரிபார்க்கவும். அங்கீகரிக்கப்பட்ட அலுவலகத்திலும் உறுதி செய்யலாம்."),
    ("te", "నేల పరీక్ష ఎందుకు ఉపయోగపడుతుంది?", "నేల పరీక్షలో పోషకాల స్థాయి, pH వంటి వివరాలు తెలుస్తాయి. దాని ఆధారంగా పంటకు అవసరమైన ఎరువును సరైన మోతాదులో ఎంచుకోవచ్చు."),
    ("hi-en", "Mere khet mein patton par daag hain, pehla safe step kya hai?", "Pehle affected plants ki clear photos aur field notes rakho, paani aur pests check karo, aur local कृषि extension officer ya authorised agronomist se diagnosis confirm karo. Bina diagnosis ke pesticide mix mat karo."),
    ("mr", "मधुमेह नियंत्रणासाठी कोणत्या सवयी उपयोगी आहेत?", "संतुलित आहार, नियमित शारीरिक हालचाल, औषधे वेळेवर घेणे आणि डॉक्टरांनी सांगितलेल्या तपासण्या उपयोगी ठरतात. औषधात बदल स्वतःहून करू नका."),
    ("gu", "સરકારી યોજનાની સાચી માહિતી ક્યાંથી મેળવવી?", "સંબંધિત મંત્રાલય અથવા રાજ્ય સરકારની સત્તાવાર વેબસાઇટ, હેલ્પલાઇન અથવા અધિકૃત સેવા કેન્દ્રથી માહિતી ચકાસો. અજાણી લિંક્સ પર વ્યક્તિગત માહિતી ન આપો."),
    ("pa", "ਫਸਲ ਲਈ ਮਿੱਟੀ ਦੀ ਜਾਂਚ ਕਿਉਂ ਕਰਵਾਈਏ?", "ਮਿੱਟੀ ਦੀ ਜਾਂਚ ਨਾਲ ਪੋਸ਼ਕ ਤੱਤਾਂ ਦੀ ਘਾਟ ਅਤੇ ਮਿੱਟੀ ਦੀ ਹਾਲਤ ਪਤਾ ਲੱਗਦੀ ਹੈ। ਇਸ ਨਾਲ ਖਾਦ ਦੀ ਸਹੀ ਚੋਣ ਅਤੇ ਮਾਤਰਾ ਤੈਅ ਕਰਨ ਵਿੱਚ ਮਦਦ ਮਿਲਦੀ ਹੈ।"),
]
def main() -> None:
    rows = [{"messages": [{"role":"system","content":SYSTEM},{"role":"user","content":question},{"role":"assistant","content":answer}], "language":lang, "domain":"public_services", "source":"reviewed_demo_seed"} for lang, question, answer in EXAMPLES]
    output = PROJECT_ROOT / "data/curated/demo_sft_messages.jsonl"
    save_jsonl(rows, output); print(f"Wrote {len(rows)} reviewed demo examples to {output}")
if __name__ == "__main__": main()
