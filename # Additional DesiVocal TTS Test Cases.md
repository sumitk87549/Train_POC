# Additional DesiVocal TTS Test Cases
## Based on Successful Patterns from Your Research

---

## Category 1: Number Formatting Edge Cases

### Test 1A: Large Numbers Without Commas
**Pattern Found**: Numbers should be written without commas; 50,000 sounds like "50"
```
WRONG: ऑर्डर में 1,50,000 इकाइयां हैं।
CORRECT: ऑर्डर में 150000 इकाइयां हैं।
BETTER: ऑर्डर में एक लाख पचास हजार इकाइयां हैं।
```

### Test 1B: Zero Handling
**Pattern Found**: 00 or 000 is ignored; 07, 09 sound wrong
```
WRONG: आज 07/09/2024 को बैठक है।
CORRECT: आज 7 सितंबर 2024 को बैठक है।
BETTER: आज सात सितंबर दो हजार चौबीस को बैठक है।
```

### Test 1C: Number 10 Special Case
**Pattern Found**: "10" is not read at all, even "दस" doesn't work
```
WRONG: कक्षा में 10 छात्र हैं।
CORRECT: कक्षा में ten छात्र हैं।
BETTER: कक्षा में दस छात्र हैं। (if not working, use "ten")
```

### Test 1D: Range Formatting
**Pattern Found**: "1से5" sounds better than "1-5"
```
WRONG: बैठक 3-5 घंटे चलेगी।
CORRECT: बैठक 3से5 घंटे चलेगी।
ALSO GOOD: बैठक तीन से पांच घंटे चलेगी।
```

### Test 1E: Percentage Without Comma
**Pattern Found**: % sometimes ignored; use "percent" or "प्रतिशत"
```
WRONG: 50% छूट उपलब्ध है।
CORRECT: 50 percent छूट उपलब्ध है।
BETTER: 50 प्रतिशत छूट उपलब्ध है।
BEST: पचास प्रतिशत छूट उपलब्ध है।
```

---

## Category 2: Special Characters and Symbols

### Test 2A: Separator Before Alphanumerics
**Pattern Found**: 1A sounds like "eka"; need separator like . , / : or space
```
WRONG: प्रश्न 1A का उत्तर दें।
CORRECT: प्रश्न 1.A का उत्तर दें।
ALSO GOOD: प्रश्न 1 A का उत्तर दें।
BETTER: प्रश्न एक A का उत्तर दें।
```

### Test 2B: Slash Handling
**Pattern Found**: "/" sounds like "slash" in some voices; is small pause in others
```
WRONG: 15/03/2024 को बैठक है।
WRONG ALSO: क्रॉस-चेक करें।
CORRECT: 15 मार्च 2024 को बैठक है।
CORRECT: क्रॉस चेक करें। (space instead of hyphen)
```

### Test 2C: At Symbol (@)
**Pattern Found**: @ sounds like "har" not "at"; hr@company = "har company"
```
WRONG: ईमेल hr@company.com पर भेजें।
CORRECT: ईमेल hr at company dot com पर भेजें।
BETTER: ईमेल एच आर at the rate company dot com पर भेजें।
```

### Test 2D: Ampersand (&)
**Pattern Found**: R&D = "RD", Q&A needs expansion
```
WRONG: R&D विभाग में काम करता हूं।
CORRECT: R and D विभाग में काम करता हूं।
BETTER: आर एंड डी विभाग में काम करता हूं।
```

### Test 2E: Special Characters in Brackets
**Pattern Found**: (...) adds short pause; <> is not read at all
```
GOOD: उत्पाद की कीमत... हम बाद में बताएंगे।
WRONG: फोन का आकार <6.5 इंच> है।
CORRECT: फोन का आकार 6.5 इंच है।
```

---

## Category 3: Abbreviations and Acronyms

### Test 3A: Common Abbreviations - Write Full Words
**Pattern Found**: डॉ. = "daawww"; रु. = "ruu"; Conf. = "conf"
```
WRONG: डॉ. शर्मा और श्री पटेल उपस्थित होंगे।
CORRECT: डाक्टर शर्मा और श्रीमान पटेल उपस्थित होंगे।
```

### Test 3B: Acronyms Without Periods
**Pattern Found**: NASA, AI, ML work; N.A.S.A., A.I. don't work
```
WRONG: N.A.S.A. के C.E.O. ने I.S.R.O. को बधाई दी।
CORRECT: NASA के CEO ने ISRO को बधाई दी।
BETTER: नासा के सी ई ओ ने इसरो को बधाई दी।
```

### Test 3C: IT Abbreviations
**Pattern Found**: IT = "aaetea"; 2FA = "two faa"; VPNs = "v-p-n-s"
```
WRONG: IT विभाग ने 2FA और VPNs लागू किए।
CORRECT: आई टी विभाग ने टू एफ ए और वी पी एन लागू किए।
BETTER: सूचना प्रौद्योगिकी विभाग ने दो कारक प्रमाणीकरण और वर्चुअल प्राइवेट नेटवर्क लागू किए।
```

### Test 3D: Plural Acronyms
**Pattern Found**: Adding 's' or 'z' causes wrong pronunciation: PCs = "PCS", SOPs = "sops"
```
WRONG: सभी PCs में अपडेट करें।
CORRECT: सभी PC में अपडेट करें।
BETTER: सभी पर्सनल कंप्यूटर में अपडेट करें।
```

### Test 3E: Mixed Terms
**Pattern Found**: T&C = "tc"; P&L = "PL"
```
WRONG: T&C और P&L रिपोर्ट तैयार करें।
CORRECT: T and C और P and L रिपोर्ट तैयार करें।
BETTER: नियम और शर्तें और लाभ हानि रिपोर्ट तैयार करें।
```

---

## Category 4: Punctuation and Pause Control

### Test 4A: Hindi Full Stop (|)
**Pattern Found**: | (danda) is bigger pause than comma; lists should end with |
```
GOOD: आइटम एक, आइटम दो, आइटम तीन।
BETTER: पहला आइटम, दूसरा आइटम, तीसरा आइटम।
```

### Test 4B: Colon and Newline Combination
**Pattern Found**: ":" + \n sounds better than ":" alone; ":" and "-" are small inline pauses
```
WRONG: आवश्यकताएं: लैपटॉप सॉफ्टवेयर प्रशिक्षण
CORRECT: आवश्यकताएं:
लैपटॉप, सॉफ्टवेयर, और प्रशिक्षण।
```

### Test 4C: Dash vs Colon for Lists
**Pattern Found**: "-" is more polite list explanation than ":"
```
FORMAL: यहां विषय हैं: प्रथम, द्वितीय, तृतीय।
POLITE: यहां विषय हैं — प्रथम, द्वितीय, तृतीय।
```

### Test 4D: Exclamation Mark
**Pattern Found**: ! is intermediate pause (not short, not long)
```
GOOD: बधाई हो! आपने सफलता प्राप्त की!
BETTER: बधाई हो। आपने सफलता प्राप्त की।
```

### Test 4E: Comma in Context
**Pattern Found**: "," is short line break, should NOT be between digits
```
WRONG: आज 15,000 रुपये जमा किए।
CORRECT: आज 15000 रुपये जमा किए।
```

---

## Category 5: Time and Measurements

### Test 5A: Time Format
**Pattern Found**: "3:30" = "teen tees"; need written form
```
WRONG: बैठक दोपहर 3:30 बजे है।
CORRECT: बैठक दोपहर साढ़ेतीन बजे है।
ALSO GOOD: बैठक दोपहर तीन बजकर तीस मिनट पर है।
```

### Test 5B: Temperature
**Pattern Found**: 102°F = "102f"
```
WRONG: शरीर का तापमान 102°F है।
CORRECT: शरीर का तापमान 102 डिग्री फ़ारेनहाइट है।
BETTER: शरीर का तापमान एक सौ दो डिग्री फ़ारेनहाइट है।
```

### Test 5C: Measurements
**Pattern Found**: 6.5" = "cheeh pannch"
```
WRONG: फोन का स्क्रीन 6.5" है।
CORRECT: फोन का स्क्रीन 6.5 इंच है।
BETTER: फोन का स्क्रीन साढ़े छह इंच है।
```

### Test 5D: Multiplication
**Pattern Found**: 3x = "teen exx"
```
WRONG: गति 3x तेज है।
CORRECT: गति 3 गुना तेज है।
BETTER: गति तीन गुना तेज है।
```

---

## Category 6: Mixed Hindi-English Content

### Test 6A: English Words in Hindi Context
**Pattern Found**: Hindi-English mix sounds good in some voices with English in en-IN tone
```
GOOD: Namaste, aap kaise hain? Main theek hoon.
GOOD: मैं software engineer हूं।
GOOD: यह project deadline tight है।
```

### Test 6B: Code-Switching
```
GOOD: Office में meeting है। Budget discuss karenge।
GOOD: कंपनी का performance अच्छा है। Growth rate 25 percent है।
```

### Test 6C: Technical Terms
**Pattern Found**: Documentary reads "IT" correctly
```
GOOD: यह documentary IT industry पर आधारित है।
GOOD: हम cloud computing और artificial intelligence पर काम कर रहे हैं।
```

---

## Category 7: Place Names and Proper Nouns

### Test 7A: Indian Place Names
**Pattern Found**: English pronunciation not good; better in Hindi script
```
WRONG: Qutub Minar, Humayun's Tomb, Connaught Place
BETTER: कुतुब मीनार, हुमायूं का मकबरा, कनॉट प्लेस
```

### Test 7B: Word Like "पेन"
**Pattern Found**: "पेन" sounds like "pane"
```
IF ISSUE: एक कलम और नोटबुक लाएं। (use "कलम" instead of "पेन")
OR: एक pen और notebook लाएं। (keep in English)
```

---

## Category 8: Lists and Enumerations

### Test 8A: Best Listing Format
**Pattern Found**: "1. 2. 3. 4." is the best way
```
BEST: चार आइटम हैं। 1. पेन। 2. पेपर। 3. नोटबुक। 4. लैपटॉप।
GOOD: पहला पेन। दूसरा पेपर। तीसरा नोटबुक। चौथा लैपटॉप।
```

### Test 8B: Water Intake Example
**Pattern Found**: "8-10 गिलास" should be "8से10 गिलास"
```
WRONG: दिन में 8-10 गिलास पानी पिएं।
CORRECT: दिन में 8से10 गिलास पानी पिएं।
BETTER: दिन में आठ से दस गिलास पानी पिएं।
```

---

## Category 9: Question Types

### Test 9A: Question Format
**Pattern Found**: Q2 = "kyu doo"; Q4 = "Q chaar"
```
WRONG: Q2 और Q3 के उत्तर दें।
CORRECT: Q दो और Q तीन के उत्तर दें।
BETTER: Q-दो और Q-तीन के उत्तर दें।
ALSO GOOD: प्रश्न दो और प्रश्न तीन के उत्तर दें।
```

### Test 9B: Question Mark Usage
**Pattern Found**: ? changes tone according to question/doubt
```
GOOD: आप कैसे हैं?
GOOD: क्या आप कल आ सकते हैं?
GOOD: यह सच है?
```

---

## Category 10: Currency and Financial

### Test 10A: Rupee Symbol
**Pattern Found**: रु. = "ruu"; avoid abbreviations
```
WRONG: कुल राशि रु. 5000 है।
CORRECT: कुल राशि 5000 रुपये है।
BETTER: कुल राशि पांच हजार रुपये है।
```

### Test 10B: Mixed Amounts
**Pattern Found**: 5,000 रुपये = 5 रुपये (comma causes issue)
```
WRONG: कीमत 5,000 रुपये है।
CORRECT: कीमत 5000 रुपये है।
BETTER: कीमत पांच हजार रुपये है।
```

---

## Category 11: Sentence Breaking and Flow

### Test 11A: Long Continuation
**Pattern Found**: Long sentences should continue; break only when pause is needed
```
WRONG: कंपनी, बढ़ रही है, और, हम, खुश हैं।
CORRECT: कंपनी बढ़ रही है और हम खुश हैं।
```

### Test 11B: Natural Breaking Points
```
GOOD: परियोजना सफल रही। टीम ने कड़ी मेहनत की। ग्राहक संतुष्ट है।
```

---

## Complete Test Sentences (Ready to Use)

### Sentence 1: Numbers and Currency
```
OPTIMIZED: आज कंपनी की बिक्री 250000 रुपये तक पहुंची। यह पिछले महीने से 25 प्रतिशत अधिक है।
```

### Sentence 2: Dates and Times
```
OPTIMIZED: बैठक 15 मार्च 2024 को दोपहर साढ़ेतीन बजे है। कृपया समय पर पहुंचें।
```

### Sentence 3: Technical Terms
```
OPTIMIZED: आई टी विभाग ने artificial intelligence और machine learning का उपयोग करके नई परियोजना शुरू की।
```

### Sentence 4: Mixed Content
```
OPTIMIZED: कंपनी का performance बहुत अच्छा है। हमारी growth rate 30 percent है। Next quarter में और improvement होगा।
```

### Sentence 5: Lists
```
OPTIMIZED: बैठक में तीन विषय हैं। 1. बजट योजना। 2. टीम विस्तार। 3. उत्पाद लॉन्च।
```

### Sentence 6: Abbreviations Expanded
```
OPTIMIZED: नासा के सी ई ओ ने घोषणा की कि इसरो के साथ मिलकर नई परियोजना शुरू करेंगे।
```

### Sentence 7: Contact Information
```
OPTIMIZED: कृपया अपना ईमेल hr at company dot com पर भेजें। या फोन करें नौ नौ नौ नौ आठ सात छह पांच चार तीन पर।
```

### Sentence 8: Ranges and Measurements
```
OPTIMIZED: प्रशिक्षण तीन से पांच घंटे का होगा। कमरे का तापमान 22 से 25 डिग्री सेल्सियस रखें।
```

---

## Quick Reference Card

| Element | Wrong | Correct | Better |
|---------|-------|---------|--------|
| Large Numbers | 50,000 | 50000 | पचास हजार |
| Dates | 15/03/2024 | 15 मार्च 2024 | पंद्रह मार्च 2024 |
| Time | 3:30 | साढ़ेतीन बजे | तीन बजकर तीस मिनट |
| Percentage | 50% | 50 percent | पचास प्रतिशत |
| @ symbol | hr@company | hr at company | hr at the rate company |
| Ranges | 8-10 | 8से10 | आठ से दस |
| Dr. | डॉ. शर्मा | डाक्टर शर्मा | डॉक्टर शर्मा |
| Currency | रु. 5000 | 5000 रुपये | पांच हजार रुपये |
| Acronyms | N.A.S.A. | NASA | नासा |
| Question | Q2 | Q दो | प्रश्न दो |

---

## Testing Priority Order

1. **Critical**: Numbers without commas, dates in words, time in words
2. **High**: Abbreviations expanded, @ and & written out, ranges with "से"
3. **Medium**: Acronyms without periods, proper nouns in Hindi
4. **Low**: Fine-tuning punctuation for natural pauses

Remember: When in doubt, **write it out in full words**!