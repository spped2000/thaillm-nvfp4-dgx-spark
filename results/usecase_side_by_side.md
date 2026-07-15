# Use-case side-by-side: BF16 vs NVFP4 (greedy, seed 0, 200 tokens)

## th_factual_qa
*diverges at token 76*

**Prompt:** `คำถาม: ประเทศไทยมีทั้งหมดกี่จังหวัด
คำตอบ:`

**BF16:**
```
ประเทศไทยมีทั้งหมด 77 จังหวัด
คำถาม: จังหวัดที่มีพื้นที่มากที่สุดในประเทศไทยคือจังหวัดอะไร
คำตอบ: จังหวัดที่มีพื้นที่มากที่สุดในประเทศไทยคือจังหวัดนครราชสีมา
คำถาม: จังหวัดที่มีประชากรมากที่สุดในประเทศไทยคือจังหวัดอะไร
คำตอบ: จังหวัดที่มีประชากรมากที่สุดในประเทศไทยคือกรุงเทพมหานคร
คำถาม: จังหวัดที่มีพื้นที่น้อยที่สุดในประเทศไทยคือจังหวัดอะไร
คำตอบ: จังหวัดที่มีพื้นที่น้อยที่สุด
```

**NVFP4:**
```
ประเทศไทยมีทั้งหมด 77 จังหวัด
คำถาม: จังหวัดที่มีพื้นที่มากที่สุดในประเทศไทยคือจังหวัดอะไร
คำตอบ: จังหวัดที่มีพื้นที่มากที่สุดในประเทศไทยคือ จังหวัดนครราชสีมา
คำถาม: จังหวัดที่มีประชากรมากที่สุดในประเทศไทยคือจังหวัดอะไร
คำตอบ: จังหวัดที่มีประชากรมากที่สุดในประเทศไทยคือ กรุงเทพมหานคร
คำถาม: จังหวัดที่มีประชากรน้อยที่สุดในประเทศไทยคือจังหวัดอะไร
คำตอบ: จังหวัดที่มีประชากรน้อยที่สุดในประเทศไทยคือ จัง
```

## th_news
*diverges at token 3*

**Prompt:** `กรุงเทพฯ – ธนาคารแห่งประเทศไทยแถลงวันนี้ว่า เศรษฐกิจไทยในไตรมาสที่ผ่านมา`

**BF16:**
```
ขยายตัวร้อยละ 3.3 ชะลอลงจากไตรมาสก่อนหน้าที่ขยายตัวร้อยละ 3.5 ตามการชะลอตัวของอุปสงค์ในประเทศ ขณะที่การส่งออกสินค้าและบริการขยายตัวต่อเนื่อง ด้านเสถียรภาพเศรษฐกิจ อัตราเงินเฟ้อทั่วไปลดลงตามราคาน้ำมันขายปลีกในประเทศที่ลดลง ขณะที่อัตราการว่างงานทรงตัวในระดับต่ำ
เศรษฐกิจไทยในไตรมาสที่สามของปี 2555 ขยายตัวร้อยละ 3.3 ชะลอลงจากไตรมาสก่อนหน้าที่ขยายตัวร้อยละ 3.5 ตามการ
```

**NVFP4:**
```
ขยายตัวต่อเนื่อง แต่ยังมีความเสี่ยงจากปัจจัยภายนอก จึงต้องติดตามสถานการณ์อย่างใกล้ชิด พร้อมกับปรับลดประมาณการเศรษฐกิจปีนี้ลงเหลือ 3.3% จากเดิม 3.5% ขณะที่คาดว่าอัตราเงินเฟ้อทั่วไปจะอยู่ที่ 1.2% ลดลงจากเดิมที่คาดไว้ 1.5% ด้านการส่งออกคาดว่าจะขยายตัว 4.5% ลดลงจากเดิมที่คาดไว้ 5.5% ขณะที่การลงทุนภาคเอกชนคาดว่าจะขยายตัว 4.5% ลดลงจากเดิมที่คาดไว้
```

## th_legal
*diverges at token 0*

**Prompt:** `ตามประมวลกฎหมายแพ่งและพาณิชย์ มาตรา 420 ผู้ใดจงใจหรือประมาทเลินเล่อ`

**BF16:**
```
ทำต่อบุคคลอื่นโดยผิดกฎหมายให้เขาเสียหายถึงแก่ชีวิตก็ดี แก่ร่างกายก็ดี อนามัยก็ดี เสรีภาพก็ดี ทรัพย์สินหรือสิทธิอย่างหนึ่งอย่างใดก็ดี ท่านว่าผู้นั้นทำละเมิดจำต้องใช้ค่าสินไหมทดแทนเพื่อการนั้น ดังนั้น ผู้ที่จะต้องรับผิดในความเสียหายที่เกิดขึ้นต้องเป็นผู้ที่ทำละเมิด ซึ่งการกระทำที่จะเป็นละเมิดนั้นต้องเป็นการกระทำโดยจงใจหรือประมาทเลินเล่อ
```

**NVFP4:**
```
ทำต่อบุคคลภายนอกโดยผิดกฎหมายให้เขาเสียหายถึงแก่ชีวิตก็ดี แก่ร่างกายก็ดี อนามัยก็ดี เสรีภาพก็ดี ทรัพย์สินหรือสิทธิอย่างหนึ่งอย่างใดก็ดี ท่านว่าผู้นั้นทำละเมิดจำเลยที่ 1 ขับรถยนต์โดยประมาทชนรถยนต์ของผู้เสียหายที่ 1 ที่ 2 และที่ 3 ซึ่งเป็นรถยนต์ที่ผู้เสียหายทั้งสามใช้ในการประกอบธุรกิจของผู้เสียหายทั้งสาม ทำให้รถยนต์ของผู้เสียหายทั้งสามเส
```

## th_medical
*diverges at token 7*

**Prompt:** `โรคเบาหวานชนิดที่ 2 มีสาเหตุหลักมาจาก`

**BF16:**
```
ความผิดปกติของระบบเผาผลาญอาหารในร่างกาย ทำให้เกิดภาวะดื้อต่ออินซูลิน ซึ่งเป็นฮอร์โมนที่ช่วยควบคุมระดับน้ำตาลในเลือด ทำให้ระดับน้ำตาลในเลือดสูงขึ้น ซึ่งหากไม่ได้รับการรักษาอย่างเหมาะสม อาจนำไปสู่ภาวะแทรกซ้อนต่างๆ เช่น โรคหัวใจ โรคไต โรคหลอดเลือดสมอง เป็นต้น
การรักษาโรคเบาหวานชนิดที่ 2 ประกอบด้วยการรักษาด้วยยาและการรักษาด้วยวิธีอื่นๆ ดังนี้
การรักษาด้วยยา
```

**NVFP4:**
```
ความผิดปกติของฮอร์โมนอินซูลินในร่างกาย ซึ่งเป็นฮอร์โมนที่ช่วยให้ร่างกายนำน้ำตาลในเลือดไปใช้เป็นพลังงาน ผู้ป่วยโรคเบาหวานชนิดที่ 2 จะมีภาวะดื้อต่ออินซูลิน ทำให้ร่างกายไม่สามารถนำอินซูลินไปใช้ได้อย่างมีประสิทธิภาพ จึงทำให้ระดับน้ำตาลในเลือดสูงขึ้น ซึ่งหากปล่อยไว้โดยไม่ได้รับการรักษา อาจนำไปสู่ภาวะแทรกซ้อนร้ายแรงได้
อาการของโรคเบาหวานชนิดที่ 2
อาการของโรคเบาหวานชนิดที่
```

## th_education
*diverges at token 17*

**Prompt:** `การสังเคราะห์ด้วยแสง (photosynthesis) คือกระบวนการที่`

**BF16:**
```
พืช สาหร่าย และแบคทีเรียบางชนิด ใช้พลังงานจากแสงอาทิตย์ในการสังเคราะห์สารอาหารจากคาร์บอนไดออกไซด์และน้ำ กระบวนการนี้เกิดขึ้นในคลอโรพลาสต์ของเซลล์พืช ซึ่งเป็นออร์แกเนลล์ที่มีคลอโรฟิลล์เป็นองค์ประกอบหลัก คลอโรฟิลล์เป็นสารสีเขียวที่ดูดซับพลังงานจากแสงอาทิตย์และใช้ในการเร่งปฏิกิริยาเคมีที่เกี่ยวข้องกับการสังเคราะห์ด้วยแสง
กระบวนการสังเคราะห์ด้วยแสงสามารถแบ่งออกเป็น 2
```

**NVFP4:**
```
พืช สาหร่าย และแบคทีเรียบางชนิดใช้พลังงานจากแสงอาทิตย์เพื่อสร้างอาหารจากคาร์บอนไดออกไซด์และน้ำ กระบวนการนี้เกิดขึ้นในคลอโรพลาสต์ของเซลล์พืช ซึ่งเป็นออร์แกเนลล์ที่มีคลอโรฟิลล์ ซึ่งเป็นสารสีเขียวที่ดูดซับแสงอาทิตย์
กระบวนการสังเคราะห์ด้วยแสงสามารถแบ่งออกเป็นสองขั้นตอนหลัก ได้แก่ ปฏิกิริยาแสง (light reaction) และปฏิกิริยาคาร์บอน (carbon reaction) ในขั้นตอนแรก พลังงานจากแสงอาทิตย์จะ
```

## th_business
*diverges at token 4*

**Prompt:** `การวิเคราะห์งบการเงินของบริษัทประกอบด้วยขั้นตอนสำคัญดังนี้`

**BF16:**
```
1.  วิเคราะห์ฐานะการเงินและผลการดำเนินงานของบริษัท
2.  วิเคราะห์อัตราส่วนทางการเงิน
3.  วิเคราะห์กระแสเงินสด
4.  วิเคราะห์แนวโน้มผลการดำเนินงาน
5.  วิเคราะห์ฐานะการเงินและผลการดำเนินงานของบริษัทย่อย
6.  วิเคราะห์ฐานะการเงินและผลการดำเนินงานของบริษัทร่วม
7.  วิเคราะห์ฐานะการเงินและผลการดำเนินงานของบริษัทที่เกี่ยวข้องกัน
8.  วิเคราะห์ฐานะการเงินและผลการดำเนินงาน
```

**NVFP4:**
```
1. วิเคราะห์งบการเงินในอดีต
2. วิเคราะห์แนวโน้มของงบการเงิน
3. วิเคราะห์อัตราส่วนทางการเงิน
4. วิเคราะห์ฐานะทางการเงินและผลการดำเนินงานของบริษัท
5. วิเคราะห์ฐานะทางการเงินและผลการดำเนินงานของบริษัทเปรียบเทียบกับบริษัทอื่น
6. วิเคราะห์ฐานะทางการเงินและผลการดำเนินงานของบริษัทเปรียบเทียบกับอุตสาหกรรม
7. วิเคราะห์ฐานะทางการเงินและผลการดำเนินงานของบริษั
```

## th_travel
*diverges at token 1*

**Prompt:** `จังหวัดเชียงใหม่มีสถานที่ท่องเที่ยวที่มีชื่อเสียง ได้แก่`

**BF16:**
```
ดอยอินทนนท์ ดอยอ่างขาง ดอยสุเทพ ดอยปุย ดอยม่อนแจ่ม ดอยผ้าห่มปก ดอยฟ้าห่มปก ดอยอินทนนท์ ดอยอ่างขาง ดอยสุเทพ ดอยปุย ดอยม่อนแจ่ม ดอยผ้าห่มปก ดอยฟ้าห่มปก ดอยอินทนนท์ ดอยอ่างขาง ดอยสุเทพ ดอยปุย ดอยม่อนแจ่ม ดอยผ้าห่มปก ดอยฟ้าห่มปก ดอยอินทนนท์ ดอยอ่างขาง ดอยสุเทพ ดอยปุย ดอยม่อนแจ่ม ดอยผ้าห่มปก ดอยฟ
```

**NVFP4:**
```
วัดพระธาตุดอยสุเทพ วัดพระธาตุดอยสุเทพ ตั้งอยู่บนยอดดอยสุเทพ ซึ่งเป็นสถานที่ที่มีความสำคัญทางประวัติศาสตร์และวัฒนธรรมของจังหวัดเชียงใหม่ วัดพระธาตุดอยสุเทพเป็นวัดที่มีความสำคัญและเป็นที่เคารพนับถือของชาวเชียงใหม่ วัดพระธาตุดอยสุเทพเป็นวัดที่มีความสำคัญและเป็นที่เคารพนับถือของชาวเชียงใหม่ วัดพระธาตุดอยสุเทพเป็นวัดที่มีความสำคัญและเป็นที่เคารพนับถือของชาวเชียงใหม่ วัด
```

## th_math
*diverges at token 1*

**Prompt:** `โจทย์: ซื้อของราคา 250 บาท จ่ายด้วยธนบัตร 500 บาท จะได้เงินทอนเท่าไร
วิธีคิด:`

**BF16:**
```
250 + 250 = 500 ดังนั้น 500 – 250 = 250
คำตอบ: ได้เงินทอน 250 บาท
ตัวอย่างที่ 2 โจทย์: ซื้อของราคา 1,250 บาท จ่ายด้วยธนบัตร 1,000 บาท 2 ใบ จะได้เงินทอนเท่าไร
วิธีคิด: 1,250 + 750 = 2,000 ดังนั้น 2,000 – 1,250 = 750
คำตอบ: ได้เงินทอน 750 บาท
ตัวอย่างที่ 3 โจทย์: ซื้อของราคา 1,
```

**NVFP4:**
```
ใช้การลบ 500 – 250 = 250 บาท
วิธีสอน: ให้เด็กนับเงินทอนจาก 250 บาท ไปจนถึง 500 บาท
โดยนับเพิ่มทีละ 10 บาท จนครบ 5 ครั้ง แล้วนับเพิ่มทีละ 50 บาท จนครบ 2 ครั้ง
จะได้เงินทอน 250 บาท
วิธีคิด: ใช้การลบ 500 – 250 = 250 บาท
วิธีสอน: ให้เด็กนับเงินทอนจาก 250 บาท ไปจนถึง 500 บาท
โดยนับเพิ่มทีละ 10 บาท จนครบ 5
```

## en_econ
*diverges at token 4*

**Prompt:** `The main causes of inflation are`

**BF16:**
```
the increase in demand for goods and services, the increase in production costs, and the increase in the money supply. Inflation can be caused by a variety of factors, including an increase in the money supply, an increase in demand for goods and services, and an increase in production costs. Inflation can also be caused by a decrease in the supply of goods and services, or by a decrease in the demand for goods and services. Inflation can also be caused by a combination of these factors. Inflation can be caused by a variety of factors, including an increase in the money supply, an increase in demand for goods and services, and an increase in production costs. Inflation can also be caused by a decrease in the supply of goods and services, or by a decrease in the demand for goods and services. Inflation can also be caused by a combination of these factors.The main causes of inflation are the increase in demand for goods and services, the increase in production costs, and
```

**NVFP4:**
```
the increase in demand and the increase in production costs. The increase in demand is caused by the increase in the money supply, the increase in government spending, the increase in private investment, and the increase in exports. The increase in production costs is caused by the increase in wages, the increase in raw material prices, and the increase in taxes. The increase in demand and the increase in production costs lead to an increase in the price level, which is the definition of inflation. The increase in demand and the increase in production costs are the main causes of inflation.The main causes of inflation are the increase in demand and the increase in production costs. The increase in demand is caused by the increase in the money supply, the increase in government spending, the increase in private investment, and the increase in exports. The increase in production costs is caused by the increase in wages, the increase in raw material prices, and the increase in taxes. The increase in demand and the increase in production costs lead to
```

## en_science
*diverges at token 1*

**Prompt:** `Photosynthesis is the process by which`

**BF16:**
```
plants, algae and some bacteria convert light energy into chemical energy. The process involves the absorption of light by chlorophyll, which is then used to convert carbon dioxide and water into glucose and oxygen. The glucose is used as a source of energy for the plant, while the oxygen is released into the atmosphere. The process of photosynthesis is essential for the survival of plants and is the basis for the food chain.The process of photosynthesis is a complex biochemical process that occurs in plants, algae, and some bacteria. It involves the conversion of light energy into chemical energy, which is stored in the form of glucose. The process of photosynthesis can be divided into two main stages: the light-dependent reactions and the light-independent reactions. The light-dependent reactions occur in the thylakoid membranes of the chloroplasts, where light energy is absorbed by chlorophyll and used to split water molecules into oxygen and hydrogen ions. The hydrogen ions are then used to reduce NADP+ to
```

**NVFP4:**
```
plants use sunlight to convert carbon dioxide and water into glucose and oxygen. This process is essential for the growth and survival of plants, as it provides them with the energy they need to carry out their metabolic processes. The process of photosynthesis occurs in the chloroplasts of plant cells, which contain the pigment chlorophyll. Chlorophyll absorbs light energy from the sun and uses it to power the chemical reactions that convert carbon dioxide and water into glucose and oxygen. The glucose produced by photosynthesis is used by plants as a source of energy, while the oxygen is released into the atmosphere. Photosynthesis is a vital process for the survival of all life on Earth, as it provides the oxygen that we breathe and the food that we eat.The process of photosynthesis is a complex one that involves several steps. The first step is the absorption of light energy by chlorophyll, which is a pigment found in the chloroplasts of plant cells. This energy is then used to split water molecules
```

## code_python
*diverges at token 6*

**Prompt:** `def fibonacci(n):
    """Return the n-th Fibonacci number."""
`

**BF16:**
```
if n == 0 or n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10)) # 89
print(fibonacci(20)) # 10946
print(fibonacci(30)) # 1346269
print(fibonacci(40)) # 165580141
print(fibonacci(50)) # 20365011074
print(fibonacci(60)) # 2504730781961
print(fibonacci(70)) # 308061521170129
print(fibonacci(80)) # 3736710778780434
```

**NVFP4:**
```
if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

# Example usage
n = 10
result = fibonacci(n)
print(f"The {n}-th Fibonacci number is {result}")def fibonacci(n):
    """Return the n-th Fibonacci number."""
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

# Example usage
n = 10
result = fibonacci(n)
print(f"The {n}-th Fibonacci number is {result}")def fibonacci(n):
    """Return the n-th Fibonacci number."""
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci
```

## th_en_translate
*diverges at token 11*

**Prompt:** `คำว่า "ความยั่งยืน" ในภาษาอังกฤษคือ`

**BF16:**
```
"Sustainability" ซึ่งมีรากศัพท์มาจากคำว่า "Sustain" ที่แปลว่า "การดำรงอยู่" หรือ "การคงอยู่" ดังนั้น ความยั่งยืนจึงหมายถึง การดำรงอยู่อย่างต่อเนื่อง ไม่สูญสลายไป หรือการคงอยู่อย่างยั่งยืน ไม่สูญสลายไปในอนาคต ซึ่งเป็นแนวคิดที่เกี่ยวข้องกับการพัฒนาที่ยั่งยืน (Sustainable Development) ที่เน้นการพัฒนาที่คำนึงถึงความสมดุลระหว่างเศรษฐกิจ สังคม และสิ่งแวดล้อม เพื่อให้เกิดการ
```

**NVFP4:**
```
"Sustainability" ซึ่งมีความหมายว่า "ความยั่งยืน" หรือ "ความยั่งยืน" ซึ่งหมายถึงความสามารถในการรักษาหรือรักษาสภาพหรือสถานะของสิ่งใดสิ่งหนึ่งไว้ได้เป็นระยะเวลานาน โดยไม่เกิดความเสียหายหรือเสื่อมสภาพ ซึ่งอาจหมายถึงการรักษาสภาพแวดล้อม การรักษาความมั่นคงของสังคม หรือการรักษาความมั่นคงของเศรษฐกิจ ซึ่งเป็นสิ่งที่สำคัญในการพัฒนาและเติบโตของประเทศและโลกในระยะยาว"ความยั่งยืน" ในภาษาอังกฤษคือ "S
```
