import pandas as pd

# ======================
# 1️⃣ Dataset contoh
# ======================
data = {
    'Day': ['Weekday','Weekday','Weekday','Weekend','Weekend','Weekend','Weekday','Weekend'],
    'FreeDelivery': ['Yes','No','Yes','Yes','No','No','Yes','Yes'],
    'Discount': ['Yes','No','No','Yes','No','Yes','Yes','No'],
    'Buy': ['Yes','No','Yes','Yes','No','No','Yes','No']
}

df = pd.DataFrame(data)

print("📊 Data Training:")
print(df)
print("\n====================================\n")

# ======================
# 2️⃣ Hitung probabilitas prior P(Buy)
# ======================
p_buy = df['Buy'].value_counts(normalize=True)['Yes']
p_not_buy = df['Buy'].value_counts(normalize=True)['No']

print(f"P(Buy) = {p_buy:.2%}")
print(f"P(Not Buy) = {p_not_buy:.2%}")
print("\n====================================\n")

# ======================
# 3️⃣ Fungsi bantu
# ======================
def conditional_prob(attribute, value, target_value):
    subset = df[df['Buy'] == target_value]
    return len(subset[subset[attribute] == value]) / len(subset)

# ======================
# 4️⃣ Rumus Naive Bayes
# ======================
def naive_bayes(day, delivery, discount, target):
    p_target = p_buy if target == 'Yes' else p_not_buy
    p_day = conditional_prob('Day', day, target)
    p_delivery = conditional_prob('FreeDelivery', delivery, target)
    p_discount = conditional_prob('Discount', discount, target)
    return p_target * p_day * p_delivery * p_discount

# ======================
# 5️⃣ Kasus uji (5 baris)
# ======================
cases = [
    ('Weekday', 'Yes', 'Yes'),
    ('Weekday', 'No', 'No'),
    ('Weekend', 'Yes', 'Yes'),
    ('Weekend', 'No', 'No'),
    ('Weekend', 'Yes', 'No')
]

# ======================
# 6️⃣ Hasil Probabilitas & Prediksi
# ======================
print("📈 Hasil Probabilitas & Prediksi:")
for i, (day, delivery, discount) in enumerate(cases, start=1):
    p_buy_case = naive_bayes(day, delivery, discount, 'Yes')
    p_not_case = naive_bayes(day, delivery, discount, 'No')
    prediction = 'Buy' if p_buy_case > p_not_case else 'Not Buy'

    print(f"{i}. ({day}, {delivery}, {discount})")
    print(f"   → P(Buy)     = {p_buy_case:.4%}")
    print(f"   → P(Not Buy) = {p_not_case:.4%}")
    print(f"   👉 Prediksi: {prediction}")
    print()
