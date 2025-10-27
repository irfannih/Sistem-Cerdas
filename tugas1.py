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

print("Data training:")
print(df)
print()

# ======================
# 2️⃣ Hitung probabilitas prior P(Buy)
# ======================
p_buy = df['Buy'].value_counts(normalize=True)['Yes']
p_not_buy = df['Buy'].value_counts(normalize=True)['No']

print(f"P(Buy) = {p_buy:.2f}")
print(f"P(Not Buy) = {p_not_buy:.2f}")
print()

# ======================
# 3️⃣ Fungsi bantu: hitung probabilitas kondisi
# ======================
def conditional_prob(attribute, value, target_value):
    subset = df[df['Buy'] == target_value]
    return len(subset[subset[attribute] == value]) / len(subset)

# ======================
# 4️⃣ Rumus Naive Bayes
# ======================
def naive_bayes(day, delivery, discount, target):
    # P(Buy=target | fitur) ∝ P(Buy=target)*P(Day|target)*P(Delivery|target)*P(Discount|target)
    p_target = p_buy if target == 'Yes' else p_not_buy
    p_day = conditional_prob('Day', day, target)
    p_delivery = conditional_prob('FreeDelivery', delivery, target)
    p_discount = conditional_prob('Discount', discount, target)
    
    return p_target * p_day * p_delivery * p_discount

# ======================
# 5️⃣ Hitung semua kasus dari soal
# ======================
cases = [
    ('Weekday','Yes','Yes'),
    ('Weekday','No','No'),
    ('Weekday','Yes','Yes'),
    ('Weekday','No','No'),
    ('Weekend','Yes','Yes'),
    ('Weekend','No','No'),
    ('Weekend','Yes','Yes'),
    ('Weekend','No','No'),
]

print("Hasil Probabilitas:")
for i, (day, delivery, discount) in enumerate(cases, start=1):
    p_buy_case = naive_bayes(day, delivery, discount, 'Yes')
    p_not_case = naive_bayes(day, delivery, discount, 'No')
    print(f"{i}. P(Buy | {day}, {delivery}, {discount}) = {p_buy_case:.6f}")
    print(f"   P(Not Buy | {day}, {delivery}, {discount}) = {p_not_case:.6f}")
    print()
