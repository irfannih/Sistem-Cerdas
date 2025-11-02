import pandas as pd

# ======================
# 1️⃣ Baca data dari CSV
# ======================
df = pd.read_csv('shooping_demo.csv')

print("📄 Data dari file 'shooping_demo.csv':")
print(df.head(5))  # hanya tampilkan 4 baris pertama
print("\n=========================================\n")

# Normalisasi nama kolom agar aman
df.columns = df.columns.str.strip().str.lower()

print("🧾 Kolom yang terbaca:", list(df.columns))
print("\n=========================================\n")

# ======================
# 2️⃣ Hitung probabilitas prior P(Buy)
# ======================
p_buy = df['buy'].value_counts(normalize=True)['Yes']
p_not_buy = df['buy'].value_counts(normalize=True)['No']

print(f"P(Buy) = {p_buy:.2%}")
print(f"P(Not Buy) = {p_not_buy:.2%}")
print("\n=========================================\n")

# ======================
# 3️⃣ Fungsi bantu
# ======================
def conditional_prob(attribute, value, target_value):
    subset = df[df['buy'] == target_value]
    return len(subset[subset[attribute] == value]) / len(subset)

# ======================
# 4️⃣ Rumus Naive Bayes
# ======================
def naive_bayes(day, delivery, discount, target):
    p_target = p_buy if target == 'Yes' else p_not_buy
    p_day = conditional_prob('day', day, target)
    p_delivery = conditional_prob('free delivery', delivery, target)  # pakai spasi
    p_discount = conditional_prob('discount', discount, target)
    return p_target * p_day * p_delivery * p_discount

# ======================
# 5️⃣ Kasus uji
# ======================
cases = [
    ('Weekday','Yes','Yes'),
    ('Weekday','No','No'),
    ('Weekend','Yes','Yes'),
    ('Weekend','No','No'),
]

# ======================
# 6️⃣ Simpan hasil ke DataFrame
# ======================
results = []
for (day, delivery, discount) in cases:
    p_buy_case = naive_bayes(day, delivery, discount, 'Yes')
    p_not_case = naive_bayes(day, delivery, discount, 'No')
    prediction = 'Buy' if p_buy_case > p_not_case else 'Not Buy'
    results.append({
        'Day': day,
        'Free Delivery': delivery,
        'Discount': discount,
        'P(Buy)': f"{p_buy_case*100:.2f}%",
        'P(Not Buy)': f"{p_not_case*100:.2f}%",
        'Prediction': prediction
    })

# ======================
# 7️⃣ Tampilkan tabel hasil
# ======================
results_df = pd.DataFrame(results)
print("Hasil Probabilitas dan Prediksi (dalam Persen):")
print(results_df.to_string(index=False))
