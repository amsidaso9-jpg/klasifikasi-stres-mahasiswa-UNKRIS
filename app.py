# =====================================================================
# APLIKASI KLASIFIKASI TINGKAT STRES MAHASISWA — STREAMLIT
# Berbasis PSS-10 + Variabel Konteks Akademik | Universitas Krisnadwipayana Jakarta
# Model: Support Vector Machine (SVM) — kernel linear, C=100, gamma='scale'
# =====================================================================

import streamlit as st
import joblib
import numpy as np

# =====================================================================
# KONFIGURASI HALAMAN
# =====================================================================
st.set_page_config(
    page_title="Deteksi Stres Mahasiswa UNKRIS",
    page_icon="🧠",
    layout="centered"
)

# =====================================================================
# LOAD MODEL, SCALER, DAN MAPPING ENCODING
# =====================================================================
@st.cache_resource
def load_semua_file():
    """
    Load model SVM (Pipeline lengkap), scaler, dan mapping encoding.
    Pastikan file berikut ada di folder yang sama dengan app.py:
      - model_stres_terbaik.pkl
      - scaler_minmax.pkl
      - mapping_encoding.pkl
    """
    try:
        model   = joblib.load('model_stres_terbaik.pkl')
        scaler  = joblib.load('scaler_minmax.pkl')
        mapping = joblib.load('mapping_encoding.pkl')
        return model, scaler, mapping, None
    except FileNotFoundError as e:
        return None, None, None, str(e)

model, scaler, mapping, error_msg = load_semua_file()

# =====================================================================
# HEADER APLIKASI
# =====================================================================
st.title("🧠 Aplikasi Klasifikasi Tingkat Stres Mahasiswa")
st.markdown("**Universitas Krisnadwipayana Jakarta**")
st.write("Isi kuesioner di bawah ini sesuai kondisi yang Anda rasakan **dalam satu bulan terakhir**.")

if error_msg:
    st.error(f"❌ File tidak ditemukan: `{error_msg}`")
    st.info("Pastikan file `model_stres_terbaik.pkl`, `scaler_minmax.pkl`, dan "
            "`mapping_encoding.pkl` ada di folder yang sama dengan `app.py`.")
    st.stop()

st.write("---")

# =====================================================================
# ENCODING MAPPING (sesuai proses training di Colab)
# =====================================================================
# ipk_map, tidur_map, olah_map diambil dari mapping_encoding.pkl (Cell 15)
ipk_map   = mapping['ipk_map']     # {'2.50 - 2.99': 1, '3.00 - 3.49': 2, '3.50 - 4.00': 3}
tidur_map = mapping['tidur_map']   # {'< 4 Jam': 1, '4 - 5 Jam': 2, '6 - 7 Jam': 3, '> 7 Jam': 4}
olah_map  = mapping['olah_map']    # {'Tidak Pernah': 0, '1 - 2 Kali': 1, '3 - 4 Kali': 2, 'Setiap Hari': 3}
label_map = mapping['label_map']   # {0: 'Rendah', 1: 'Sedang', 2: 'Tinggi'}

# encode_aktivitas() di Colab adalah fungsi if/elif, bukan dictionary sederhana,
# sehingga logikanya ditulis ulang langsung di sini (bukan dari mapping_encoding.pkl)
aktivitas_map = {
    'Tidak Ada'       : 0,
    'Organisasi'      : 1,
    'Kerja Part-time' : 2,
    'Lainnya'         : 3   
}

# =====================================================================
# PANDUAN SKALA PSS-10
# =====================================================================
with st.expander("📖 Panduan Pengisian Skala PSS-10"):
    st.markdown("""
    | Nilai | Keterangan |
    |-------|------------|
    | 0     | Tidak Pernah |
    | 1     | Hampir Tidak Pernah |
    | 2     | Kadang-kadang |
    | 3     | Cukup Sering |
    | 4     | Sangat Sering |
    """)

# =====================================================================
# BAGIAN 1 — FORM INPUT PSS-10 (10 ITEM)
# =====================================================================
st.subheader("1️⃣ Kuesioner PSS-10")

pertanyaan = {
    "PSS1" : "1. Dalam sebulan terakhir seberapa sering Anda merasa kecewa karena sesuatu yang terjadi secara tidak terduga?",
    "PSS2" : "2. Dalam sebulan terakhir seberapa sering Anda merasa tidak mampu mengendalikan hal-hal penting dalam hidup Anda?",
    "PSS3" : "3. Dalam sebulan terakhir seberapa sering Anda merasa gelisah dan stres?",
    "PSS4" : "4. Dalam sebulan terakhir seberapa sering Anda merasa percaya diri dengan kemampuan Anda menyelesaikan masalah pribadi?",
    "PSS5" : "5. Dalam sebulan terakhir seberapa sering Anda merasa segala sesuatu berjalan sesuai keinginan Anda?",
    "PSS6" : "6. Dalam sebulan terakhir seberapa sering Anda merasa tidak bisa mengatasi hal-hal yang harus Anda lakukan?",
    "PSS7" : "7. Dalam sebulan terakhir seberapa sering Anda mampu mengendalikan hal-hal yang menjengkelkan dalam hidup Anda?",
    "PSS8" : "8. Dalam sebulan terakhir seberapa sering Anda merasa mampu mengendalikan permasalahan Anda?",
    "PSS9" : "9. Dalam sebulan terakhir seberapa sering Anda marah karena hal-hal yang terjadi di luar kendali Anda?",
    "PSS10": "10. Dalam sebulan terakhir seberapa sering Anda merasa tidak mampu menyelesaikan permasalahan yang menumpuk?",
}

label_skala = {0: "0 – Tidak Pernah", 1: "1 – Hampir Tidak Pernah",
               2: "2 – Kadang-kadang", 3: "3 – Cukup Sering", 4: "4 – Sangat Sering"}

jawaban_pss = {}
for kode, teks in pertanyaan.items():
    st.markdown(f"**{teks}**")
    jawaban_pss[kode] = st.select_slider(
        label=kode,
        options=[0, 1, 2, 3, 4],
        value=2,
        format_func=lambda x: label_skala[x],
        label_visibility="collapsed",
        key=f"slider_{kode}"
    )
    st.write("")

st.write("---")

# =====================================================================
# BAGIAN 2 — VARIABEL KONTEKS AKADEMIK (4 VARIABEL)
# =====================================================================
st.subheader("2️⃣ Konteks Akademik & Gaya Hidup")

ipk_pilihan = st.selectbox(
    "IPK Terakhir Anda:",
    options=list(ipk_map.keys())
)

aktivitas_pilihan = st.selectbox(
    "Apakah Anda memiliki aktivitas selain kuliah?",
    options=list(aktivitas_map.keys())
)

tidur_pilihan = st.selectbox(
    "Rata-rata jam tidur Anda per malam:",
    options=list(tidur_map.keys())
)

olahraga_pilihan = st.selectbox(
    "Seberapa sering Anda berolahraga dalam seminggu?",
    options=list(olah_map.keys())
)

st.write("---")

# =====================================================================
# BAGIAN 3 — SUMBER STRES (3 VARIABEL, BISA PILIH LEBIH DARI 1)
# =====================================================================
st.subheader("3️⃣ Sumber Stres Utama")
st.caption("Boleh pilih lebih dari satu.")

col_a, col_b, col_c = st.columns(3)
with col_a:
    stres_skripsi = st.checkbox("Skripsi / Tugas Akhir")
with col_b:
    stres_keuangan = st.checkbox("Keuangan")
with col_c:
    stres_akademik = st.checkbox("Tugas & Beban Akademik Lain")

st.write("---")

# =====================================================================
# TOMBOL PREDIKSI
# =====================================================================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    tombol = st.button("🔍 Cek Tingkat Stres Saya", type="primary", use_container_width=True)

if tombol:
    # --- 1. Kumpulkan & reverse scoring PSS-10 ---
    pss_values = [jawaban_pss[f"PSS{i}"] for i in range(1, 11)]
    pss_reversed = pss_values.copy()
    for idx in [3, 4, 6, 7]:   # PSS4, PSS5, PSS7, PSS8 (index ke-4,5,7,8 -> 0-based 3,4,6,7)
        pss_reversed[idx] = 4 - pss_reversed[idx]

    total_skor = sum(pss_reversed)

    # --- 2. Encoding variabel konteks ---
    ipk_enc       = ipk_map[ipk_pilihan]
    aktivitas_enc = aktivitas_map[aktivitas_pilihan]
    tidur_enc     = tidur_map[tidur_pilihan]
    olahraga_enc  = olah_map[olahraga_pilihan]

    # --- 3. Encoding sumber stres (binary 0/1) ---
    stres_skripsi_enc  = int(stres_skripsi)
    stres_keuangan_enc = int(stres_keuangan)
    stres_akademik_enc = int(stres_akademik)

    # --- 4. Susun 17 fitur SESUAI URUTAN SAAT TRAINING ---
    # Urutan wajib: 10 PSS (reversed) + IPK + Aktivitas + Tidur + Olahraga
    #               + Stres_Skripsi + Stres_Keuangan + Stres_Akademik
    input_lengkap = np.array([pss_reversed + [
        ipk_enc, aktivitas_enc, tidur_enc, olahraga_enc,
        stres_skripsi_enc, stres_keuangan_enc, stres_akademik_enc
    ]])

    # --- 5. Normalisasi dengan scaler yang sama dari Colab ---
    input_scaled = scaler.transform(input_lengkap)

    # --- 6. Prediksi ---
    prediksi     = model.predict(input_scaled)[0]
    probabilitas = model.predict_proba(input_scaled)[0]

    # =====================================================================
    # TAMPILKAN HASIL
    # =====================================================================
    st.subheader("📊 Hasil Analisis Tingkat Stres")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Total Skor PSS-10", f"{total_skor} / 40")
    with col_b:
        kategori_skor = (
            "Rendah (0–13)" if total_skor <= 13
            else "Sedang (14–26)" if total_skor <= 26
            else "Tinggi (27–40)"
        )
        st.metric("Kategori Skor PSS-10", kategori_skor)

    st.write("")

    if prediksi == 0:
        st.success("### ✅ Tingkat Stres Anda: RENDAH")
        st.info("""
        **Interpretasi:** Anda berada dalam kondisi psikologis yang baik.
        Tekanan yang Anda rasakan masih dalam batas yang wajar dan dapat dikelola.

        **Saran:**
        - Pertahankan kebiasaan positif yang sudah Anda lakukan
        - Tetap luangkan waktu untuk relaksasi di sela aktivitas perkuliahan
        - Jaga pola tidur dan olahraga rutin
        """)

    elif prediksi == 1:
        st.warning("### ⚠️ Tingkat Stres Anda: SEDANG")
        st.info("""
        **Interpretasi:** Anda mengalami tekanan yang cukup berarti dan perlu diperhatikan
        sebelum berkembang menjadi stres berat.

        **Saran:**
        - Kelola waktu pengerjaan skripsi dengan jadwal harian yang realistis
        - Ceritakan beban yang Anda rasakan kepada teman dekat atau keluarga
        - Batasi konsumsi kafein dan perbaiki kualitas tidur
        - Manfaatkan sesi bimbingan dengan dosen pembimbing secara rutin
        """)

    else:
        st.error("### 🚨 Tingkat Stres Anda: TINGGI")
        st.info("""
        **Interpretasi:** Anda mengalami tekanan yang signifikan dan memerlukan perhatian segera.

        **Saran:**
        - Segera hubungi layanan konseling atau psikolog kampus UNKRIS
        - Bicarakan kondisi Anda dengan dosen pembimbing atau dosen wali
        - Prioritaskan istirahat karena produktivitas tidak akan optimal dalam kondisi stres tinggi
        - Jangan ragu untuk meminta bantuan, mencari bantuan adalah tanda keberanian
        """)

    st.write("")
    st.markdown("**Distribusi Probabilitas Prediksi Model:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rendah", f"{probabilitas[0]*100:.1f}%")
    with col2:
        st.metric("Sedang", f"{probabilitas[1]*100:.1f}%")
    with col3:
        st.metric("Tinggi", f"{probabilitas[2]*100:.1f}%")

    st.write("")
    with st.expander("ℹ️ Catatan Metodologis"):
        st.markdown(f"""
        - Instrumen: **PSS-10** (Cohen et al., 1983), diperkaya dengan **7 variabel konteks
          akademik & sumber stres**, total **17 variabel input**
        - Algoritma klasifikasi: **Support Vector Machine** (kernel linear, C=100, gamma='scale')
          — model terbaik dari hasil perbandingan (Akurasi 91,89%, F1-Score Macro 0,9261)
        - Reverse scoring diterapkan pada item: PSS4, PSS5, PSS7, PSS8
        - SMOTE diterapkan di dalam Pipeline selama pelatihan untuk menangani ketidakseimbangan
          kelas, tanpa memengaruhi proses prediksi data baru ini
        - Input dinormalisasi menggunakan **Min-Max Scaler** yang sama dengan proses pelatihan
        - Aplikasi ini merupakan **Proof of Concept** bukan pengganti diagnosis klinis profesional
        """)

# =====================================================================
# FOOTER
# =====================================================================
st.write("---")
st.caption("Aplikasi ini dikembangkan sebagai bagian dari penelitian skripsi | "
           "Ambrosia Woga Daso | Universitas Krisnadwipayana Jakarta | 2026")
