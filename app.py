
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'model.joblib'
st.set_page_config(page_title='CC General | Segmentasi Pelanggan', page_icon='📊', layout='wide')


@st.cache_resource(show_spinner='Memuat model segmentasi...')
def load_bundle(path, modification_time_ns):
    """Model dimuat sekali; waktu modifikasi membatalkan cache saat model diganti."""
    bundle = joblib.load(path)
    required = {'pipeline', 'label_map', 'metadata', 'reports'}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError('Format model tidak sesuai. Ekspor ulang model dari notebook revisi.')
    return bundle


try:
    bundle = load_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
except FileNotFoundError:
    st.error('model.joblib belum ditemukan. Jalankan seluruh notebook, lalu letakkan hasil model.joblib di folder yang sama dengan app.py.')
    st.stop()
except (ValueError, ImportError, AttributeError, EOFError) as error:
    st.error(f'Model tidak dapat dimuat: {error}. Gunakan requirements.txt yang dihasilkan notebook.')
    st.stop()

meta = bundle['metadata']
reports = bundle['reports']
profile = reports['profile']
FEATURES = meta['features']
DESCRIPTIONS = meta['feature_descriptions']
RATIO_FEATURES = meta['ratio_features']
COUNT_FEATURES = meta['count_features']
SOURCE_URL = meta['source_url']


def validate_features(frame):
    """Validasi skema, tipe, dan domain; NaN diimputasi oleh pipeline terlatih."""
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError('Kolom wajib belum tersedia: ' + ', '.join(missing))
    if frame.empty:
        raise ValueError('Data tidak boleh kosong.')
    if len(frame) > 20000:
        raise ValueError('Maksimum 20.000 baris per prediksi.')
    selected = frame.loc[:, FEATURES]
    numeric = selected.apply(pd.to_numeric, errors='coerce')
    invalid = selected.notna() & numeric.isna()
    if invalid.any().any():
        raise ValueError('Nilai bukan angka pada: ' + ', '.join(invalid.columns[invalid.any()]))
    if numeric.isna().all(axis=1).any():
        raise ValueError('Ada baris yang seluruh fiturnya kosong.')
    if np.isinf(numeric.to_numpy()).any() or (numeric < 0).any().any():
        raise ValueError('Nilai negatif atau tak terhingga tidak diperbolehkan.')
    if (numeric[RATIO_FEATURES] > 1).any().any():
        raise ValueError('Fitur proporsi harus berada pada rentang 0–1.')
    for col in COUNT_FEATURES:
        valid = numeric[col].dropna().to_numpy()
        if not np.allclose(valid, np.round(valid)):
            raise ValueError(f'{col} harus berupa bilangan bulat.')
    if (numeric['TENURE'].dropna() <= 0).any():
        raise ValueError('TENURE harus lebih besar dari nol.')
    return numeric.astype(float)


def predict_customers(bundle, frame):
    x = validate_features(frame)
    pipeline = bundle['pipeline']
    labels = pipeline.predict(x)
    distances = pipeline.transform(x).min(axis=1)
    result = frame.copy()
    result['Segment'] = [bundle['label_map'][int(i)] for i in labels]
    result['Nama_Segmen'] = result.Segment.map(bundle['metadata']['segment_names'])
    result['Jarak_Centroid'] = distances
    return result


st.title('Segmentasi Pelanggan Kartu Kredit')
st.caption('CC General • K-Means • Project CRISP-DM Pertemuan 4')
page = st.sidebar.radio('Menu', ['Ringkasan', 'Prediksi pelanggan', 'Prediksi CSV', 'Metodologi'])
st.sidebar.caption(f"{meta['rows']:,} pelanggan · {len(FEATURES)} fitur · {meta['best_k']} segmen")
st.sidebar.markdown(f'[Sumber dataset]({SOURCE_URL})')

if page == 'Ringkasan':
    metrics = st.columns(4)
    metrics[0].metric('Pelanggan', f"{meta['rows']:,}")
    metrics[1].metric('Jumlah segmen', meta['best_k'])
    metrics[2].metric('Silhouette holdout', f"{meta['holdout_metrics']['silhouette']:.3f}")
    metrics[3].metric('Fitur perilaku', len(FEATURES))
    st.subheader('Profil setiap segmen')
    st.dataframe(profile[['Segment', 'Nama_Segmen', 'Jumlah', 'Persentase', 'PURCHASES',
                          'CASH_ADVANCE', 'PURCHASES_FREQUENCY', 'PRC_FULL_PAYMENT']].round(3),
                 hide_index=True)
    st.caption('Nilai indikator merupakan rata-rata pada skala asli dataset. Nama segmen bersifat relatif.')
    left, right = st.columns([1, 2])
    with left:
        st.subheader('Jumlah pelanggan')
        st.bar_chart(profile.set_index('Segment')['Jumlah'])
    with right:
        st.subheader('Peta kemiripan pelanggan')
        projection = reports['projection']
        st.scatter_chart(projection, x='PC1', y='PC2', color='Segment', height=350)
        st.caption(f"PCA dua dimensi merangkum {sum(meta['pca_explained_variance']):.1%} variasi; model tetap memakai 17 fitur.")
    st.download_button('Unduh hasil segmentasi', reports['assignments'].to_csv(index=False).encode('utf-8'),
                       file_name='hasil_segmentasi_cc_general.csv', mime='text/csv')

elif page == 'Prediksi pelanggan':
    st.subheader('Temukan segmen pelanggan')
    st.write('Masukkan indikator dengan definisi dan periode pengamatan yang sama seperti dataset. '
             'Nilai awal adalah median dataset; nominal mengikuti satuan pada data sumber.')
    with st.form('customer_form'):
        columns = st.columns(3)
        values = {}
        for i, feature in enumerate(FEATURES):
            with columns[i % 3]:
                default = float(meta['input_defaults'][feature])
                if feature in COUNT_FEATURES:
                    values[feature] = st.number_input(feature, min_value=1 if feature == 'TENURE' else 0,
                                                       value=int(round(default)), step=1,
                                                       help=DESCRIPTIONS[feature])
                elif feature in RATIO_FEATURES:
                    values[feature] = st.number_input(feature, min_value=0.0, max_value=1.0,
                                                       value=default, step=0.01, format='%.6f',
                                                       help=DESCRIPTIONS[feature])
                else:
                    values[feature] = st.number_input(feature, min_value=0.0, value=default,
                                                       step=0.01 if feature == 'CASH_ADVANCE_FREQUENCY' else 10.0,
                                                       help=DESCRIPTIONS[feature])
        submitted = st.form_submit_button('Tentukan segmen', type='primary')
    if submitted:
        try:
            result = predict_customers(bundle, pd.DataFrame([values]))
            row = result.iloc[0]
            st.success(f"{row['Segment']} — {row['Nama_Segmen']}")
            st.caption(f"Jarak ke pusat segmen: {row['Jarak_Centroid']:.3f}. Jarak ini bukan probabilitas atau tingkat keyakinan.")
            outside = [c for c in FEATURES if values[c] < meta['input_min'][c] or values[c] > meta['input_max'][c]]
            if outside:
                st.warning('Nilai di luar rentang data pelatihan: ' + ', '.join(outside))
            st.dataframe(profile.loc[
                lambda table: table.Segment.eq(row.Segment)], hide_index=True)
        except ValueError as error:
            st.error(str(error))

elif page == 'Prediksi CSV':
    st.subheader('Segmentasi banyak pelanggan')
    st.write('CSV harus memuat 17 kolom fitur; CUST_ID bersifat opsional. Maksimum 20.000 baris. '
             'Nilai kosong akan diisi oleh median yang telah dipelajari model.')
    template = reports['example_customers'].copy()
    st.download_button('Unduh contoh CSV', template.to_csv(index=False).encode('utf-8'),
                       file_name='contoh_pelanggan.csv', mime='text/csv')
    uploaded = st.file_uploader('Pilih CSV pelanggan', type=['csv'])
    use_example = st.checkbox('Gunakan 20 pelanggan contoh')
    if uploaded is not None or use_example:
        try:
            frame = template if use_example else pd.read_csv(uploaded)
            result = predict_customers(bundle, frame)
            st.success(f'{len(result):,} pelanggan berhasil dikelompokkan.')
            st.dataframe(result.head(100), hide_index=True)
            st.download_button('Unduh hasil prediksi', result.to_csv(index=False).encode('utf-8'),
                               file_name='prediksi_segmen.csv', mime='text/csv')
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(str(error))

else:
    st.subheader('Metode dan evaluasi')
    st.markdown('Tujuan: mengenali pola penggunaan kartu untuk menyusun hipotesis layanan '
                'dan komunikasi yang sesuai dengan perilaku kelompok pelanggan.')
    st.markdown('**Alur model:** validasi → imputasi median → log1p pada nominal dan jumlah '
                'transaksi → StandardScaler → K-Means. CUST_ID tidak digunakan sebagai fitur.')
    st.dataframe(reports['model_selection'].round(4), hide_index=True)
    st.write('Jumlah cluster dipilih menggunakan silhouette pada sampel 2.500 baris data latih. '
             'Data holdout 20% baru dipakai setelah pemilihan cluster. Model final telah '
             'dilatih ulang pada seluruh dataset di notebook. Aplikasi hanya memuat model.joblib.')
    st.json({'evaluasi_holdout': meta['holdout_metrics'], 'stabilitas_inisialisasi': meta['stability']})
    st.markdown('**Batas interpretasi:** segmen merupakan pola deskriptif, bukan label risiko '
                'kredit, fraud, atau kemampuan membayar. Efektivitas strategi layanan perlu diuji '
                'secara terpisah. Pantau perubahan distribusi data dan proporsi segmen setelah deployment.')
    st.caption(f"Terdapat {meta['cash_advance_frequency_above_one']} nilai CASH_ADVANCE_FREQUENCY > 1 pada sumber. Nilai dipertahankan "
               'karena definisi normalisasi perlu dikonfirmasi; tidak diubah secara otomatis.')
