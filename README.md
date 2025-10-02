# Instagram Engagement Prediction

Proyek penelitian untuk menganalisis engagement Instagram dari akun @fst_unja (Fakultas Sains dan Teknologi Universitas Jambi).

## Data

Dataset berisi **271 posts** dari Instagram @fst_unja dengan metadata lengkap:
- Post ID, shortcode, URL
- Tanggal posting
- Jumlah likes
- Caption, hashtags, mentions
- Tipe konten (foto/video)

**File Hasil:** `fst_unja_from_gallery_dl.csv`

## Statistik

- Total posts: 271
- Total likes: 69,426
- Rata-rata likes: 256.18 per post
- Konten: 219 foto, 52 video

## Tools

### Gallery-dl
Tool utama untuk download posts Instagram dengan metadata.

**Instalasi:**
```bash
python -m venv venv
source venv/bin/activate
pip install gallery-dl
```

**Konfigurasi:** `config.json`
```json
{
    "extractor": {
        "instagram": {
            "include": "posts",
            "metadata": true,
            "videos": true,
            "image-filter": "username == 'fst_unja'"
        }
    }
}
```

**Menjalankan:**
```bash
gallery-dl --config config.json https://www.instagram.com/fst_unja/
```

### Ekstraksi Data
Script untuk mengekstrak metadata dari JSON files ke CSV.

```bash
python extract_from_gallery_dl.py
```

## Catatan

- Data untuk keperluan penelitian akademik
- Hanya posts publik dari @fst_unja
- Comments tidak tersedia (keterbatasan API)
