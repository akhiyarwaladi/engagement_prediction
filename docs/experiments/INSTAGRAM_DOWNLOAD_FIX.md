# Instagram Download Fix - gallery-dl dengan Cookies

**Date:** October 4, 2025  
**Status:** ✅ WORKING - 396 posts downloaded successfully

---

## 🚨 PROBLEM

Instagram API tidak lagi mendukung login dengan **username/password** di gallery-dl.

```bash
# ❌ TIDAK BEKERJA (401 Unauthorized)
gallery-dl --config config.json https://www.instagram.com/fst_unja/
```

**Error:**
```
[instagram][error] HttpError: '401 Unauthorized'
```

---

## ✅ SOLUTION YANG BEKERJA

### Command Final (Tested October 4, 2025):

```bash
gallery-dl --cookies cookies.txt --write-metadata --filter "username == 'fst_unja'" https://www.instagram.com/fst_unja/
```

**Hasil:** 
- ✅ 396 media files (photos + videos)
- ✅ 396 JSON metadata files
- ✅ Hanya @fst_unja (tidak termasuk akun yang di-tag)

---

## 🔑 PENJELASAN PARAMETER

| Parameter | Fungsi | Penting? |
|-----------|--------|----------|
| `--cookies cookies.txt` | Authentication menggunakan cookies dari browser | ✅ WAJIB |
| `--write-metadata` | Generate file JSON metadata per post | ✅ WAJIB |
| `--filter "username == 'fst_unja'"` | Filter HANYA @fst_unja, tidak ikut akun tagged | ✅ SANGAT PENTING |
| `https://www.instagram.com/fst_unja/` | URL Instagram target account | ✅ WAJIB |

---

## 📥 CARA MEMBUAT cookies.txt

### Method Manual (RECOMMENDED):

1. **Login ke Instagram** di browser (Chrome/Edge)
   - URL: https://www.instagram.com

2. **Tekan F12** → Tab **Application** (Chrome) atau **Storage** (Firefox)

3. **Klik Cookies** → `https://www.instagram.com`

4. **Copy nilai 3 cookies penting:**
   - `sessionid` ← PALING PENTING!
   - `csrftoken`
   - `ds_user_id`

5. **Buat file `cookies.txt`** dengan format Netscape:

```
# Netscape HTTP Cookie File
.instagram.com	TRUE	/	TRUE	0	sessionid	PASTE_SESSIONID_DISINI
.instagram.com	TRUE	/	TRUE	0	csrftoken	PASTE_CSRFTOKEN_DISINI
.instagram.com	TRUE	/	TRUE	0	ds_user_id	PASTE_DS_USER_ID_DISINI
```

**⚠️ PENTING:** Gunakan **TAB** (bukan spasi) sebagai separator antar kolom!

6. **Save sebagai `cookies.txt`** di folder project

---

## 🔍 KENAPA FILTER PENTING?

### Tanpa Filter:

```bash
# ❌ Download SEMUA akun yang di-tag di post @fst_unja
gallery-dl --cookies cookies.txt --write-metadata https://www.instagram.com/fst_unja/
```

**Hasil:** Download dari banyak akun:
- @fst_unja ✅
- @analiskimia_unja ❌ (tagged di post)
- @teknikgeologi_fstunja ❌ (tagged)
- @pkkmbfst_unja ❌ (tagged)
- @lppm.unja ❌ (tagged)
- @bem_fst.unja ❌ (tagged)
- Dan lainnya...

### Dengan Filter:

```bash
# ✅ Download HANYA @fst_unja
gallery-dl --cookies cookies.txt --write-metadata --filter "username == 'fst_unja'" https://www.instagram.com/fst_unja/
```

**Hasil:** Download HANYA dari:
- @fst_unja ✅ ONLY

---

## 📊 HASIL DOWNLOAD

**Directory:**
```
C:\Users\MyPC PRO\Documents\engagement_prediction\gallery-dl\instagram\fst_unja\
```

**Total Files:**
- **396 media files** (`.jpg`, `.mp4`, `.webp`)
- **396 JSON metadata files** (`.json`)

**Breakdown:**
- Photos: ~350 files
- Videos: ~46 files

**File Naming:**
- Single post: `3706084487405513439.mp4`
- Multi-post (carousel): `3687220646804788732_3687220633517176405.jpg`

---

## 🛠️ TROUBLESHOOTING

### Error: 401 Unauthorized

**Penyebab:**
- Sessionid expired
- Cookies tidak valid
- Tidak login di browser

**Solusi:**
1. Login ulang ke Instagram di browser
2. Export cookies baru
3. Ganti nilai `sessionid` di `cookies.txt`

### Error: Permission Denied (Chrome cookies)

**Penyebab:**
- Browser masih berjalan (file cookies terkunci)

**Solusi:**
```bash
# Option 1: Tutup semua browser dulu
# Option 2: Gunakan method manual (export cookies.txt)
```

### Download banyak akun (bukan hanya @fst_unja)

**Penyebab:**
- Tidak ada filter `username`

**Solusi:**
```bash
# Tambahkan --filter "username == 'fst_unja'"
gallery-dl --cookies cookies.txt --write-metadata --filter "username == 'fst_unja'" https://www.instagram.com/fst_unja/
```

### Tidak ada file JSON metadata

**Penyebab:**
- Lupa flag `--write-metadata`

**Solusi:**
```bash
# Tambahkan --write-metadata
gallery-dl --cookies cookies.txt --write-metadata --filter "username == 'fst_unja'" https://www.instagram.com/fst_unja/
```

---

## 🔄 NEXT STEPS

Setelah download selesai, ekstrak metadata ke CSV:

```bash
python extract_from_gallery_dl.py
```

**Output:** `fst_unja_from_gallery_dl.csv`

---

## ⚠️ SECURITY: JANGAN COMMIT cookies.txt!

File `cookies.txt` mengandung **session credentials** yang sangat sensitif!

**Tambahkan ke .gitignore:**
```
cookies.txt
*.txt
```

**Jika sudah ter-commit:**
```bash
git rm --cached cookies.txt
git commit -m "Remove sensitive cookies.txt"
git push --force  # HATI-HATI!
```

---

## 📚 REFERENSI

**Gallery-dl Documentation:**
- Filter syntax: https://github.com/mikf/gallery-dl#filter-and-range
- Instagram extractor: https://github.com/mikf/gallery-dl/blob/master/docs/configuration.rst#extractorinstagram

**Filter Examples:**
```bash
# By username
--filter "username == 'fst_unja'"

# By likes
--filter "likes >= 100"

# By date
--filter "date >= datetime(2024, 1, 1)"

# Combined
--filter "username == 'fst_unja' and likes >= 50"
```

**Instagram Cookies Explained:**
- `sessionid`: Session authentication token (WAJIB!)
- `csrftoken`: Cross-site request forgery protection
- `ds_user_id`: ID user yang sedang login

---

## 📝 CHANGELOG

**October 4, 2025:**
- ✅ Fixed 401 Unauthorized error dengan cookies.txt
- ✅ Added filter untuk menghindari download akun tagged
- ✅ Berhasil download 396 posts dari @fst_unja
- ✅ Dokumentasi lengkap dibuat

**Previously (Failed Attempts):**
- ❌ Username/password di config.json → 401 error
- ❌ `--cookies-from-browser chrome` → Permission denied
- ❌ `--cookies-from-browser edge` → Permission denied / DPAPI decrypt error

---

**Last Updated:** October 4, 2025 18:50 WIB  
**Status:** ✅ Verified Working
**Total Posts:** 396 (updated from previous 271)
