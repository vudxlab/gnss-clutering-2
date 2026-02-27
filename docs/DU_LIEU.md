# Bo du lieu GNSS

## Tong quan

Bo du lieu gom **18 file CSV** ghi nhan toa do dich chuyen tu cac tram GNSS (Global Navigation Satellite System) trong khoang **13 ngay**, tu **29/05/2015** den **11/06/2015**, voi tan so lay mau **1 Hz** (1 mau/giay).

Cac tram duoc chia thanh 2 nhom theo vi tri:

- **Nhom East (E):** 6 tram – `1e`, `2e`, `3e`, `4e`, `5e`, `6e`
- **Nhom West (W):** 11 tram – `1w` ... `6w`, `101w` ... `105w`

---

## Cau truc file CSV

Moi file co 4 cot:

| Cot | Kieu | Mo ta |
|-----|------|-------|
| `Timestamp` | datetime | Thoi diem ghi nhan (yyyy-mm-dd HH:MM:SS), moi giay 1 dong |
| `X_Coord` | float | Toa do ngang X (m) |
| `Y_Coord` | float | Toa do ngang Y (m) |
| `h_Coord` | float | **Toa do thang dung** – do cao (m). Day la kenh chinh dung de phan tich |

Cac dong bi thieu du lieu (mat tin hieu, mat dien...) duoc danh dau bang `NaN` tai cac cot `X_Coord`, `Y_Coord`, `h_Coord`.

---

## Danh sach tram

### Nhom chan (h ~ 13–16 m)

| File | So dong | Thoi gian | X (m) | Y (m) | h trung binh (m) |
|------|---------|-----------|-------|-------|-------------------|
| `full_gnss_1e.csv` | 1,130,625 | 02:38:21 29/05 → 04:42:05 11/06 | ~85,059 | ~2,332,159 | ~13.8 |
| `full_gnss_1w.csv` | 1,130,718 | 02:36:48 29/05 → 04:42:05 11/06 | ~85,028 | ~2,332,173 | ~13.8 |
| `full_gnss_2e.csv` | 1,130,464 | 02:41:02 29/05 → 04:42:05 11/06 | ~85,147 | ~2,332,366 | ~14.8 |
| `full_gnss_2w.csv` | 1,130,568 | 02:39:25 29/05 → 04:42:12 11/06 | ~85,116 | ~2,332,379 | ~14.8 |
| `full_gnss_3e.csv` | 1,130,323 | 02:44:18 29/05 → 04:43:00 11/06 | ~85,263 | ~2,332,640 | ~16.1 |
| `full_gnss_3w.csv` | 1,130,468 | 02:41:53 29/05 → 04:43:00 11/06 | ~85,232 | ~2,332,653 | ~16.1 |
| `full_gnss_4e.csv` | 1,130,160 | 02:47:01 29/05 → 04:43:00 11/06 | ~85,381 | ~2,332,918 | ~16.1 |
| `full_gnss_4w.csv` | 1,130,256 | 02:45:25 29/05 → 04:43:00 11/06 | ~85,350 | ~2,332,931 | ~16.1 |
| `full_gnss_5e.csv` | 1,129,070 | 03:05:10 29/05 → 04:42:59 11/06 | ~85,500 | ~2,333,196 | ~14.8 |
| `full_gnss_5w.csv` | 1,129,127 | 03:04:14 29/05 → 04:43:00 11/06 | ~85,468 | ~2,333,209 | ~14.8 |
| `full_gnss_6e.csv` | 1,128,962 | 03:07:53 29/05 → 04:43:54 11/06 | ~85,587 | ~2,333,402 | ~13.8 |
| `full_gnss_6w.csv` | 1,129,001 | 03:07:14 29/05 → 04:43:54 11/06 | ~85,556 | ~2,333,415 | ~13.8 |

### Nhom cao (h ~ 84–87 m)

| File | So dong | Thoi gian | X (m) | Y (m) | h trung binh (m) |
|------|---------|-----------|-------|-------|-------------------|
| `full_gnss_101w.csv` | 1,130,873 | 02:33:17 29/05 → 04:41:09 11/06 | ~85,069 | ~2,332,236 | ~84.6 |
| `full_gnss_102w.csv` | 1,130,910 | 02:32:40 29/05 → 04:41:09 11/06 | ~85,187 | ~2,332,512 | ~86.2 |
| `full_gnss_103w.csv` | 1,130,818 | 02:34:12 29/05 → 04:41:09 11/06 | ~85,304 | ~2,332,788 | ~87.0 |
| `full_gnss_104w.csv` | 1,130,816 | 02:35:09 29/05 → 04:42:04 11/06 | ~85,421 | ~2,333,064 | ~86.1 |
| `full_gnss_105w.csv` | 1,130,768 | 02:35:57 29/05 → 04:42:04 11/06 | ~85,539 | ~2,333,340 | ~84.6 |

### File trung lap

| File | Ghi chu |
|------|---------|
| `full_gnss_e.csv` | Trung voi `full_gnss_3e.csv` (cung so dong, cung toa do) |

---

## Thong ke du lieu

| Thong so | Gia tri |
|----------|---------|
| Tong so tram | 17 (+ 1 file trung lap) |
| Khoang thoi gian | ~13 ngay (29/05 – 11/06/2015) |
| Tan so lay mau | 1 Hz (1 mau/giay) |
| So dong moi file | ~1,129,000 – 1,131,000 |
| Tong so dong toan bo | ~20.3 trieu |
| Ty le du lieu co | ~79–83% (phan con lai la NaN) |

---

## Nguon du lieu thieu

Du lieu thieu (NaN) xuat hien chu yeu tai:

1. **Ngay dau** (29/05/2015): cac tram bat dau ghi tu 02:30–03:08, khong co du lieu truoc do
2. **Ngay cuoi** (11/06/2015): cac tram ngung ghi luc 04:41–04:43, chi co ~5 gio du lieu
3. **Ngay 01/06/2015**: mat hoan toan du lieu (tat ca cac tram)
4. **Cac khoang ngan trong ngay**: mat tin hieu GNSS do nhieu, che khuat, hoac loi thiet bi
5. **`full_gnss_2w.csv`**: thieu nhieu hon cac file khac (~305K dong NaN so voi ~233K cua cac tram con lai)

---

## Bo tri khong gian cac tram

Dua tren toa do X, Y, cac tram duoc bo tri **theo tuyen thang** (Y tang dan ~276m giua cac cap), voi moi vi tri co 2 tram East va West cach nhau ~30m theo phuong X:

```
            X (m)
            85,000                85,600
              |                     |
 Y=2,333,400  ·  ·················· 6w  6e          h ~ 13.8m
              |                    |
 Y=2,333,200  ·  ·················· 5w  5e  105w    h ~ 14.8m / 84.6m
              |                    |
 Y=2,332,900  ·  ·················· 4w  4e  104w    h ~ 16.1m / 86.1m
              |                    |
 Y=2,332,650  ·  ·················· 3w  3e  103w    h ~ 16.1m / 87.0m
              |                    |
 Y=2,332,370  ·  ·················· 2w  2e  102w    h ~ 14.8m / 86.2m
              |                    |
 Y=2,332,160  ·  ·················· 1w  1e  101w    h ~ 13.8m / 84.6m
```

Nhom `101w`–`105w` co cung vi tri mat bang (X, Y) voi cac tram `1`–`5` nhung do cao lon hon nhieu (~84–87m so voi ~13–16m), cho thay chung duoc dat o **do cao khac nhau** tren cung cot/tru.

---

## Su dung trong project

Project hien dang phan tich tram `2e` (`full_gnss_2e.csv`) lam vi du. De doi sang tram khac, sua `DATA_PATH` trong `gnss_clustering/config.py`:

```python
DATA_PATH = os.path.join(DATA_DIR, 'full_gnss_3e.csv')   # doi sang tram 3e
```

Sau do chay lai voi `--no-cache` de tai du lieu moi:

```bash
python step1_find_k.py --no-cache
python step2_cluster.py --k1 4 --k2 2 --no-cache
```
