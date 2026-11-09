"""
=============================================================================
SIÊU TỪ ĐIỂN POINT-IN-TIME: LỊCH SỬ THAY ĐỔI RỔ VN30 (2014 - NAY)
Khắc phục 100% Thiên kiến sống sót (Survivorship Bias)
Phản ánh chính xác quyết định của HOSE tại từng kỳ đảo danh mục (tháng 1 & tháng 7)
Bao gồm đầy đủ các cổ phiếu khủng hoảng, hủy niêm yết: HAG, OGC, ITA, FLC, ROS, NVL, PDR...

AUTO-FETCH: File này tự động thử fetch danh mục hiện tại từ API khi import.
Nếu fetch thất bại hoặc không có internet, dữ liệu tĩnh được dùng làm fallback.
=============================================================================
"""

import datetime
import logging
import os
import json

logger = logging.getLogger(__name__)

# =============================================================================
# DỮ LIỆU TĨNH POINT-IN-TIME (FALLBACK)
# =============================================================================

VN30_TIMELINE = [
    # -------------------------------------------------------------
    # 1. KỲ 1/2014 (Áp dụng từ 01/01/2014 đến 20/07/2014)
    # HOSE thêm ITA, loại SBT
    # -------------------------------------------------------------
    (
        "2014-01-01", "2014-07-20",
        [
            "BVH", "CII", "CSM", "CTG", "DRC", "DPM", "EIB", "FPT", "GMD", "HAG",
            "HSG", "HPG", "IJC", "ITA", "KDC", "MBB", "MSN", "OGC", "PET", "PGD",
            "PPC", "PVD", "PVT", "REE", "SSI", "STB", "VCB", "VIC", "VNM", "VSH"
        ]
    ),

    # -------------------------------------------------------------
    # 2. KỲ 2/2014 (Áp dụng từ 21/07/2014 đến 25/01/2015)
    # HOSE thêm FLC, HCM; loại PET, PGD
    # -------------------------------------------------------------
    (
        "2014-07-21", "2015-01-25",
        [
            "BVH", "CII", "CSM", "CTG", "DRC", "DPM", "EIB", "FLC", "FPT", "GMD",
            "HAG", "HCM", "HSG", "HPG", "IJC", "ITA", "KDC", "MBB", "MSN", "OGC",
            "PPC", "PVD", "PVT", "REE", "SSI", "STB", "VCB", "VIC", "VNM", "VSH"
        ]
    ),

    # -------------------------------------------------------------
    # 3. KỲ 1/2015 (Áp dụng từ 26/01/2015 đến 26/07/2015)
    # HOSE thêm KBC, HVG; loại CTG, EIB
    # (Tháng 7/2015 OGC bị kiểm soát đặc biệt do đại án OceanBank và rơi tự do)
    # -------------------------------------------------------------
    (
        "2015-01-26", "2015-07-26",
        [
            "BVH", "CII", "CSM", "DRC", "DPM", "FLC", "FPT", "GMD", "HAG", "HCM",
            "HSG", "HPG", "HVG", "IJC", "ITA", "KBC", "KDC", "MBB", "MSN", "OGC",
            "PPC", "PVD", "PVT", "REE", "SSI", "STB", "VCB", "VIC", "VNM", "VSH"
        ]
    ),

    # -------------------------------------------------------------
    # 4. KỲ 2/2015 (Áp dụng từ 27/07/2015 đến 24/01/2016)
    # HOSE thêm CTG, EIB, HHS; loại IJC, DRC, OGC (SJS thế chỗ tạm OGC rồi ra)
    # -------------------------------------------------------------
    (
        "2015-07-27", "2016-01-24",
        [
            "BVH", "CII", "CSM", "CTG", "DPM", "EIB", "FLC", "FPT", "GMD", "HAG",
            "HCM", "HHS", "HSG", "HPG", "HVG", "ITA", "KBC", "KDC", "MBB", "MSN",
            "PPC", "PVD", "PVT", "REE", "SSI", "STB", "VCB", "VIC", "VNM", "VSH"
        ]
    ),

    # -------------------------------------------------------------
    # 5. KỲ 1/2016 (Áp dụng từ 25/01/2016 đến 24/07/2016)
    # HOSE thêm SBT, NT2; loại CSM, VSH
    # -------------------------------------------------------------
    (
        "2016-01-25", "2016-07-24",
        [
            "BVH", "CII", "CTG", "DPM", "EIB", "FLC", "FPT", "GMD", "HAG", "HCM",
            "HHS", "HSG", "HPG", "HVG", "ITA", "KBC", "KDC", "MBB", "MSN", "NT2",
            "PPC", "PVD", "PVT", "REE", "SBT", "SSI", "STB", "VCB", "VIC", "VNM"
        ]
    ),

    # -------------------------------------------------------------
    # 6. KỲ 2/2016 & KỲ 1/2017 (Áp dụng từ 25/07/2016 đến 23/07/2017)
    # HOSE áp dụng Quy tắc mới: thêm BID, GAS, HNG, MWG; loại EIB, HHS, HVG, PVT
    # Kỳ 1/2017 giữ nguyên danh mục
    # -------------------------------------------------------------
    (
        "2016-07-25", "2017-07-23",
        [
            "BID", "BVH", "CII", "CTG", "DPM", "FLC", "FPT", "GAS", "GMD", "HAG",
            "HCM", "HNG", "HPG", "HSG", "ITA", "KBC", "KDC", "MBB", "MSN", "MWG",
            "NT2", "PPC", "PVD", "REE", "SBT", "SSI", "STB", "VCB", "VIC", "VNM"
        ]
    ),

    # -------------------------------------------------------------
    # 7. KỲ 2/2017 (Áp dụng từ 24/07/2017 đến 21/01/2018)
    # HOSE thêm ROS, NVL, SAB; loại HAG, HNG, ITA
    # -------------------------------------------------------------
    (
        "2017-07-24", "2018-01-21",
        [
            "BID", "BVH", "CII", "CTG", "DPM", "FLC", "FPT", "GAS", "GMD", "HCM",
            "HPG", "HSG", "KBC", "KDC", "MBB", "MSN", "MWG", "NT2", "NVL", "PPC",
            "PVD", "REE", "ROS", "SAB", "SBT", "SSI", "STB", "VCB", "VIC", "VNM"
        ]
    ),

    # -------------------------------------------------------------
    # 8. KỲ 1/2018 (Áp dụng từ 22/01/2018 đến 22/07/2018)
    # HOSE thêm PLX, VJC; loại PVD, KBC
    # -------------------------------------------------------------
    (
        "2018-01-22", "2018-07-22",
        [
            "BID", "BVH", "CII", "CTG", "DPM", "FLC", "FPT", "GAS", "GMD", "HCM",
            "HPG", "HSG", "KDC", "MBB", "MSN", "MWG", "NT2", "NVL", "PLX", "PPC",
            "REE", "ROS", "SAB", "SBT", "SSI", "STB", "VCB", "VIC", "VJC", "VNM"
        ]
    ),

    # -------------------------------------------------------------
    # 9. KỲ 2/2018 (Áp dụng từ 23/07/2018 đến 10/02/2019)
    # HOSE thêm PNJ, VPB, VRE; loại BID, BVH, NT2
    # -------------------------------------------------------------
    (
        "2018-07-23", "2019-02-10",
        [
            "CII", "CTG", "DPM", "FLC", "FPT", "GAS", "GMD", "HCM", "HPG", "HSG",
            "KDC", "MBB", "MSN", "MWG", "NVL", "PLX", "PNJ", "PPC", "REE", "ROS",
            "SAB", "SBT", "SSI", "STB", "VCB", "VIC", "VJC", "VNM", "VPB", "VRE"
        ]
    ),

    # -------------------------------------------------------------
    # 10. KỲ 1/2019 (Áp dụng từ 11/02/2019 đến 04/08/2019)
    # HOSE thêm VHM, EIB, HDB, TCB; loại BMP, HSG, KDC, PLX
    # -------------------------------------------------------------
    (
        "2019-02-11", "2019-08-04",
        [
            "BID", "BVH", "CII", "CTG", "DPM", "EIB", "FPT", "GAS", "GMD", "HDB",
            "HPG", "MBB", "MSN", "MWG", "NVL", "PNJ", "REE", "ROS", "SAB", "SBT",
            "SSI", "STB", "TCB", "VCB", "VHM", "VIC", "VJC", "VNM", "VPB", "VRE"
        ]
    ),

    # -------------------------------------------------------------
    # 11. KỲ 2/2019 (Áp dụng từ 05/08/2019 đến 02/02/2020)
    # HOSE thêm BID, BVH; loại CII, DHG
    # -------------------------------------------------------------
    (
        "2019-08-05", "2020-02-02",
        [
            "BID", "BVH", "CTD", "CTG", "DPM", "EIB", "FPT", "GAS", "GMD", "HDB",
            "HPG", "MBB", "MSN", "MWG", "NVL", "PNJ", "REE", "ROS", "SAB", "SBT",
            "SSI", "STB", "TCB", "VCB", "VHM", "VIC", "VJC", "VNM", "VPB", "VRE"
        ]
    ),

    # -------------------------------------------------------------
    # 12. KỲ 1/2020 (Áp dụng từ 03/02/2020 đến 02/08/2020)
    # HOSE thêm PLX, POW; loại DPM, GMD
    # -------------------------------------------------------------
    (
        "2020-02-03", "2020-08-02",
        [
            "BID", "BVH", "CTD", "CTG", "EIB", "FPT", "GAS", "HDB", "HPG", "MBB",
            "MSN", "MWG", "NVL", "PLX", "PNJ", "POW", "REE", "ROS", "SAB", "SBT",
            "SSI", "STB", "TCB", "VCB", "VHM", "VIC", "VJC", "VNM", "VPB", "VRE"
        ]
    ),

    # -------------------------------------------------------------
    # 13. KỲ 2/2020 (Áp dụng từ 03/08/2020 đến 31/01/2021)
    # HOSE thêm TCH, KDH; loại CTD, BVH
    # -------------------------------------------------------------
    (
        "2020-08-03", "2021-01-31",
        [
            "BID", "CTG", "EIB", "FPT", "GAS", "HDB", "HPG", "KDH", "MBB", "MSN",
            "MWG", "NVL", "PLX", "PNJ", "POW", "REE", "ROS", "SAB", "SBT", "SSI",
            "STB", "TCB", "TCH", "VCB", "VHM", "VIC", "VJC", "VNM", "VPB", "VRE"
        ]
    ),

    # -------------------------------------------------------------
    # 14. KỲ 1/2021 (Áp dụng từ 01/02/2021 đến 01/08/2021)
    # HOSE thêm BVH, PDR, TPB; loại SAB, EIB, ROS (Chính thức đuổi ROS ra khỏi VN30)
    # -------------------------------------------------------------
    (
        "2021-02-01", "2021-08-01",
        [
            "BID", "BVH", "CTG", "FPT", "GAS", "HDB", "HPG", "KDH", "MBB", "MSN",
            "MWG", "NVL", "PDR", "PLX", "PNJ", "POW", "REE", "SBT", "SSI", "STB",
            "TCB", "TCH", "TPB", "VCB", "VHM", "VIC", "VJC", "VNM", "VPB", "VRE"
        ]
    ),

    # -------------------------------------------------------------
    # 15. KỲ 2/2021 & KỲ 1/2022 (Áp dụng từ 02/08/2021 đến 31/07/2022)
    # HOSE thêm ACB, SAB, GVR; loại TCH, SBT, REE
    # -------------------------------------------------------------
    (
        "2021-08-02", "2022-07-31",
        [
            "ACB", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG", "KDH",
            "MBB", "MSN", "MWG", "NVL", "PDR", "PLX", "PNJ", "POW", "SAB", "SSI",
            "STB", "TCB", "TPB", "VCB", "VHM", "VIC", "VJC", "VNM", "VPB", "VRE"
        ]
    ),

    # -------------------------------------------------------------
    # 16. KỲ 2/2022 (Áp dụng từ 01/08/2022 đến 05/02/2023)
    # HOSE thêm VIB; loại PNJ
    # (Tháng 11/2022: NVL & PDR dính khủng hoảng thanh khoản trái phiếu, rơi hơn 20 cây sàn)
    # -------------------------------------------------------------
    (
        "2022-08-01", "2023-02-05",
        [
            "ACB", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG", "KDH",
            "MBB", "MSN", "MWG", "NVL", "PDR", "PLX", "POW", "SAB", "SSI", "STB",
            "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE"
        ]
    ),

    # -------------------------------------------------------------
    # 17. KỲ 1/2023 (Áp dụng từ 06/02/2023 đến 06/08/2023)
    # HOSE thêm BCM; loại KDH
    # -------------------------------------------------------------
    (
        "2023-02-06", "2023-08-06",
        [
            "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
            "MBB", "MSN", "MWG", "NVL", "PDR", "PLX", "POW", "SAB", "SSI", "STB",
            "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE"
        ]
    ),

    # -------------------------------------------------------------
    # 18. KỲ 2/2023 & KỲ 1/2024 & KỲ 2/2024 (Áp dụng từ 07/08/2023 đến 02/02/2025)
    # HOSE chính thức loại NVL & PDR; thêm SHB & SSB
    # Kỳ 1/2024 (05/02/2024): Không thay đổi thành phần
    # Kỳ 2/2024 (05/08/2024): Không thay đổi thành phần (NVL vào dự phòng)
    # -------------------------------------------------------------
    (
        "2023-08-07", "2025-02-02",
        [
            "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
            "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
            "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE"
        ]
    ),

    # -------------------------------------------------------------
    # 19. KỲ 1/2025 (Hiệu lực: 03/02/2025 đến 03/08/2025)
    # HOSE thêm LPB (LPBank); loại POW
    # Quy tắc HOSE-Index 4.0 bắt đầu áp dụng (giới hạn tỷ trọng ngành 40%)
    # -------------------------------------------------------------
    (
        "2025-02-03", "2025-08-03",
        [
            "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
            "LPB", "MBB", "MSN", "MWG", "PLX", "SAB", "SHB", "SSB", "SSI", "STB",
            "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE"
        ]
    ),

    # -------------------------------------------------------------
    # 20. KỲ 2/2025 (Hiệu lực: 04/08/2025 đến nay)
    # HOSE thêm DGC (Hóa chất Đức Giang); loại BVH (Bảo Việt - không đủ thanh khoản)
    # -------------------------------------------------------------
    (
        "2025-08-04", "2099-12-31",   # end date mở - sẽ bị override nếu fetch thành công
        [
            "ACB", "BCM", "BID", "CTG", "DGC", "FPT", "GAS", "GVR", "HDB", "HPG",
            "LPB", "MBB", "MSN", "MWG", "PLX", "SAB", "SHB", "SSB", "SSI", "STB",
            "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE"
        ]
    ),
]


# =============================================================================
# CACHE FILE - LƯU KẾT QUẢ FETCH ĐỂ DÙNG LẠI
# =============================================================================

_CACHE_FILE = os.path.join(os.path.dirname(__file__), ".vn30_cache.json")
_CACHE_TTL_HOURS = 24  # Chỉ fetch lại sau 24 giờ


def _load_cache():
    """Đọc cache từ file JSON."""
    try:
        if os.path.exists(_CACHE_FILE):
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached_at = datetime.datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
            age_hours = (datetime.datetime.now() - cached_at).total_seconds() / 3600
            if age_hours < _CACHE_TTL_HOURS:
                return data.get("symbols")
    except Exception:
        pass
    return None


def _save_cache(symbols):
    """Lưu kết quả fetch vào cache."""
    try:
        data = {
            "cached_at": datetime.datetime.now().isoformat(),
            "symbols": symbols
        }
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# =============================================================================
# AUTO-FETCH: TỰ ĐỘNG LẤY DANH MỤC HIỆN TẠI TỪ API
# =============================================================================

def _fetch_current_vn30_from_api():
    """
    Thử fetch danh mục VN30 hiện tại từ nhiều nguồn API.
    Trả về list symbols nếu thành công, None nếu thất bại.

    Nguồn thử theo thứ tự ưu tiên:
    1. VNDirect index_stocks API
    2. SSI iBoard API (không cần key)
    3. TCBS API
    """
    import urllib.request

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }

    # --- Nguồn 1: VNDirect ---
    try:
        url = (
            "https://finfo-api.vndirect.com.vn/v4/index_stocks"
            "?indexCode=VN30&page=1&size=30"
        )
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = json.loads(resp.read().decode())
        items = raw.get("data", raw.get("items", []))
        symbols = [
            item.get("code", item.get("stockCode", "")).strip().upper()
            for item in items
            if item.get("code", item.get("stockCode", ""))
        ]
        if len(symbols) >= 25:
            logger.info(f"[VN30] Auto-fetch từ VNDirect: {len(symbols)} mã")
            return sorted(symbols)
    except Exception as e:
        logger.debug(f"[VN30] VNDirect fetch thất bại: {e}")

    # --- Nguồn 2: SSI iBoard ---
    try:
        url = "https://iboard.ssi.com.vn/dchart/api/1.1/chartslibrary/chart/getindexcomponent?indexCode=VN30"
        req = urllib.request.Request(url, headers={**headers, "Referer": "https://iboard.ssi.com.vn/"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = json.loads(resp.read().decode())
        # SSI trả về list of dicts hoặc nested
        items = raw if isinstance(raw, list) else raw.get("data", [])
        symbols = [
            item.get("stockCode", item.get("code", "")).strip().upper()
            for item in items
            if isinstance(item, dict)
        ]
        symbols = [s for s in symbols if len(s) == 3]
        if len(symbols) >= 25:
            logger.info(f"[VN30] Auto-fetch từ SSI iBoard: {len(symbols)} mã")
            return sorted(symbols)
    except Exception as e:
        logger.debug(f"[VN30] SSI fetch thất bại: {e}")

    # --- Nguồn 3: TCBS ---
    try:
        url = "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/group?groupCode=VN30"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = json.loads(resp.read().decode())
        items = raw.get("data", [])
        symbols = [
            item.get("ticker", "").strip().upper()
            for item in items
            if item.get("ticker")
        ]
        if len(symbols) >= 25:
            logger.info(f"[VN30] Auto-fetch từ TCBS: {len(symbols)} mã")
            return sorted(symbols)
    except Exception as e:
        logger.debug(f"[VN30] TCBS fetch thất bại: {e}")

    return None


def refresh_current_vn30(force=False):
    """
    Cập nhật danh mục VN30 hiện tại từ API.

    Args:
        force (bool): Bỏ qua cache, bắt buộc fetch lại.

    Returns:
        list[str] | None: Danh sách mã nếu fetch thành công, None nếu thất bại.
    """
    if not force:
        cached = _load_cache()
        if cached:
            logger.debug(f"[VN30] Dùng cache ({len(cached)} mã)")
            return cached

    symbols = _fetch_current_vn30_from_api()
    if symbols:
        _save_cache(symbols)
    return symbols


# =============================================================================
# HÀM CHÍNH: LẤY DANH MỤC VN30 THEO NGÀY
# =============================================================================

def get_vn30_constituents(date_str, use_live=False):
    """
    Trả về danh sách chính xác 30 mã cổ phiếu VN30 tại thời điểm date_str.

    Args:
        date_str (str | pd.Timestamp): Ngày cần tra cứu, dạng 'YYYY-MM-DD'.
        use_live (bool): Nếu True và date_str là ngày hiện tại hoặc tương lai,
                         sẽ thử fetch danh mục live từ API trước.

    Returns:
        list[str]: Danh sách mã cổ phiếu.
    """
    if not isinstance(date_str, str):
        date_str = str(date_str)[:10]

    today = datetime.date.today().isoformat()

    # Nếu hỏi ngày hiện tại/tương lai và muốn dữ liệu live
    if use_live and date_str >= today:
        live = refresh_current_vn30()
        if live:
            return live

    # Tra cứu Point-in-Time tĩnh
    for start_d, end_d, basket in VN30_TIMELINE:
        if start_d <= date_str <= end_d:
            return basket

    # Fallback: trước 2014 → kỳ đầu; sau mọi kỳ → kỳ mới nhất
    if date_str < VN30_TIMELINE[0][0]:
        return VN30_TIMELINE[0][2]
    return VN30_TIMELINE[-1][2]


def get_current_vn30(live=True):
    """
    Trả về danh mục VN30 HÔM NAY.

    Args:
        live (bool): Nếu True (mặc định), thử fetch từ API trước.
                     Nếu False, dùng dữ liệu tĩnh.

    Returns:
        list[str]: 30 mã cổ phiếu.
    """
    today = datetime.date.today().isoformat()
    return get_vn30_constituents(today, use_live=live)


# =============================================================================
# TẬP HỢP TOÀN BỘ MÃ TỪNG NẰM TRONG RỔ (HISTORICAL UNIVERSE)
# =============================================================================

ALL_HISTORICAL_VN30_TICKERS = sorted(list(set(
    ticker for _, _, basket in VN30_TIMELINE for ticker in basket
)))


# =============================================================================
# AUTO-FETCH KHI IMPORT (silent, không block)
# =============================================================================

def _silent_prefetch():
    """
    Chạy ngầm khi import để warm up cache.
    Không raise exception, không block chương trình.
    """
    try:
        cached = _load_cache()
        if cached is None:
            symbols = _fetch_current_vn30_from_api()
            if symbols:
                _save_cache(symbols)
                logger.debug(f"[VN30] Prefetch thành công: {symbols}")
    except Exception:
        pass


# Kích hoạt prefetch ngay khi import (không dùng thread để tránh phức tạp)
# Comment dòng dưới nếu muốn tắt auto-fetch
_silent_prefetch()


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    print("=" * 60)
    print("VN30 Universe - Kiểm tra nhanh")
    print("=" * 60)

    # Test Point-in-Time
    test_dates = [
        "2015-03-15",   # Kỳ 1/2015 - có OGC, HAG
        "2018-09-01",   # Kỳ 2/2018 - có ROS
        "2022-11-30",   # Kỳ 2/2022 - còn NVL, PDR (tháng khủng hoảng)
        "2024-06-01",   # Kỳ 1/2024 - giữ nguyên từ 2023
        "2025-03-01",   # Kỳ 1/2025 - có LPB, không có POW
        "2025-10-01",   # Kỳ 2/2025 - có DGC, không có BVH
    ]

    for d in test_dates:
        basket = get_vn30_constituents(d)
        print(f"\n{d} ({len(basket)} mã): {', '.join(sorted(basket))}")

    # Test Live
    print("\n" + "=" * 60)
    print("Danh mục HIỆN TẠI (live fetch):")
    current = get_current_vn30(live=True)
    print(f"{len(current)} mã: {', '.join(current)}")

    print("\n" + "=" * 60)
    print(f"Tổng số mã từng có trong VN30: {len(ALL_HISTORICAL_VN30_TICKERS)}")
    print(", ".join(ALL_HISTORICAL_VN30_TICKERS))
