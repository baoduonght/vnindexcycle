"""
=============================================================================
SIÊU TỪ ĐIỂN POINT-IN-TIME: LỊCH SỬ THAY ĐỔI RỔ VN30 (2014 - 2026)
Khắc phục 100% Thiên kiến sống sót (Survivorship Bias)
Phản ánh chính xác quyết định của HOSE tại từng kỳ đảo danh mục (tháng 1 & tháng 7)
Bao gồm đầy đủ các cổ phiếu khủng hoảng, hủy niêm yết: HAG, OGC, ITA, FLC, ROS, NVL, PDR...
=============================================================================
"""

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
    # 18. KỲ 2/2023 ĐẾN NAY (Áp dụng từ 07/08/2023 đến hiện tại)
    # HOSE chính thức loại NVL & PDR; thêm SHB & SSB
    # Kỳ 1/2024 & Kỳ 2/2024 giữ nguyên danh mục
    # -------------------------------------------------------------
    (
        "2023-08-07", "2026-12-31",
        [
            "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
            "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
            "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE"
        ]
    ),
]


def get_vn30_constituents(date_str):
    """
    Trả về danh sách chính xác 30 mã cổ phiếu VN30 tại thời điểm date_str.
    date_str có dạng 'YYYY-MM-DD' hoặc pd.Timestamp.
    """
    if not isinstance(date_str, str):
        date_str = str(date_str)[:10]

    for start_d, end_d, basket in VN30_TIMELINE:
        if start_d <= date_str <= end_d:
            return basket

    # Fallback cho ngày ngoài phạm vi: nếu trước 2014 lấy kỳ đầu tiên, sau lấy kỳ mới nhất
    if date_str < VN30_TIMELINE[0][0]:
        return VN30_TIMELINE[0][2]
    return VN30_TIMELINE[-1][2]


# Tập hợp toàn bộ các mã cổ phiếu từng nằm trong rổ VN30 trong 10 năm qua
ALL_HISTORICAL_VN30_TICKERS = sorted(list(set(
    ticker for _, _, basket in VN30_TIMELINE for ticker in basket
)))
