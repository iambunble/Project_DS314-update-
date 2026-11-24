RÀ SOÁT METADATA — bộ 5000 ảnh Kitchen (day1)
==============================================

ĐÚNG HẾT (không cần sửa): imageID, owner, day, hour, frame, tên file ảnh. 5000 dòng CSV khớp đúng 5000 ảnh.
CÓ LỖI: cột class_count (số người) và cột description (mô tả).

Mỗi ảnh trong các thư mục đã được vẽ sẵn:
  - khung người YOLO, có đánh số (xanh lá = YOLO chắc chắn, cam = độ tin cậy thấp 0.35–0.6)
  - dòng trên cùng: CSV = số người theo cột class_count, YOLO = số người YOLO đếm,
    MÔ TẢ = số người câu mô tả nói ("?" = mô tả không ghi số rõ ràng)
  - nhóm 03/04: dòng dưới cùng là câu mô tả có nhắc tới người
Ảnh gốc không bị đụng tới — đây chỉ là bản sao để xem.

CÁC NHÓM (thứ tự nên làm: 01 → 05 → 03 → 02 → 04)
---------------------------------------------------
01_anh_hieu_chuan_khong_phai_canh_bep (50)
   Ảnh có lưới ô / chữ "CASTLE 2024 Dataset", không phải cảnh bếp. Đề nghị loại khỏi bộ dữ liệu.
   Chỉ cần lướt xác nhận.

05_frame_vuot_3600__thuc_te_la_14h (11)
   Giờ = 13 nhưng frame 3608–3671 (> 3600 giây) -> thời điểm thật là 14:00:08–14:01:11.
   Hỏi "lúc 14h" sẽ không ra các ảnh này. day1_Kitchen_13_3635 trùng thời điểm với day1_Kitchen_14_35.

03_mo_ta_noi_co_nguoi__nhung_CSV_YOLO_deu_0 (547)
   Mô tả nói "a person is seated at the dining table..." nhưng cả CSV và YOLO đều không thấy ai.
   Đa số là mô tả BỊA (ảnh trống). Đôi khi có người thật ở sofa/phòng khách mà cả hai bỏ sót.
   -> Ghi số người thật vào so_nguoi_dung.

02_so_nguoi_lech__CSV_vs_YOLO (883)  — CẦN ĐẾM TAY
   a_YOLO_dem_IT_hon_CSV (219): YOLO hay sót người đang cúi ở bếp/bồn rửa phía xa, hoặc gộp 2 người sát nhau.
   b_YOLO_dem_NHIEU_hon_CSV (664): thường YOLO đúng (bắt được người nhỏ ở phòng khách),
                                  nhưng đôi khi khung cam (tin cậy thấp) là đếm trùng.
   Gợi ý = max(CSV, YOLO) — kiểm thử trên 9 ảnh thì đúng 8.

04_mo_ta_dem_sai_so_nguoi__CSV_YOLO_khop_nhau (908)
   CSV và YOLO cùng ra 1 số, mô tả nói khác (98% là mô tả đếm THIẾU, vd nói "two individuals" mà có 3–4 người).
   Gần như chắc mô tả sai -> có thể sửa tự động theo CSV/YOLO, không bắt buộc kiểm tay.

GÁN LẠI
-------
Mở review.csv (hoặc _danh_sach.csv trong từng thư mục) bằng Excel, điền:
   so_nguoi_dung      : số người thật trong ảnh (để trống = đồng ý với gợi ý)
   loai_khoi_du_lieu  : x nếu muốn bỏ ảnh
   ghi_chu            : tuỳ ý (vd "người ở sofa", "mô tả sai hành động")
Mẹo: ảnh xếp theo thời gian, khung hình liền nhau cách ~7 giây nên thường cùng số người — làm theo cụm sẽ nhanh.

Không có nhóm "màu áo sai": bộ nhận màu áo hiện tại chưa đủ chính xác để làm chuẩn so sánh.
_anh_mau_claude_da_xem/: các ảnh ghép Claude dùng để kiểm tra, có thể xoá.
