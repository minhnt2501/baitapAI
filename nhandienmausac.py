# Nhận diện màu sắc
import numpy as np
import cv2

# Lấy hình ảnh từ webcam
img = cv2.VideoCapture(0)
# Bắt đầu vòng lặp
while (1):
    # Nhận diện video từ các khung ảnh lấy từ webcam
    _, imgFrame = img.read()
    # chuyển từ BGR sang HSV
    hsvFrame = cv2.cvtColor(imgFrame, cv2.COLOR_BGR2HSV)
    # Đặt phạm vi của màu đỏ và xác định mặt nạ
    mauDo_lower = np.array([136, 87, 111], np.uint8)
    mauDo_upper = np.array([180, 255, 255], np.uint8)
    mauDo_mask = cv2.inRange(hsvFrame, mauDo_lower, mauDo_upper)
    # Đặt phạm vi của màu vàng và xác định mặt nạ
    vang_lower = np.array([22, 60, 200], np.uint8)
    vang_upper = np.array([60, 255, 255], np.uint8)
    vang_mask = cv2.inRange(hsvFrame, vang_lower, vang_upper)
    # Đặt phạm vi của màu xanh nước biển và xác định mặt nạ
    xanhNuocBien_lower = np.array([94, 80, 2], np.uint8)
    xanhNuocBien_upper = np.array([120, 255, 255], np.uint8)
    xanhNuocBien_mask = cv2.inRange(hsvFrame, xanhNuocBien_lower, xanhNuocBien_upper)
    # Đặt phạm vi của màu trắng và xác định mặt nạ
    trang_lower = np.array([0, 0, 200], np.uint8)
    trang_upper = np.array([180, 20, 255], np.uint8)
    trang_mask = cv2.inRange(hsvFrame, trang_lower, trang_upper)
    # Đặt phạm vi của màu đen và xác định mặt nạ
    den_lower = np.array([0, 0, 0], np.uint8)
    den_upper = np.array([180, 255, 30], np.uint8)
    den_mask = cv2.inRange(hsvFrame, den_lower, den_upper)
    # Biến đổi hình thái, giãn nở
    # cho mỗi màu và toán tử bitwise_and
    # giữa imgFrame và mặt nạ xác định
    # để chỉ phát hiện màu đó
    kernal = np.ones((5, 5), "uint8")

    mauDo_mask = cv2.dilate(mauDo_mask, kernal)
    res_mauDo = cv2.bitwise_and(imgFrame, imgFrame, mask=mauDo_mask)

    vang_mask = cv2.dilate(vang_mask, kernal)
    res_vang = cv2.bitwise_and(imgFrame, imgFrame, mask=vang_mask)

    xanhNuocBien_mask = cv2.dilate(xanhNuocBien_mask, kernal)
    res_xanhNuocBien = cv2.bitwise_and(imgFrame, imgFrame, mask=xanhNuocBien_mask)

    trang_mask = cv2.dilate(trang_mask, kernal)
    res_trang = cv2.bitwise_and(imgFrame, imgFrame, mask=trang_mask)

    den_mask = cv2.dilate(den_mask, kernal)
    res_den = cv2.bitwise_and(imgFrame, imgFrame, mask=den_mask)

    # Tạo viền để theo dõi các màu xác định
    # Màu đỏ
    (contours, hierarchy) = cv2.findContours(
        mauDo_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 400:
            x, y, w, h = cv2.boundingRect(contour)
            imgFrame = cv2.rectangle(imgFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(
                imgFrame, "Mau do : ", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255)
            )
    # Màu vàng
    (contours, hierarchy) = cv2.findContours(
        vang_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 400:
            x, y, w, h = cv2.boundingRect(contour)
            imgFrame = cv2.rectangle(imgFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                imgFrame,
                "Mau vang : ",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
            )
    # Màu xanh nước biển
    (contours, hierarchy) = cv2.findContours(
        xanhNuocBien_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 400:
            x, y, w, h = cv2.boundingRect(contour)
            imgFrame = cv2.rectangle(imgFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                imgFrame,
                "Mau xanh nuoc bien : ",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
            )
    # Màu trắng
    (contours, hierarchy) = cv2.findContours(
        trang_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 400:
            x, y, w, h = cv2.boundingRect(contour)
            imgFrame = cv2.rectangle(
                imgFrame, (x, y), (x + w, y + h), (255, 255, 255), 2
            )
            cv2.putText(
                imgFrame,
                "Mau trang : ",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
            )
    # Màu đen
    (contours, hierarchy) = cv2.findContours(
        den_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 400:
            x, y, w, h = cv2.boundingRect(contour)
            imgFrame = cv2.rectangle(imgFrame, (x, y), (x + w, y + h), (0, 0, 0), 2)
            cv2.putText(
                imgFrame, "Mau den : ", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0)
            )

    cv2.imshow("Phan mem nhan dien mau sac", imgFrame)
    if cv2.waitKey(10) & 0xFF == 27:
        cap.release()
        cv2.destroyAllWindows()
        break