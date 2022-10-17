cv2.imshow("original", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
cv2.imshow("gray", image_gray)
cv2.imshow("noise reduce", cv2.cvtColor(image_less_noise, cv2.COLOR_RGB2BGR))
cv2.imshow("edge detect", image_detect_edge)