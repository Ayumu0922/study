import cv2
import numpy as np
import os
import glob


class detectMidpoints:
    def find_contour_midpoints(self, image):
        # ノイズを除去しないとノイズが輪郭線と検出されてアルゴリズムが機能しない
        gaussian_image = cv2.GaussianBlur(image, (15, 15), 3)

        edges = cv2.Canny(image=gaussian_image, threshold1=10, threshold2=10)

        # 膨張処理に使用するカーネルを定義
        kernel = np.ones((11, 11), np.uint8)

        # 膨張処理を適用
        dilated_edges = cv2.dilate(edges, kernel, iterations=3)
        eroded_edges = cv2.erode(dilated_edges, kernel, iterations=3)

        # 輪郭を検出
        contours, _ = cv2.findContours(
            eroded_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 輪郭の面積でソートして上位二つを取得
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        # 重心の中点を計算
        if len(sorted_contours) >= 2:
            moments1 = cv2.moments(sorted_contours[0])
            moments2 = cv2.moments(sorted_contours[1])
            centroid1 = (
                int(moments1["m10"] / moments1["m00"]),
                int(moments1["m01"] / moments1["m00"]),
            )
            centroid2 = (
                int(moments2["m10"] / moments2["m00"]),
                int(moments2["m01"] / moments2["m00"]),
            )
            midpoint = (
                (centroid1[0] + centroid2[0]) // 2,
                (centroid1[1] + centroid2[1]) // 2,
            )

            return midpoint, contours

        else:
            return None, None

    def process_images(self, input_dir, output_dir):
        for image_path in glob.glob(os.path.join(input_dir, "*.png")):
            print(image_path)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            midpoint, contours = self.find_contour_midpoints(image)

            if midpoint is not None:
                output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                cv2.circle(output_image, midpoint, 10, (0, 0, 255), -1)
                cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)

                basename = os.path.basename(image_path)
                output_path = os.path.join(output_dir, basename)
                cv2.imwrite(output_path, output_image)
            else:
                print(f"midpointがありません")


if __name__ == "__main__":
    input_dir = "detect_hallux_valgus/output/clahe_image"
    output_dir = "detect_hallux_valgus/output/midpoints"
    os.makedirs(output_dir, exist_ok=True)

    detect_midpoint = detectMidpoints()
    detect_midpoint.process_images(input_dir=input_dir, output_dir=output_dir)
