import cv2
import numpy as np
import os
import glob
from detect_image_center import detectMidpoints


class MakeBone:
    def __init__(self, image):
        if len(image.shape) == 3 and image.shape[2] == 3:
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.image = image

    def find_lower_brightness_positions(self, start_position):
        x, y = start_position

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.clahe_image = clahe.apply(self.image)

        start_brightness = self.clahe_image[y, x]
        width = self.clahe_image.shape[1]

        # 左側に探索して、肉の輝度値を取得
        left_position = None
        for x_left in range(x, -1, -1):
            if self.clahe_image[y, x_left] <= start_brightness - 10:
                left_position = (x_left, y)
                break

        # 右側に探索して、肉の輝度値を取得
        right_position = None
        for x_right in range(x, width):
            if self.clahe_image[y, x_right] <= start_brightness - 10:
                right_position = (x_right, y)
                break

        left_lower_position = None
        if left_position:
            left_x, left_y = left_position
            left_brightness = self.clahe_image[left_y, left_x]
            for x_left_lower in range(left_x, -1, -1):
                if self.clahe_image[left_y, x_left_lower] <= left_brightness - 80:
                    left_lower_position = (x_left_lower, left_y)
                    break

        right_lower_position = None
        if right_position:
            right_x, right_y = right_position
            right_brightness = self.clahe_image[right_y, right_x]
            for x_right_lower in range(right_x, width):
                if self.clahe_image[right_y, x_right_lower] <= right_brightness - 80:
                    right_lower_position = (x_right_lower, right_y)
                    break

        right_lower_brightness = self.clahe_image[right_y, x_right_lower]
        left_lower_brightness = self.clahe_image[left_y, x_left_lower]

        return (
            left_position,
            right_position,
            left_lower_position,
            right_lower_position,
        ), (
            left_brightness,
            right_brightness,
            left_lower_brightness,
            right_lower_brightness,
        )

    def make_bone_binary(self, brightness, threshold=0.55):
        # 指定した輝度値よりも小さい輝度値を持つ部分を残して、それ以外を255にする
        brightness_threshold = brightness[0] * threshold
        bone = np.where(self.clahe_image <= brightness_threshold, self.clahe_image, 255)

        bone_canny = cv2.Canny(image=bone, threshold1=10, threshold2=10)

        # エッジ画像を膨張処理(これ以上kernelを上げると画像がつぶれる)
        dilate_bone_canny = cv2.dilate(
            bone_canny, kernel=np.ones((5, 5), np.uint8), iterations=3
        )

        # エッジ画像を膨張処理しても埋められなかった部分を補完する処理
        contours, _ = cv2.findContours(
            dilate_bone_canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )

        modified_dilate_bone_canny = dilate_bone_canny.copy()

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < 10000:
                # 輪郭の内側のピクセルを取得して、白い部分が多い場合は黒く埋める、黒い部分が多い場合は白く埋める
                mask = np.zeros_like(dilate_bone_canny)
                cv2.drawContours(mask, [contour], -1, color=255, thickness=cv2.FILLED)
                pts = np.where(mask == 255)

                white_pixels = np.sum(dilate_bone_canny[pts] == 255)
                black_pixels = np.sum(dilate_bone_canny[pts] == 0)

                if white_pixels > black_pixels:
                    cv2.drawContours(
                        modified_dilate_bone_canny,
                        [contour],
                        -1,
                        color=0,
                        thickness=cv2.FILLED,
                    )
                else:
                    cv2.drawContours(
                        modified_dilate_bone_canny,
                        [contour],
                        -1,
                        color=255,
                        thickness=cv2.FILLED,
                    )

        # それでも修正できなかった部分を埋める処理
        bone_binary = cv2.morphologyEx(
            modified_dilate_bone_canny, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )

        return bone_binary, bone


if __name__ == "__main__":
    input_dir = "detect_hallux_valgus/src/preprocessed"
    output_dir = "detect_hallux_valgus/output/bone_binary"
    os.makedirs(output_dir, exist_ok=True)

    for image_path in glob.glob(os.path.join(input_dir, "*.png")):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        detect_midpoints = detectMidpoints()

        # 画像の中心を検出
        midpoints, contour = detect_midpoints.find_contour_midpoints(image=image)

        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        make_bone = MakeBone(image=image)

        positions, brightness = make_bone.find_lower_brightness_positions(
            start_position=midpoints
        )

        # 輝度値の取得に使用した点を描画
        for position in positions:
            if position is not None:
                cv2.circle(
                    color_image,
                    position,
                    radius=10,
                    color=(0, 0, 255),
                    thickness=-1,
                )

        bone_binary, bone = make_bone.make_bone_binary(brightness=brightness)

        # 処理結果の保存
        basename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, basename)
        cv2.imwrite(output_path, bone_binary)
