import cv2
import glob
import os
import numpy as np
import re
from detect_image_center import detectMidpoints
from make_bone_binary import MakeBone


class GetPromptPosition:

    def __init__(self, image):
        if len(image.shape) == 3 and image.shape[2] == 3:
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.image = image
        self.detect_mid_points = detectMidpoints()
        self.make_bone = MakeBone(image=image)

    def get_heel_positions(self):

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.clahe_image = clahe.apply(self.image)

        # 輝度値が50以下の部分を抽出
        _, binary_image = cv2.threshold(
            self.clahe_image, 50, 255, cv2.THRESH_BINARY_INV
        )

        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        areas_and_centroids = [
            (cv2.contourArea(contour), cv2.moments(contour)) for contour in contours
        ]

        areas_and_centroids.sort(key=lambda x: x[0], reverse=True)

        centroids = []
        for i in range(min(2, len(areas_and_centroids))):
            M = areas_and_centroids[i][1]
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))

        centroids.sort(key=lambda x: x[0])

        left_heel, right_heel = centroids[0], centroids[1]

        return left_heel, right_heel

    def get_tip_position(self):
        self.midpoints, contours = self.detect_mid_points.find_contour_midpoints(
            image=self.clahe_image
        )

        positions, brightness = self.make_bone.find_lower_brightness_positions(
            start_position=self.midpoints
        )
        self.bone_binary, bone = self.make_bone.make_bone_binary(brightness=brightness)

        height, width = self.bone_binary.shape

        center_x = self.midpoints[0]

        for y in range(height):
            # 中央から左に向かって白いピクセルを探索
            left_index = -1
            for x in range(center_x, -1, -1):
                if self.bone_binary[y, x] == 255:
                    left_index = x
                    break

            # 中央から右に向かって白いピクセルを探索
            right_index = -1
            for x in range(center_x, width):
                if self.bone_binary[y, x] == 255:
                    right_index = x
                    break

            # 両方の白いピクセルが見つかった場合
            if left_index != -1 and right_index != -1:
                return (left_index, y), (right_index, y)

        return None, None

    # プロンプトに使用する垂直方向の位置の検出
    def get_vertical_position(self):
        left_heel, right_heel = self.get_heel_positions()
        left_tip, right_tip = self.get_tip_position()

        M1M2_vertical_left = int((left_heel[1] - left_tip[1]) * 0.4 + left_tip[1])
        M1M2_vertical_right = int((right_heel[1] - right_tip[1]) * 0.4 + right_tip[1])

        P1_vertival_left = int((left_heel[1] - left_tip[1]) * 0.18 + left_tip[1])
        P1_vertival_right = int((right_heel[1] - right_tip[1]) * 0.18 + right_tip[1])

        M5_vertical_left = int((left_heel[1] - left_tip[1]) * 0.52 + left_tip[1])
        M5_vertical_right = int((right_heel[1] - right_tip[1]) * 0.52 + right_tip[1])

        background1_left = int((left_heel[1] - left_tip[1]) * 0.7 + left_tip[1])
        background1_right = int((right_heel[1] - right_tip[1]) * 0.7 + right_tip[1])

        return (
            (M1M2_vertical_left, M1M2_vertical_right),
            (P1_vertival_left, P1_vertival_right),
            (M5_vertical_left, M5_vertical_right),
            (background1_left, background1_right),
        )

    # プロンプトを取得する
    def find_prompt(self, start_position, direction, mode):
        x, y = start_position
        if direction == "right":
            step = 1
        else:
            step = -1

        def find_transition(x, y, step, change_from, change_to):
            while 0 <= x < self.bone_binary.shape[1]:
                if (
                    self.bone_binary[y, x] == change_from
                    and self.bone_binary[y, x + step] == change_to
                ):
                    return x
                x += step
            return None

        if mode == "M1M2":
            first_transition = find_transition(x, y, step, 0, 255)
            if first_transition is not None:
                if direction == "right":
                    first_position = first_transition + 50
                else:
                    first_position = first_transition - 50

                second_transition = find_transition(first_position, y, step, 0, 255)

                if second_transition is not None:
                    if direction == "right":
                        second_position = second_transition + 30
                    else:
                        second_position = second_transition - 30

                    return (first_position, y), (second_position, y)

        elif mode == "P1":
            first_transition = find_transition(x, y, step, 0, 255)
            if first_transition is not None:
                if direction == "right":
                    first_position = first_transition + 50
                else:
                    first_position = first_transition - 50

                return first_position, y

        elif mode == "M5":
            first_transition = find_transition(x, y, step, 0, 255)
            if first_transition is not None:
                if direction == "right":
                    first_position = first_transition + 30
                else:
                    first_position = first_transition - 30

                return first_position, y

        return None

    # 背景のプロンプトを取得
    def find_background_prompt(self, start_position, direction):
        height, width = self.bone_binary.shape

        x, y = start_position

        first_change = None
        second_change = None

        while 0 <= x < width and 0 <= y < height:
            current_pixel = self.bone_binary[y, x]

            if first_change is None and current_pixel == 255:
                first_change = (x, y)

            elif (
                first_change is not None
                and second_change is None
                and current_pixel == 0
            ):
                second_change = (x, y)
                break

            if direction == "right":
                x += 1
            elif direction == "left":
                x -= 1

        if first_change is not None and second_change is not None:
            midpoint = (
                (first_change[0] + second_change[0]) // 2,
                (first_change[1] + second_change[1]) // 2,
            )
            return midpoint
        else:
            return None

    def get_prompt(self):
        (
            M1M2_vertival,
            P1_vertical,
            M5_vertical,
            background1_vertical,
        ) = self.get_vertical_position()

        height, width = self.image.shape

        M1_left, M2_left = self.find_prompt(
            start_position=(self.midpoints[0], M1M2_vertival[0]),
            direction="left",
            mode="M1M2",
        )

        M1_right, M2_right = self.find_prompt(
            start_position=(self.midpoints[0], M1M2_vertival[1]),
            direction="right",
            mode="M1M2",
        )

        P1_left = self.find_prompt(
            start_position=(self.midpoints[0], P1_vertical[0]),
            direction="left",
            mode="P1",
        )
        P1_right = self.find_prompt(
            start_position=(self.midpoints[0], P1_vertical[1]),
            direction="right",
            mode="P1",
        )

        M5_left = self.find_prompt(
            start_position=(0, M5_vertical[0]),
            direction="right",
            mode="M5",
        )
        M5_right = self.find_prompt(
            start_position=(width - 1, M5_vertical[1]),
            direction="left",
            mode="M5",
        )

        background1_left = self.find_background_prompt(
            start_position=(self.midpoints[0], background1_vertical[0]),
            direction="left",
        )

        background1_right = self.find_background_prompt(
            start_position=(self.midpoints[0], background1_vertical[1]),
            direction="right",
        )

        background3_left = (
            int((M2_left[0] + M5_left[0]) / 2),
            int((M2_left[1] + M5_left[1]) / 2),
        )
        background3_right = (
            int((M2_right[0] + M5_right[0]) / 2),
            int((M2_right[1] + M5_right[1]) / 2),
        )

        return (
            (M1_left, M1_right),
            (M2_left, M2_right),
            (P1_left, P1_right),
            (M5_left, M5_right),
            (background1_left, background1_right),
            (background3_left, background3_right),
        )


if __name__ == "__main__":
    input_dir = "detect_hallux_valgus/src/preprocessed"
    output_dir = "detect_hallux_valgus/output/prompt"

    os.makedirs(output_dir, exist_ok=True)

    for image_path in glob.glob(os.path.join(input_dir, "*.png")):
        print(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        height, width = image.shape
        get_prompt_position = GetPromptPosition(image=image)

        M1, M2, P1, M5, background1, background3 = get_prompt_position.get_prompt()

        print(M1, M2, P1, M5, background1, background3)

        cv2.circle(
            img=color_image,
            center=M1[0],
            radius=10,
            color=(0, 0, 255),
            thickness=-1,
        )

        cv2.circle(
            img=color_image,
            center=M1[1],
            radius=10,
            color=(0, 0, 255),
            thickness=-1,
        )
        cv2.circle(
            img=color_image,
            center=M2[0],
            radius=10,
            color=(0, 0, 255),
            thickness=-1,
        )

        cv2.circle(
            img=color_image,
            center=M2[1],
            radius=10,
            color=(0, 0, 255),
            thickness=-1,
        )

        cv2.circle(
            img=color_image,
            center=P1[0],
            radius=10,
            color=(0, 0, 255),
            thickness=-1,
        )

        cv2.circle(
            img=color_image,
            center=P1[1],
            radius=10,
            color=(0, 0, 255),
            thickness=-1,
        )

        cv2.circle(
            img=color_image,
            center=M5[0],
            radius=10,
            color=(0, 0, 255),
            thickness=-1,
        )

        cv2.circle(
            img=color_image,
            center=M5[1],
            radius=10,
            color=(0, 0, 255),
            thickness=-1,
        )

        cv2.circle(
            img=color_image,
            center=background1[0],
            radius=10,
            color=(0, 255, 0),
            thickness=-1,
        )

        cv2.circle(
            img=color_image,
            center=background1[1],
            radius=10,
            color=(0, 255, 0),
            thickness=-1,
        )

        cv2.circle(
            img=color_image,
            center=background3[0],
            radius=10,
            color=(0, 255, 0),
            thickness=-1,
        )

        cv2.circle(
            img=color_image,
            center=background3[1],
            radius=10,
            color=(0, 255, 0),
            thickness=-1,
        )

        # # ============================================== 使用した部分の線を引く ===================================================
        # midpoints, contours = (
        #     get_prompt_position.detect_mid_points.find_contour_midpoints(
        #         image=clahe_image
        #     )
        # )

        # positions, brightness = (
        #     get_prompt_position.make_bone.find_lower_brightness_positions(
        #         image=clahe_image, start_position=midpoints
        #     )
        # )

        # bone_binary, bone = get_prompt_position.make_bone.make_bone_binary(
        #     clahe_image=clahe_image, brightness=brightness
        # )

        # color_image = cv2.cvtColor(bone_binary, cv2.COLOR_GRAY2BGR)

        # left_heel, right_heel = get_prompt_position.get_heel_positions(
        #     clahe_image=clahe_image
        # )

        # left_tip, right_tip = get_prompt_position.get_tip_position(
        #     clahe_image=clahe_image
        # )

        # # =============================高さ調節=================================

        # # M1M2用
        # left_L = int((left_heel[1] - left_tip[1]) * 0.38 + left_tip[1])
        # right_L = int((right_heel[1] - right_tip[1]) * 0.38 + right_tip[1])

        # start_L = (0, left_L)
        # end_L = (int(width / 2), left_L)

        # start_R = (int(width / 2 + 1), right_L)
        # end_R = (width - 1, right_L)
        # cv2.line(color_image, start_L, end_L, (0, 0, 255), thickness=8)
        # cv2.line(color_image, start_R, end_R, (0, 0, 255), thickness=8)

        # # P1用
        # left_L = int((left_heel[1] - left_tip[1]) * 0.2 + left_tip[1])
        # right_L = int((right_heel[1] - right_tip[1]) * 0.2 + right_tip[1])

        # start_L = (0, left_L)
        # end_L = (int(width / 2), left_L)

        # start_R = (int(width / 2 + 1), right_L)
        # end_R = (width - 1, right_L)
        # cv2.line(color_image, start_L, end_L, (0, 0, 255), thickness=8)
        # cv2.line(color_image, start_R, end_R, (0, 0, 255), thickness=8)

        # # M5用
        # left_L = int((left_heel[1] - left_tip[1]) * 0.47 + left_tip[1])
        # right_L = int((right_heel[1] - right_tip[1]) * 0.47 + right_tip[1])

        # start_L = (0, left_L)
        # end_L = (int(width / 2), left_L)

        # start_R = (int(width / 2 + 1), right_L)
        # end_R = (width - 1, right_L)
        # cv2.line(color_image, start_L, end_L, (0, 255, 255), thickness=8)
        # cv2.line(color_image, start_R, end_R, (0, 255, 255), thickness=8)

        # cv2.circle(
        #     img=color_image,
        #     center=left_heel,
        #     radius=20,
        #     color=(0, 255, 0),
        #     thickness=-1,
        # )
        # cv2.circle(
        #     img=color_image,
        #     center=right_heel,
        #     radius=20,
        #     color=(0, 255, 0),
        #     thickness=-1,
        # )
        # cv2.circle(
        #     img=color_image, center=left_tip, radius=20, color=(0, 255, 0), thickness=-1
        # )
        # cv2.circle(
        #     img=color_image,
        #     center=right_tip,
        #     radius=20,
        #     color=(0, 255, 0),
        #     thickness=-1,
        # )
        # # ============================================== 使用した部分の線を引く ===================================================

        output_image_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_image_path, color_image)
