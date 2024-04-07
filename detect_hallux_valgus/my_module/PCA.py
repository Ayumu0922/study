import cv2
import numpy as np
import math
import os
import numpy as np


class AngleMeasurement:
    def __init__(self, output_image_path):
        # color image for output
        self.image = cv2.imread(output_image_path, cv2.IMREAD_COLOR)

    def sort_label_images_by_area(self, labelImages):
        # ラベルの総数
        nLabels = np.max(labelImages) + 1
        # ラベルのピクセル数を取得
        sizes = [np.sum(labelImages == label) for label in range(nLabels)]
        # sizesを降順にソート
        sorted_indices = np.argsort(sizes)[::-1]
        # labelImagesと同じサイズの0行列
        labelImages_sorted = np.zeros_like(labelImages)
        # 再ラベリング
        for new_label, old_label in enumerate(sorted_indices):
            labelImages_sorted[labelImages == old_label] = new_label
        labelImages = labelImages_sorted

        return labelImages

    # 1つのオブジェクトを入力にその第一主成分軸ベクトルと重心を取得
    def process_pca(self, object):
        sz = len(object[0])
        data_pts = np.empty((sz, 2), dtype=np.float64)
        data_pts[:, 0] = object[1][:]
        data_pts[:, 1] = object[0][:]

        try:
            # PCA(mean : 平均ベクトル eigenbectors : 固有ベクトル eigenvalues : 固有値)
            mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean=None)
        except cv2.error:
            return None, None

        # 重心
        center = (mean[0, 0], mean[0, 1])

        # 第一主成分軸の方向ベクトル
        p1 = (eigenvectors[0][0], eigenvectors[0][1])

        return center, p1

    def angle_calculation(self, vec1, vec2, mode):
        if mode == "normal":
            if vec1 is None or vec2 is None:
                return None

            angle_1 = math.atan2(vec1[1], vec1[0])

            angle_2 = math.atan2(vec2[1], vec2[0])

            # 2つのvector間の角度差
            angle_difference = abs(math.degrees(angle_2 - angle_1))

            return angle_difference

        elif mode == "modify":
            return None

    def draw_line(self, center, vec, mode):
        # 重心に点を描画
        if center is not None:
            # cv2.circle(
            #     self.image, (int(center[0]), int(center[1])), 10, (255, 0, 0), -1
            # )

            line_length = 800
            if vec is not None:
                # 線の色をmodeに応じて設定
                if mode == "HV":
                    line_color = (0, 0, 255)  # 赤色
                elif mode == "M1M5":
                    line_color = (0, 255, 0)  # 緑色
                else:
                    line_color = (0, 0, 0)  # デフォルトは黒色

                # 第一主成分軸の正方向に線を描画
                cv2.line(
                    self.image,
                    (int(center[0]), int(center[1])),
                    (
                        int(center[0] + vec[0] * line_length),
                        int(center[1] + vec[1] * line_length),
                    ),
                    line_color,  # modeに応じて色を設定
                    5,
                )
                # 第一主成分軸の負方向に線を描画
                cv2.line(
                    self.image,
                    (int(center[0]), int(center[1])),
                    (
                        int(center[0] - vec[0] * line_length * 2),
                        int(center[1] - vec[1] * line_length * 2),
                    ),
                    line_color,  # modeに応じて色を設定
                    5,
                )
        else:
            print("描画処理をスキップ:centerまたはvecがNone")

    def draw_text(self, direction, mode, angle_difference):
        if angle_difference is not None:
            if mode == "HV":
                if direction == "right":
                    cv2.putText(
                        self.image,
                        f"Right_HV:{round(angle_difference, 1)}deg ",
                        (1200, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3.0,
                        (0, 0, 255),
                        8,
                        cv2.LINE_AA,
                    )
                elif direction == "left":
                    cv2.putText(
                        self.image,
                        f"Left_HV:{round(angle_difference, 1)}deg ",
                        (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3.0,
                        (0, 0, 255),
                        8,
                        cv2.LINE_AA,
                    )
            if mode == "M1M5":
                if direction == "right":
                    cv2.putText(
                        self.image,
                        f"Right_M1M5:{round(angle_difference, 1)}deg ",
                        (1200, 300),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3.0,
                        (0, 255, 0),
                        8,
                        cv2.LINE_AA,
                    )
                elif direction == "left":
                    cv2.putText(
                        self.image,
                        f"Left_M1M5:{round(angle_difference, 1)}deg ",
                        (100, 300),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3.0,
                        (0, 255, 0),
                        8,
                        cv2.LINE_AA,
                    )

    def process_mask_image(self, mask, mode):
        if mode == "HV":
            # label_HVの数は２つ
            _, label_HV = cv2.connectedComponents(mask)
            label_HV = self.sort_label_images_by_area(label_HV)
            if label_HV is not None:
                object1 = np.where(label_HV == 1)
                object2 = np.where(label_HV == 2)

                center1, vec_1 = self.process_pca(object1)
                center2, vec_2 = self.process_pca(object2)

                return (center1, vec_1), (center2, vec_2)
            else:
                return None, None

        elif mode == "M1M5":
            _, label_M1M5 = cv2.connectedComponents(mask)
            label_M1M5 = self.sort_label_images_by_area(label_M1M5)
            if label_M1M5 is not None:
                object1 = np.where(label_M1M5 == 1)
                center1, vec_1 = self.process_pca(object1)
                return (center1, vec_1)
            else:
                return None, None

    def main(self, mask_HV_right, mask_HV_left, mask_M1M5_right, mask_M1M5_left):
        import time

        start = time.time()
        HV_r_1, HV_r_2 = self.process_mask_image(mask_HV_right, mode="HV")
        HV_l_1, HV_l_2 = self.process_mask_image(mask_HV_left, mode="HV")

        M1M5_r_1 = self.process_mask_image(mask_M1M5_right, mode="M1M5")
        M1M5_l_1 = self.process_mask_image(mask_M1M5_left, mode="M1M5")

        # 右足のHV
        self.draw_line(HV_r_1[0], HV_r_1[1], mode="HV")
        self.draw_line(HV_r_2[0], HV_r_2[1], mode="HV")
        # 左足のHV
        self.draw_line(HV_l_1[0], HV_l_1[1], mode="HV")
        self.draw_line(HV_l_2[0], HV_l_2[1], mode="HV")

        # # 右足のM1M5
        self.draw_line(M1M5_r_1[0], M1M5_r_1[1], mode="M1M5")
        # # 左足のM1M5
        self.draw_line(M1M5_l_1[0], M1M5_l_1[1], mode="M1M5")

        # HV角
        angle_HV_r = self.angle_calculation(HV_r_1[1], HV_r_2[1], mode="normal")
        angle_HV_l = self.angle_calculation(HV_l_1[1], HV_l_2[1], mode="normal")

        # # M1-M5角
        angle_M1M5_r = self.angle_calculation(HV_r_1[1], M1M5_r_1[1], mode="normal")
        angle_M1M5_l = self.angle_calculation(HV_l_1[1], M1M5_l_1[1], mode="normal")

        # 画像に角度を描画

        self.draw_text(direction="right", mode="HV", angle_difference=angle_HV_r)
        self.draw_text(direction="left", mode="HV", angle_difference=angle_HV_l)

        self.draw_text(direction="right", mode="M1M5", angle_difference=angle_M1M5_r)
        self.draw_text(direction="left", mode="M1M5", angle_difference=angle_M1M5_l)

        end = time.time()
        print("計測時間：", end - start)

        return self.image


if __name__ == "__main__":
    image_folder = "src/images"
    input_folderHV = "assets/MASKHV"
    input_folderM1M5 = "assets/MASKM1M5"

    output_folder = "assets/診断結果（修正版）"
    if not os.path.exists(input_folderHV):
        os.makedirs(input_folderHV)
    if not os.path.exists(input_folderM1M5):
        os.makedirs(input_folderM1M5)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for input_file_name in os.listdir(image_folder):
        print(input_file_name)
        file_number = input_file_name.split("part")[1].split(".")[0]
        file_name_prefix = f"part{file_number}"

        right_HV_filename = f"{file_name_prefix}_right_HV.png"
        left_HV_filename = f"{file_name_prefix}_left_HV.png"
        right_M1M5_filename = f"{file_name_prefix}_right_M1M5.png"
        left_M1M5_filename = f"{file_name_prefix}_left_M1M5.png"

        input_image_path = os.path.join(image_folder, input_file_name)
        print(input_image_path)

        right_HV_path = os.path.join(input_folderHV, right_HV_filename)
        left_HV_path = os.path.join(input_folderHV, left_HV_filename)
        right_M1M5_path = os.path.join(input_folderM1M5, right_M1M5_filename)
        left_M1M5_path = os.path.join(input_folderM1M5, left_M1M5_filename)

        # connexted components with statsを使用するので白黒
        HV_right = cv2.imread(right_HV_path, cv2.IMREAD_GRAYSCALE)
        HV_left = cv2.imread(left_HV_path, cv2.IMREAD_GRAYSCALE)
        M1M5_right = cv2.imread(right_M1M5_path, cv2.IMREAD_GRAYSCALE)
        M1M5_left = cv2.imread(left_M1M5_path, cv2.IMREAD_GRAYSCALE)

        anglemeasurement = AngleMeasurement(input_image_path)
        image = anglemeasurement.main(HV_right, HV_left, M1M5_right, M1M5_left)
        output_image_path = os.path.join(output_folder, input_file_name)
        cv2.imwrite(output_image_path, image)
