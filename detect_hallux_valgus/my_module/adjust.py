import cv2
import numpy as np
import math
import os
from adjust_module import AdjustModule


class Adjustment:
    def __init__(self, right_HV_path, left_HV_path, right_M1M5_path, left_M1M5_path):
        self.HV_right = cv2.imread(right_HV_path, cv2.IMREAD_GRAYSCALE)
        self.HV_left = cv2.imread(left_HV_path, cv2.IMREAD_GRAYSCALE)
        self.M1M5_right = cv2.imread(right_M1M5_path, cv2.IMREAD_GRAYSCALE)
        self.M1M5_left = cv2.imread(left_M1M5_path, cv2.IMREAD_GRAYSCALE)

    def sort_label_images(self, labelImages):
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

    # オブジェクトからベクトルと重心を取得（）
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
        p2 = (eigenvectors[1][0], eigenvectors[1][1])

        return center, p1, p2

    # 二つのベクトルを入力に角度を測定する（このベクトルはprocess_pcaから取得）
    def angle_calculation(self, vec1, vec2):
        if vec1 is None or vec2 is None:
            print("vec1 vec2が適切に取得できませんでした")
            return None
        else:
            angle_1 = math.atan2(vec1[1], vec1[0])
            angle_2 = math.atan2(vec2[1], vec2[0])

            # 2つのvector間の角度差
            angle_difference = abs(math.degrees(angle_2 - angle_1))

            return angle_difference

    # 画像を入力にマスクのラベリングから第一主成分軸の計算までを一連の処理でまとめある
    def process_mask_image(self, mask, mode):
        if mode == "HV":
            # label_HVの数は２つ
            _, label_HV = cv2.connectedComponents(mask)
            label_HV = self.sort_label_images(label_HV)
            if label_HV is not None:
                object1 = np.where(label_HV == 1)
                object2 = np.where(label_HV == 2)

                center1, vec_1_1, vec_1_2 = self.process_pca(object1)
                center2, vec_2_1, vec_2_2 = self.process_pca(object2)

                return (center1, vec_1_1, vec_1_2), (center2, vec_2_1, vec_2_2)
            else:
                return None, None, None

        elif mode == "M1M5":
            _, label_M1M5 = cv2.connectedComponents(mask)
            label_M1M5 = self.sort_label_images(label_M1M5)
            if label_M1M5 is not None:
                object1 = np.where(label_M1M5 == 1)
                center1, vec_1_1, vec_1_2 = self.process_pca(object1)
                return (center1, vec_1_1, vec_1_2)
            else:
                return None, None, None

    def angle_calculation(self, vec1, vec2):
        if vec1 is None or vec2 is None:
            return None

        angle_1 = math.atan2(vec1[1], vec1[0])
        angle_2 = math.atan2(vec2[1], vec2[0])
        # 2つのvector間の角度差
        angle_difference = abs(math.degrees(angle_2 - angle_1))

        return angle_difference

    def draw_points_on_image(self, img, points, color):
        for point in points:
            cv2.circle(img, point, 10, color, -1)
        return img

    # segment-anythingで取得したマスクをmaskに入力する。またoutput_ijjmageに出力対象となる画像を選択する
    def main(self, output_image):
        adjust_module = AdjustModule()

        # ==========================================右足の検出===============================================
        # ==========================HV===========================
        (
            (center1_R, vec_1_1_R, vec_1_2_R),
            (center2_R, vec_2_1_R, vec_2_2_R),
        ) = self.process_mask_image(mask=self.HV_right, mode="HV")
        adjust_module.draw_line(
            output_image, center1_R, vec_1_1_R, None, line_length=650
        )
        adjust_module.draw_line(
            output_image, center2_R, vec_2_1_R, None, line_length=250
        )

        # ==========================M1M5===========================
        (center1, vec_1_1, vec_1_2) = self.process_mask_image(
            mask=self.M1M5_right, mode="M1M5"
        )

        (quarter_point_right) = adjust_module.find_edge_intersections(
            mask_image=self.M1M5_right, p1=vec_1_1, p2=vec_1_2, center=center1
        )

        self.draw_points_on_image(
            img=output_image, points=quarter_point_right, color=(0, 0, 255)
        )

        middle_point1_R = (
            (quarter_point_right[0][0] + quarter_point_right[1][0]) // 2,
            (quarter_point_right[0][1] + quarter_point_right[1][1]) // 2,
        )
        middle_point2_R = (
            (quarter_point_right[2][0] + quarter_point_right[3][0]) // 2,
            (quarter_point_right[2][1] + quarter_point_right[3][1]) // 2,
        )

        adjust_module.draw_line(
            output_image,
            None,
            None,
            ((middle_point1_R, middle_point2_R)),
            line_length=None,
        )

        # ==========================================左足の検出===============================================
        # ==========================HV===========================
        (
            (center1_L, vec_1_1_L, vec_1_2_L),
            (center2_L, vec_2_1_L, vec_2_2_L),
        ) = self.process_mask_image(mask=self.HV_left, mode="HV")
        adjust_module.draw_line(
            output_image, center1_L, vec_1_1_L, None, line_length=650
        )
        adjust_module.draw_line(
            output_image, center2_L, vec_2_1_L, None, line_length=250
        )

        # ==========================M1M5===========================
        (center1, vec_1_1, vec_1_2) = self.process_mask_image(
            mask=self.M1M5_left, mode="M1M5"
        )

        (quarter_point_left) = adjust_module.find_edge_intersections(
            mask_image=self.M1M5_left, p1=vec_1_1, p2=vec_1_2, center=center1
        )

        self.draw_points_on_image(
            img=output_image, points=quarter_point_left, color=(0, 0, 255)
        )

        middle_point1_L = (
            (quarter_point_left[0][0] + quarter_point_left[1][0]) // 2,
            (quarter_point_left[0][1] + quarter_point_left[1][1]) // 2,
        )
        middle_point2_L = (
            (quarter_point_left[2][0] + quarter_point_left[3][0]) // 2,
            (quarter_point_left[2][1] + quarter_point_left[3][1]) // 2,
        )

        adjust_module.draw_line(
            output_image,
            None,
            None,
            ((middle_point1_L, middle_point2_L)),
            line_length=None,
        )

        # =======================================角度算出とテキスト描画===============================================
        # HV_Right
        HV_R = self.angle_calculation(vec_1_1_R, vec_2_1_R)
        # adjust_module.draw_text(
        #     output_image, direction="right", mode="HV", angle_difference=HV_R
        # )
        # HV_Left
        HV_L = self.angle_calculation(vec_1_1_L, vec_2_1_L)
        # adjust_module.draw_text(
        #     output_image, direction="left", mode="HV", angle_difference=HV_L
        # )
        # M1M5_Right
        vector_M1M5_R = (
            middle_point1_R[0] - middle_point2_R[0],
            middle_point1_R[1] - middle_point2_R[1],
        )
        M1M5_Right = self.angle_calculation(vector_M1M5_R, vec_1_1_R)
        # adjust_module.draw_text(
        #     output_image, direction="right", mode="M1M5", angle_difference=M1M5_Right
        # )

        print(f"M1M5_Right {M1M5_Right:.1f}")

        # M1M5_Left
        vector_M1M5_L = (
            middle_point1_L[0] - middle_point2_L[0],
            middle_point1_L[1] - middle_point2_L[1],
        )
        M1M5_Left = self.angle_calculation(vector_M1M5_L, vec_1_1_L)
        # adjust_module.draw_text(
        #     output_image, direction="left", mode="M1M5", angle_difference=M1M5_Left
        # )
        print(f"M1M5_Left {M1M5_Left:.1f}")

        return output_image


if __name__ == "__main__":
    import os

    image_folder = "src/images"
    input_folderHV = "assets/MASKHV"
    input_folderM1M5 = "assets/MASKM1M5"

    for input_file_name in os.listdir(image_folder):
        file_number = input_file_name.split("part")[1].split(".")[0]
        print(input_file_name)
        file_name_prefix = f"part{file_number}"
        right_HV_filename = f"{file_name_prefix}_right_HV.png"
        left_HV_filename = f"{file_name_prefix}_left_HV.png"
        right_M1M5_filename = f"{file_name_prefix}_right_M1M5.png"
        left_M1M5_filename = f"{file_name_prefix}_left_M1M5.png"

        input_image_path = os.path.join(image_folder, input_file_name)

        right_HV_path = os.path.join(input_folderHV, right_HV_filename)
        left_HV_path = os.path.join(input_folderHV, left_HV_filename)
        right_M1M5_path = os.path.join(input_folderM1M5, right_M1M5_filename)
        left_M1M5_path = os.path.join(input_folderM1M5, left_M1M5_filename)

        adjustment = Adjustment(
            right_HV_path, left_HV_path, right_M1M5_path, left_M1M5_path
        )

        output_image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
        output_image = adjustment.main(output_image)

        assets_directory = "assets/診断結果（10月19日）"
        if not os.path.exists(assets_directory):
            os.makedirs(assets_directory)

        # 画像を assets ディレクトリに保存します
        output_image_path = os.path.join(assets_directory, input_file_name)

        cv2.imwrite(output_image_path, output_image)
