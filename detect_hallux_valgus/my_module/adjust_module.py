import numpy as np
import cv2


class AdjustModule:
    def find_edge_intersections(self, mask_image, p1, p2, center):
        current_distance_positive = 0
        current_distance_negative = 0

        # ベクトルの正規化
        # 第二主成分軸ベクトルの正と負の定義
        p1_unit = p1 / np.linalg.norm(p1)
        p2_positive = p2
        p2_negative = (-p2[0], -p2[1])

        current_point = np.array(center)

        step_size = 1

        current_distance_positive = 0
        current_distance_negative = 0

        # 第一主成分軸正方向に探索
        while True:
            current_point += p1_unit * step_size

            # 現在の位置がオブジェクト外かどうかを確認
            if not self.is_inside_object(mask_image, current_point):
                break

            (
                distance_positive,
                end_point_positive_1,
            ) = self.find_distance_to_black_pixel(
                mask_image=mask_image,
                start_point=current_point,
                direction_vector=p2_positive,
            )

            (
                distance_negative,
                end_point_positive_2,
            ) = self.find_distance_to_black_pixel(
                mask_image=mask_image,
                start_point=current_point,
                direction_vector=p2_negative,
            )

            # 重心からの距離を計算
            distance = distance_positive + distance_negative

            # 最大の長さを更新
            if distance > current_distance_positive:
                current_distance_positive = distance
                end_point1 = end_point_positive_1
                end_point2 = end_point_positive_2

        middle_positive = (
            ((end_point1[0] + end_point2[0])) // 2,
            (end_point1[1] + end_point2[1]) // 2,
        )

        # 第一主成分軸負方向に探索
        # 現在の位置を中心にリセット
        current_point = np.array(center)
        while True:
            current_point -= p1_unit * step_size

            # 現在の位置がオブジェクト外かどうかを確認
            if not self.is_inside_object(mask_image, current_point):
                break

            (
                distance_positive,
                end_point_negative_1,
            ) = self.find_distance_to_black_pixel(
                mask_image=mask_image,
                start_point=current_point,
                direction_vector=p2_positive,
            )

            (
                distance_negative,
                end_point_negative_2,
            ) = self.find_distance_to_black_pixel(
                mask_image=mask_image,
                start_point=current_point,
                direction_vector=p2_negative,
            )
            # 第二主成分軸の距離を測定
            distance = distance_positive + distance_negative

            # 最大の長さを更新
            if distance > current_distance_negative:
                current_distance_negative = distance
                end_point3 = end_point_negative_1
                end_point4 = end_point_negative_2

        middle_negative = (
            ((end_point3[0] + end_point4[0])) // 2,
            (end_point3[1] + end_point4[1]) // 2,
        )

        # 直線上を1/4の位置にある座標を計算
        quarter_point = self.interpolate_point(middle_positive, middle_negative, 0.25)
        # 直線上を3/4の位置にある座標を計算
        third_point = self.interpolate_point(middle_positive, middle_negative, 0.75)

        nulldistace, quarter_point_1 = self.find_distance_to_black_pixel(
            mask_image=mask_image,
            start_point=quarter_point,
            direction_vector=p2_positive,
        )
        nulldistace, quarter_point_2 = self.find_distance_to_black_pixel(
            mask_image=mask_image,
            start_point=quarter_point,
            direction_vector=p2_negative,
        )
        nulldistace, quarter_point_3 = self.find_distance_to_black_pixel(
            mask_image=mask_image, start_point=third_point, direction_vector=p2_positive
        )
        nulldistace, quarter_point_4 = self.find_distance_to_black_pixel(
            mask_image=mask_image, start_point=third_point, direction_vector=p2_negative
        )

        return (quarter_point_1, quarter_point_2, quarter_point_3, quarter_point_4)

    # 対象の点がobject内にあるかの検出
    def is_inside_object(self, object_image, point):
        x, y = int(point[0]), int(point[1])
        return object_image[y, x] == 255

    # directionの方向に黒いピクセルが検出できるまで探索し、その座標位置を返す
    def find_distance_to_black_pixel(self, mask_image, start_point, direction_vector):
        direction_unit = direction_vector / np.linalg.norm(direction_vector)
        current_point = np.array(start_point)
        step_size = 1
        found_black_pixel = False
        end_point = None

        while True:
            # 現在の座標位置を記録
            current_x, current_y = int(current_point[0]), int(current_point[1])

            # 黒いピクセルを検出したら探索を終了
            pixel_color = mask_image[current_y, current_x]
            if pixel_color == 0:
                found_black_pixel = True
                end_point = (current_x, current_y)
                break

            current_point += direction_unit * step_size

        distance = np.linalg.norm(np.array(end_point) - start_point)
        return distance, end_point

    def interpolate_point(self, point1, point2, fraction):
        x1, y1 = point1
        x2, y2 = point2
        interpolated_point = (x1 + fraction * (x2 - x1), y1 + fraction * (y2 - y1))
        return interpolated_point

    # ベクトルを入力にしても座標値を入力しても診断基準が引けるようにする
    def draw_line(self, image, center, vec, point, line_length):
        # HV角に関しては重心と第一主成分軸から線を引く
        if vec is not None and center is not None:
            line_color = (0, 255, 0)

            # 第一主成分軸の正方向に線を描画
            cv2.line(
                image,
                (int(center[0]), int(center[1])),
                (
                    int(center[0] + vec[0] * line_length),
                    int(center[1] + vec[1] * line_length),
                ),
                line_color,
                5,
            )
            # 第一主成分軸の負方向に線を描画
            cv2.line(
                image,
                (int(center[0]), int(center[1])),
                (
                    int(center[0] - vec[0] * line_length * 2),
                    int(center[1] - vec[1] * line_length * 2),
                ),
                line_color,
                5,
            )

        # M1M5角に関しては、二つの座標から線を引く
        if point is not None:
            # 指定した直線の長さ
            # 直線の色（青色）
            line_color = (0, 255, 0)

            # 2つの座標点からベクトルを計算
            dx = point[1][0] - point[0][0]
            dy = point[1][1] - point[0][1]

            # ベクトルの長さを計算
            vector_length = np.sqrt(dx**2 + dy**2)

            # ベクトルを指定した長さにスケーリング
            scaled_dx = (dx / vector_length) * 1300
            scaled_dy = (dy / vector_length) * 1300

            # 直線を描画
            cv2.line(
                image,
                (int(point[0][0]), int(point[0][1])),
                (int(point[0][0] + scaled_dx), int(point[0][1] + scaled_dy)),
                line_color,
                thickness=5,
            )
            # ベクトルを指定した長さにスケーリング
            scaled_dx = (dx / vector_length) * 400
            scaled_dy = (dy / vector_length) * 400

            # 反対方向の直線を描画
            cv2.line(
                image,
                (int(point[0][0]), int(point[0][1])),
                (int(point[0][0] - scaled_dx), int(point[0][1] - scaled_dy)),
                line_color,
                thickness=5,
            )

    def draw_text(self, image, direction, mode, angle_difference):
        if angle_difference is not None:
            if mode == "HV":
                if direction == "right":
                    cv2.putText(
                        image,
                        f"Right_HV:{round(angle_difference, 1)}deg ",
                        (1200, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3.0,
                        (0, 0, 0),
                        8,
                        cv2.LINE_AA,
                    )
                elif direction == "left":
                    cv2.putText(
                        image,
                        f"Left_HV:{round(angle_difference, 1)}deg ",
                        (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3.0,
                        (0, 0, 0),
                        8,
                        cv2.LINE_AA,
                    )
            if mode == "M1M5":
                if direction == "right":
                    cv2.putText(
                        image,
                        f"Right_M1M5:{round(angle_difference, 1)}deg ",
                        (1200, 300),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3.0,
                        (0, 0, 0),
                        8,
                        cv2.LINE_AA,
                    )
                elif direction == "left":
                    cv2.putText(
                        image,
                        f"Left_M1M5:{round(angle_difference, 1)}deg ",
                        (100, 300),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3.0,
                        (0, 0, 0),
                        8,
                        cv2.LINE_AA,
                    )
