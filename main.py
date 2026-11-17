import cv2  #카메라, 특징점 계산
import numpy as np  #배열 계산
from ultralytics import YOLO    #객체 탐지
import open3d as o3d    #3D 맵

model = YOLO("yolo11n.pt")
orb = cv2.ORB_create(nfeatures = 1000)
capture = cv2.VideoCapture(1)

previous_keypoints = None
previous_descriptors = None

matcher = cv2.BFMatcher(    #디스크럽터 비교
    cv2.NORM_HAMMING,
    crossCheck = True
)

vis = o3d.visualization.Visualizer()

if not vis.create_window(window_name = '3d맵'):
    raise RuntimeError("3d창 생성 오류")

cloud = o3d.geometry.PointCloud()
cloud_added = False

vis.get_render_option().point_size = 4.0

map_points = None
map_descriptors = None
keyframe = None

def triangulate_map_points(K, R1, t1, R2, t2, uv1, uv2):
    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))
    points_4d = cv2.triangulatePoints(
        P1, P2, uv1.T, uv2.T
    )

    valid = (
        np.isfinite(points_4d).all(axis = 0)
        &(np.abs(points_4d[3]) > 1e-8)
    )
    indices = np.flatnonzero(valid)
    xyz = (points_4d[:3, valid] / points_4d[3, valid]).T
    camera1 = (R1 @ xyz.T + t1).T
    camera2 = (R2 @ xyz.T + t2).T

    front = (
            (camera1[:, 2] > 1e-8)
            & (camera2[:, 2] > 1e-8)
    )
    indices = indices[front]
    xyz = xyz[front]
    camera1 = camera1[front]
    camera2 = camera2[front]

    image1 = camera1 @ K.T
    image2 = camera2 @ K.T

    error1 = np.linalg.norm(
        image1[:, :2] / image1[:, 2:3] - uv1[indices],
        axis=1
    )
    error2 = np.linalg.norm(
        image2[:, :2] / image2[:, 2:3] - uv2[indices],
        axis=1
    )

    ray1 = xyz + (R1.T @ t1).ravel()
    ray2 = xyz + (R2.T @ t2).ravel()

    cosine = np.sum(ray1 * ray2, axis=1) / (
            np.linalg.norm(ray1, axis=1)
            * np.linalg.norm(ray2, axis=1)
    )

    good = (
            (error1 < 3.0)
            & (error2 < 3.0)
            & (cosine < np.cos(np.deg2rad(1.5)))
    )

    return xyz[good], indices[good]

reset_requested = False
reset_button = (10, 10, 170, 50)


def on_video_mouse(event, x, y, flags, param):
    global reset_requested

    x1, y1, x2, y2 = reset_button

    if event == cv2.EVENT_LBUTTONDOWN and x1 <= x <= x2 and y1 <= y <= y2:
        reset_requested = True


cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("video", on_video_mouse)

while True:
    movements = []
    inlier_matches = []
    points_3d = None
    map_changed = False

    if reset_requested:
        map_points = None
        map_descriptors = None
        keyframe = None

        previous_keypoints = None
        previous_descriptors = None

        vis.clear_geometries()
        cloud.clear()
        cloud_added = False

        reset_requested = False
        print("지도를 초기화했습니다. 새 지도를 생성합니다.")

    ret, frame = capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray,None)
##-----------------------------------
    height, width = gray.shape
    focal = float(width)

    K = np.array([
        [focal, 0, width/2],
        [0, focal, height/2],
        [0, 0, 1]
    ], dtype = np.float64)
##-----------------------------------
    results = model(frame, conf = 0.5, verbose = False)
    annotated_frame = results[0].plot()
    annotated_frame = cv2.drawKeypoints(
        annotated_frame,
        keypoints,
        None,
        color=(0, 255, 0)
    )

    if (
        map_points is None
        and previous_descriptors is not None
        and descriptors is not None
    ):
        matches = matcher.match(
            previous_descriptors,
            descriptors
        )
        matches = sorted(matches, key=lambda m: m.distance)
        matches = sorted(matches, key=lambda m: m.distance)
        for m in matches:
            previous_x, previous_y = previous_keypoints[m.queryIdx].pt
            current_x, current_y = keypoints[m.trainIdx].pt

            dx = current_x - previous_x
            dy = current_y - previous_y

            movements.append([dx, dy])
        if len(matches) >= 20:
            previous_points = np.float32([
                previous_keypoints[m.queryIdx].pt
                for m in matches
            ])
            current_points = np.float32([
                keypoints[m.trainIdx].pt
                for m in matches
            ])

            F, mask = cv2.findFundamentalMat(
                previous_points,
                current_points,
                cv2.FM_RANSAC,
                2.0,
                0.99
            )
            if F is not None and mask is not None:
                for m, keep in zip(matches, mask.ravel()):
                    if keep == 1:
                        inlier_matches.append(m)
        R = None
        t = None
        pose_mask = None
        count = 0

        if len(inlier_matches) >= 20:
            points1 = np.float32([
                previous_keypoints[m.queryIdx].pt
                for m in inlier_matches
            ])
            points2 = np.float32([
                keypoints[m.trainIdx].pt
                for m in inlier_matches
            ])

            E, essential_mask = cv2.findEssentialMat(
                points1,
                points2,
                K,
                method = cv2.RANSAC,
                prob = 0.999,
                threshold = 1.0
            )

            if E is not None and E.shape == (3, 3):
                count, R, t, pose_mask = cv2.recoverPose(
                    E,
                    points1,
                    points2,
                    K,
                    mask = essential_mask
                )


            if pose_mask is not None and count >= 8:
                keep = pose_mask.ravel() != 0
                selected_points1 = points1[keep]
                selected_points2 = points2[keep]
                current_ids = np.array([
                    m.trainIdx for m in inlier_matches
                ])[keep]

                P1 = K @ np.hstack((
                    np.eye(3),
                    np.zeros((3, 1))
                ))

                P2 = K @ np.hstack((R, t))

                points_4d = cv2.triangulatePoints(
                    P1,
                    P2,
                    selected_points1.T,
                    selected_points2.T
                )

                valid = (
                    np.isfinite(points_4d).all(axis = 0)
                    & (np.abs(points_4d[3]) > 1e-8)
                )

                points_3d = (
                    points_4d[:3, valid] / points_4d[3, valid]
                ).T

                points_in_current = (R @ points_3d.T + t).T

                valid_depth = (
                    np.isfinite(points_3d).all(axis = 1)
                    & (points_3d[:, 2] > 0)
                    & (points_in_current[:, 2] > 0)
                )
                points_3d = points_3d[valid_depth]
                if len(points_3d) >= 20 :
                    current_ids = current_ids[valid][valid_depth]

                    map_points = points_3d.copy()
                    map_descriptors = descriptors[current_ids].copy()

                    keyframe = (
                        keypoints,
                        descriptors.copy(),
                        R.copy(),
                        t.copy(),
                        set(current_ids)
                    )

                    map_changed = True
                    print("첫 지도 생성:", len(map_points), "점")

                print("계산된 3D 점 개수:", len(points_3d))
                print("앞의 5개 좌표:\n", points_3d[:5])


        print("매칭된 점의 개수:", len(matches))

        if movements:
            print("매칭된 점의 이동:", movements[-1])

        print("유효한 이동 점:", len(inlier_matches))

    elif map_points is not None and descriptors is not None:
        map_matches = [
            m for m in matcher.match(map_descriptors, descriptors)
            if m.distance <= 55
        ]

        if len(map_matches) >= 12:
            world_xyz = np.float64([
                map_points[m.queryIdx]
                for m in map_matches
            ])
            image_uv = np.float64([
                keypoints[m.trainIdx].pt
                for m in map_matches
            ])

            ok, rvec, t_now, inliers = cv2.solvePnPRansac(
                world_xyz,
                image_uv,
                K,
                None,
                reprojectionError=3.0,
                flags=cv2.SOLVEPNP_EPNP
            )

            if ok and inliers is not None and len(inliers) >= 12:
                R_now = cv2.Rodrigues(rvec)[0]

                ref_kp, ref_desc, ref_R, ref_t, ref_mapped = keyframe

                center_now = (-R_now.T @ t_now).ravel()
                center_ref = (-ref_R.T @ ref_t).ravel()

                if np.linalg.norm(center_now - center_ref) >= 0.25:
                    current_mapped = {
                        map_matches[i].trainIdx
                        for i in inliers.ravel()
                    }

                    candidates = [
                        m for m in matcher.match(ref_desc, descriptors)
                        if m.distance <= 55
                           and m.queryIdx not in ref_mapped
                           and m.trainIdx not in current_mapped
                    ]

                    if len(candidates) >= 8:
                        uv_ref = np.float64([
                            ref_kp[m.queryIdx].pt
                            for m in candidates
                        ])
                        uv_now = np.float64([
                            keypoints[m.trainIdx].pt
                            for m in candidates
                        ])

                        xyz, indices = triangulate_map_points(
                            K,
                            ref_R, ref_t,
                            R_now, t_now,
                            uv_ref, uv_now
                        )

                        if len(xyz) >= 8:
                            new_ids = [
                                candidates[i].trainIdx
                                for i in indices
                            ]

                            map_points = np.vstack((map_points, xyz))
                            map_descriptors = np.vstack((
                                map_descriptors,
                                descriptors[new_ids]
                            ))

                            keyframe = (
                                keypoints,
                                descriptors.copy(),
                                R_now.copy(),
                                t_now.copy(),
                                current_mapped.union(new_ids)
                            )

                            map_changed = True
                            print(
                                "새 점:", len(xyz),
                                "전체 지도:", len(map_points)
                            )


    if map_changed:
        cloud.points = o3d.utility.Vector3dVector(
            map_points.astype(np.float64)
        )
        cloud.paint_uniform_color([0.1, 0.7, 0.2])

        if not cloud_added:
            cloud_added = vis.add_geometry(cloud)
        else:
            vis.update_geometry(cloud)

    previous_keypoints = keypoints
    previous_descriptors = descriptors

    if not vis.poll_events():
        break

    vis.update_renderer()
    cv2.imshow("video", annotated_frame)
    x1, y1, x2, y2 = reset_button

    cv2.rectangle(
        annotated_frame,
        (x1, y1),
        (x2, y2),
        (60, 60, 180),
        -1
    )

    cv2.putText(
        annotated_frame,
        "Reset Map",
        (x1 + 15, y1 + 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.imshow("video", annotated_frame)
    if cv2.waitKey(1) == ord('q'): break

capture.release()
cv2.destroyAllWindows()
vis.destroy_window()