import cv2
import shutil
import os
import numpy as np
import re


# --------- helpers ---------
def corners_from_contour_geometry(contour: np.ndarray) -> np.ndarray:
    """
    Robust TL, TR, BR, BL directly from the contour geometry
    using min/max of x±y (stable on rotated objects).
    Returns np.float32 array in order [TL, TR, BR, BL].
    """
    pts = contour.reshape(-1, 2).astype(np.float32)
    x, y = pts[:, 0], pts[:, 1]
    s = x + y          # small -> TL, large -> BR
    d = x - y          # large -> TR, small -> BL
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def robust_foreground_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Foreground (container) mask:
    1) HSV green threshold
    2) Keep only border-connected green as background (flood fill)
    3) Foreground = NOT(border-green)
    4) Morph clean-up
    """
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40], dtype=np.uint8)
    upper_green = np.array([95, 255, 255], dtype=np.uint8)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    for x in range(w):
        if green_mask[0, x] != 0: cv2.floodFill(green_mask, ff_mask, (x, 0), 255)
        if green_mask[h-1, x] != 0: cv2.floodFill(green_mask, ff_mask, (x, h-1), 255)
    for y in range(h):
        if green_mask[y, 0] != 0: cv2.floodFill(green_mask, ff_mask, (0, y), 255)
        if green_mask[y, w-1] != 0: cv2.floodFill(green_mask, ff_mask, (w-1, y), 255)

    obj_mask = cv2.bitwise_not(green_mask)
    kernel = np.ones((5, 5), np.uint8)
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return obj_mask


def pick_best_container_contour(mask: np.ndarray):
    """
    Choose the contour that best matches a large, solid object.
    """
    h, w = mask.shape[:2]
    min_frac, max_frac = 0.05, 0.95
    min_solid = 0.80

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return None

    img_area = float(h*w)
    best, best_score = None, -1.0
    for c in cnts:
        area = cv2.contourArea(c)
        if area <= 0: continue
        frac = area / img_area
        if not (min_frac <= frac <= max_frac): continue
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0: continue
        solidity = area / hull_area
        score = frac * (0.5 + 0.5 * min(1.0, (solidity - min_solid) / (1.0 - min_solid + 1e-6)))
        if score > best_score:
            best_score = score
            best = c
    if best is None:
        best = max(cnts, key=cv2.contourArea)
    return best


def filter_small_edges(edge_img: np.ndarray, min_perimeter_px: int = 150) -> np.ndarray:
    """
    Remove tiny Canny fragments. Returns a new binary edge mask.
    """
    filtered = np.zeros_like(edge_img)
    cnts, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        if cv2.arcLength(c, False) >= min_perimeter_px:
            cv2.drawContours(filtered, [c], -1, 255, 1)
    return filtered


# --------- main ---------

def eval_damage_codex(inp_dir, out_path, visualize=False, draw_boundary_only=True, min_edge_perimeter_px=10000):
    # (Re)create output folder
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
        print(f"Deleted directory: {out_path}")
    os.makedirs(out_path)
    print(f"Created directory: {out_path}")

    image_files = [os.path.join(inp_dir, f) for f in os.listdir(inp_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        raise ValueError("No image files found in the specified folder.")

    for image_file in image_files:
        print("\nProcessing:", image_file)

        # regex-based name extraction (kept)
        img_name_extract = re.search(r'([^\\/]+)(?=\.jpg|jpeg|png$)', image_file, re.IGNORECASE)
        img_name_extract = img_name_extract.group()
        print("img_name (regex):", img_name_extract)
        img_name = os.path.splitext(os.path.basename(image_file))[0]

        image = cv2.imread(image_file)
        if image is None:
            print(f"Warning: Could not read image {image_file}")
            continue

        # 1) robust foreground
        largest_mask = robust_foreground_mask(image)

        # best container contour
        cnt = pick_best_container_contour(largest_mask)
        if cnt is None:
            print("No valid container contour found.")
            continue

        # original pixels only inside mask
        cutout_black = cv2.bitwise_and(image, image, mask=largest_mask)

        # 2) edges (for overlays only)
        smoothed = cv2.GaussianBlur(largest_mask, (5, 5), 0)
        edges = cv2.Canny(smoothed, 50, 150)
        if not draw_boundary_only:
            edges = filter_small_edges(edges, min_edge_perimeter_px)

        # use convex hull of the container contour for 'annotated'
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        eps  = max(2.0, 0.003 * peri)
        curve = cv2.approxPolyDP(hull, eps, True)

        # 3) corners from the actual contour (still from cnt)
        box_pts = corners_from_contour_geometry(cnt)


        # metrics
        x, y, bw, bh = cv2.boundingRect(cnt)
        mask_area = int(cv2.contourArea(cnt))
        w_est = max(1, int(np.linalg.norm(box_pts[1] - box_pts[0])))
        h_est = max(1, int(np.linalg.norm(box_pts[3] - box_pts[0])))
        rot_area = w_est * h_est
        bbox_area = bw * bh
        center_pt = tuple(np.mean(box_pts, axis=0).astype(int))

        #  midpoints of left & right edges (TL-BL) and (TR-BR) ------
        TL, TR, BR, BL = box_pts  # order from corners_from_contour_geometry
        left_mid  = tuple(np.mean([TL, BL], axis=0).astype(int))
        right_mid = tuple(np.mean([TR, BR], axis=0).astype(int))
        print(f"left_mid (TL–BL)  = {left_mid}")
        print(f"right_mid (TR–BR) = {right_mid}")
        # ----------------------------------------------------------------------

        # Two midpoints (25% and 75%) of top & bottom edges (TL-TR) and (BL-BR) ------
        if img_name == 'doors' or img_name == 'front':
            # top_mid_25 = tuple(((1 - 0.25) * TL + 0.25 * TR).astype(int))
            # top_mid_75 = tuple(((1 - 0.75) * TL + 0.75 * TR).astype(int))
            top_mid_25 = tuple(((1 - 0.33) * TL + 0.33 * TR).astype(int))
            top_mid_75 = tuple(((1 - 0.66) * TL + 0.66 * TR).astype(int))
            print(f"top_mid_25 (TL–TR)  = {top_mid_25}")
            print(f"top_mid_75 (TL–TR)  = {top_mid_75}")
            # bottom_mid_25 = tuple(((1 - 0.25) * BL + 0.25 * BR).astype(int))
            # bottom_mid_75 = tuple(((1 - 0.75) * BL + 0.75 * BR).astype(int))
            bottom_mid_25 = tuple(((1 - 0.33) * BL + 0.33 * BR).astype(int))
            bottom_mid_75 = tuple(((1 - 0.66) * BL + 0.66 * BR).astype(int))
            print(f"bottom_mid_25 (BL–BR)  = {bottom_mid_25}")
            print(f"bottom_mid_75 (BL–BR)  = {bottom_mid_75}")

        if img_name == 'right' or img_name == 'left':
            top_mid_25 = tuple(((1 - 0.25) * TL + 0.25 * TR).astype(int))
            top_mid_50 = tuple(((1 - 0.50) * TL + 0.50 * TR).astype(int))
            top_mid_75 = tuple(((1 - 0.75) * TL + 0.75 * TR).astype(int))
            print(f"top_mid_25 (TL–TR)  = {top_mid_25}")
            print(f"top_mid_50 (TL–TR)  = {top_mid_50}")
            print(f"top_mid_75 (TL–TR)  = {top_mid_75}")
            bottom_mid_25 = tuple(((1 - 0.25) * BL + 0.25 * BR).astype(int))
            bottom_mid_50 = tuple(((1 - 0.50) * BL + 0.50 * BR).astype(int))
            bottom_mid_75 = tuple(((1 - 0.75) * BL + 0.75 * BR).astype(int))
            print(f"bottom_mid_25 (BL–BR)  = {bottom_mid_25}")
            print(f"bottom_mid_25 (BL–BR)  = {bottom_mid_50}")
            print(f"bottom_mid_75 (BL–BR)  = {bottom_mid_75}")

        # 4) draw curved outline + center + corner dots (only outer boundary)
        vis = image.copy()
        cv2.drawContours(vis, [curve], -1, (0, 255, 255), 2)      # outer boundary only
        cv2.circle(vis, center_pt, 5, (255, 0, 0), -1)
        cv2.line(vis, left_mid, right_mid, (0, 0, 255), 2)              # draw mid center line ((TL-BL) and (TR-BR))

        cv2.circle(vis, top_mid_25, 5, (0, 0, 255), -1)
        cv2.circle(vis, top_mid_75, 5, (0, 0, 255), -1)

        cv2.circle(vis, bottom_mid_25, 5, (0, 0, 255), -1)
        cv2.circle(vis, bottom_mid_75, 5, (0, 0, 255), -1)

        cv2.line(vis, top_mid_25, bottom_mid_25, (0, 0, 255), 2)
        cv2.line(vis, top_mid_75, bottom_mid_75, (0, 0, 255), 2)

        if img_name == 'right' or img_name == 'left':
            cv2.circle(vis, top_mid_50, 5, (0, 0, 255), -1)
            cv2.circle(vis, bottom_mid_50, 5, (0, 0, 255), -1)
            cv2.line(vis, top_mid_50, bottom_mid_50, (0, 0, 255), 2)

        names = ["TL", "TR", "BR", "BL"]
        corner_pts_int = [tuple(map(int, p)) for p in box_pts]
        for name, p in zip(names, corner_pts_int):
            cv2.circle(vis, p, 7, (255, 0, 0), -1)
            cv2.putText(vis, name, (p[0]+6, p[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

        # draw the midpoints on the annotated image ------
        cv2.circle(vis, left_mid, 7, (255, 0, 0), -1)   # blue
        cv2.putText(vis, "LM", (left_mid[0]+6, left_mid[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(vis, right_mid, 7, (255, 0, 0), -1)  # blue
        cv2.putText(vis, "RM", (right_mid[0]+6, right_mid[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
        # ------------------------------------------------------------

        # 5) perspective crop
        dst = np.array([[0, 0], [w_est-1, 0], [w_est-1, h_est-1], [0, h_est-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(box_pts, dst)
        rectified_crop = cv2.warpPerspective(image, M, (w_est, h_est))
        print('rectified_crop_shape         : ', rectified_crop.shape)

        # 5b) Get TL, TR, BR, BL in rectified (warped) coordinates
        # box_pts: original image corner points
        # M: transform from original → rectified
        rectified_corners = cv2.perspectiveTransform(box_pts.reshape(-1, 1, 2), M).reshape(-1, 2)
        TL_r, TR_r, BR_r, BL_r = rectified_corners
        print(f"Rectified corners (TL_r, TR_r, BR_r, BL_r): {rectified_corners}")
        rectified_corners = np.rint(rectified_corners).astype(int)
        print(f"Rectified corners (TL_r, TR_r, BR_r, BL_r): {rectified_corners}")
        # Calculating center point for rectified_corners
        rect_center = tuple(np.mean(rectified_corners, axis=0).astype(int))
        print("Rectified center:", rect_center)

        # # corners in original image: box_pts (TL, TR, BR, BL)
        # # homography M (original -> rectified), and rectified size (w_est, h_est)
        # rectified_corners = cv2.perspectiveTransform(box_pts.reshape(1, -1, 2), M)[0]
        #
        # # clean up: round, cast to int, and clip to image bounds
        # rectified_corners = np.rint(rectified_corners).astype(int)
        # rectified_corners[:, 0] = np.clip(rectified_corners[:, 0], 0, w_est - 1)
        # rectified_corners[:, 1] = np.clip(rectified_corners[:, 1], 0, h_est - 1)
        # TL_r, TR_r, BR_r, BL_r = rectified_corners
        # print(f"Rectified corners (TL, TR, BR, BL): {rectified_corners}")

        ## Map from rectified to original coords
        # Minv = np.linalg.inv(M)
        # orig_from_rect = cv2.perspectiveTransform(rectified_corners.reshape(1, -1, 2).astype(np.float32), Minv)[0]
        # orig_from_rect = np.rint(orig_from_rect).astype(int)

        # 5C : midpoints of left & right edges (TL-BL) and (TR-BR) Rectified------
        left_mid_r = tuple(np.mean([TL_r, BL_r], axis=0).astype(int))
        right_mid_r = tuple(np.mean([TR_r, BR_r], axis=0).astype(int))
        print(f"left_mid_r (TL–BL)  = {left_mid_r}")
        print(f"right_mid_r (TR–BR) = {right_mid_r}")

        # 5D: Two midpoints (25% and 75%) of top & bottom edges (TL_r-TR_r) and (BL_r-BR_r) ------
        if img_name == 'doors' or img_name == 'front':
            top_mid_25_r = tuple(((1 - 0.33) * TL_r + 0.33 * TR_r).astype(int))
            top_mid_75_r = tuple(((1 - 0.66) * TL_r + 0.66 * TR_r).astype(int))
            print(f"top_mid_25_r (TL_r–TR_r)  = {top_mid_25_r}")
            print(f"top_mid_75_r (TL_r–TR_r)  = {top_mid_75_r}")
            bottom_mid_25_r = tuple(((1 - 0.33) * BL_r + 0.33 * BR_r).astype(int))
            bottom_mid_75_r = tuple(((1 - 0.66) * BL_r + 0.66 * BR_r).astype(int))
            print(f"bottom_mid_25_r (BL_r–BR_r)  = {bottom_mid_25_r}")
            print(f"bottom_mid_75 (BL_r–BR_r)  = {bottom_mid_75_r}")

        if img_name == 'right' or img_name == 'left':
            top_mid_25_r = tuple(((1 - 0.25) * TL_r + 0.25 * TR_r).astype(int))
            top_mid_50_r = tuple(((1 - 0.50) * TL_r + 0.50 * TR_r).astype(int))
            top_mid_75_r = tuple(((1 - 0.75) * TL_r + 0.75 * TR_r).astype(int))
            print(f"top_mid_25_r (TL_r–TR_r)  = {top_mid_25_r}")
            print(f"top_mid_50_r (TL_r–TR_r)  = {top_mid_50_r}")
            print(f"top_mid_75_r (TL_r–TR_r)  = {top_mid_75_r}")
            bottom_mid_25_r = tuple(((1 - 0.25) * BL_r + 0.25 * BR_r).astype(int))
            bottom_mid_50_r = tuple(((1 - 0.50) * BL_r + 0.50 * BR_r).astype(int))
            bottom_mid_75_r = tuple(((1 - 0.75) * BL_r + 0.75 * BR_r).astype(int))
            print(f"bottom_mid_25_r (BL_r–BR_r)  = {bottom_mid_25_r}")
            print(f"bottom_mid_50_r (BL_r–BR_r)  = {bottom_mid_50_r}")
            print(f"bottom_mid_75 (BL_r–BR_r)  = {bottom_mid_75_r}")

        # 6) overlays
        if draw_boundary_only:
            print('Image Name: ', img_name)
            boundary_mask = np.zeros_like(largest_mask)
            cv2.drawContours(boundary_mask, [curve], -1, 255, 2)  # hull-based boundary
            edge_overlay_full = image.copy()
            edge_overlay_full[boundary_mask > 0] = (0, 255, 255)
            crop_edges_overlay = edge_overlay_full[y:y+bh, x:x+bw]
            edges_rect = cv2.warpPerspective(boundary_mask, M, (w_est, h_est))
            rectified_edges_overlay = rectified_crop.copy()
            rectified_edges_overlay[edges_rect > 0] = (0, 255, 255)
            print('vis_shape                    : ', vis.shape)
            print('rectified_edges_overlay_shape: ', rectified_edges_overlay.shape)
            for (name, p) in zip(["TL", "TR", "BR", "BL"], rectified_corners):
                cv2.circle(rectified_edges_overlay, tuple(p), 6, (0, 0, 255), -1)
                cv2.putText(rectified_edges_overlay, name, (p[0] + 6, p[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(rectified_edges_overlay, rect_center, 6, (0, 0, 255), -1)

            cv2.circle(rectified_edges_overlay, top_mid_25_r, 5, (0, 0, 255), -1)
            cv2.circle(rectified_edges_overlay, top_mid_75_r, 5, (0, 0, 255), -1)

            cv2.circle(rectified_edges_overlay, bottom_mid_25_r, 5, (0, 0, 255), -1)
            cv2.circle(rectified_edges_overlay, bottom_mid_75_r, 5, (0, 0, 255), -1)

            if img_name == 'right' or img_name == 'left':
                cv2.circle(rectified_edges_overlay, top_mid_50_r, 5, (0, 0, 255), -1)
                cv2.circle(rectified_edges_overlay, bottom_mid_50_r, 5, (0, 0, 255), -1)

            # draw the midpoints on the annotated image ------
            cv2.circle(rectified_edges_overlay, left_mid_r, 7, (255, 0, 0), -1)  # blue
            cv2.putText(rectified_edges_overlay, "LM", (left_mid_r[0]+6, left_mid_r[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
            print(left_mid_r, (left_mid[0] + 1, left_mid[1] - 1))
            cv2.circle(rectified_edges_overlay, right_mid_r, 7, (255, 0, 0), -1)  # blue
            cv2.putText(rectified_edges_overlay, "RM", (right_mid_r[0]-30, right_mid_r[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
            # ------------------------------------------------------------

        else:
            edge_overlay_full = image.copy()
            edge_overlay_full[edges > 0] = (0, 255, 255)
            crop_edges_overlay = edge_overlay_full[y:y+bh, x:x+bw]
            edges_rect = cv2.warpPerspective(edges, M, (w_est, h_est))
            rectified_edges_overlay = rectified_crop.copy()
            rectified_edges_overlay[edges_rect > 0] = (0, 255, 255)
            print('vis_shape                    : ', vis.shape)
            print('rectified_edges_overlay_shape: ', rectified_edges_overlay.shape)

        # 6B: Drawing Segment and Section Lines
        cv2.line(rectified_edges_overlay, left_mid_r, right_mid_r, (0, 0, 255),2)  # draw mid center line ((TL_r-BL_r) and (TR_r-BR_r))

        if img_name == 'doors' or img_name == 'front':
            cv2.line(rectified_edges_overlay, top_mid_25_r, bottom_mid_25_r, (0, 0, 255), 2)
            cv2.line(rectified_edges_overlay, top_mid_75_r, bottom_mid_75_r, (0, 0, 255), 2)
        if img_name == 'right' or img_name == 'left':
            cv2.line(rectified_edges_overlay, top_mid_25_r, bottom_mid_25_r, (0, 0, 255), 2)
            cv2.line(rectified_edges_overlay, top_mid_50_r, bottom_mid_50_r, (0, 0, 255), 2)
            cv2.line(rectified_edges_overlay, top_mid_75_r, bottom_mid_75_r, (0, 0, 255), 2)

        # rectified_edges_overlay = cv2.resize(rectified_edges_overlay, (720,640))

        # 7) save outputs
        ann_path   = os.path.join(out_path, f"{img_name}_annot.jpg")
        crop_path  = os.path.join(out_path, f"{img_name}_crop_rectified.jpg")
        mask_path  = os.path.join(out_path, f"{img_name}_mask.png")
        edge_path  = os.path.join(out_path, f"{img_name}_edges.png")
        crop_edge_overlay_path   = os.path.join(out_path, f"{img_name}_crop_edges_overlay.jpg")
        rect_edge_overlay_path   = os.path.join(out_path, f"{img_name}_rectified_edges_overlay.jpg")
        edges_rect_path          = os.path.join(out_path, f"{img_name}_edges_rectified.png")
        cutout_black_path        = os.path.join(out_path, f"{img_name}_cutout_black.jpg")

        cv2.imwrite(ann_path, vis)
        cv2.imwrite(crop_path, rectified_crop)
        cv2.imwrite(mask_path, largest_mask)
        cv2.imwrite(edge_path, edges if not draw_boundary_only else boundary_mask)
        cv2.imwrite(crop_edge_overlay_path, crop_edges_overlay)
        cv2.imwrite(rect_edge_overlay_path, rectified_edges_overlay)
        cv2.imwrite(edges_rect_path, edges_rect if not draw_boundary_only else edges_rect)
        cv2.imwrite(cutout_black_path, cutout_black)

        # logs
        print("Corners (TL, TR, BR, BL):", corner_pts_int)
        print(f"rotated_rect_area (est) = {rot_area}")
        print(f"mask_area               = {mask_area}")
        print(f"aabb_area               = {bbox_area}  [x={x}, y={y}, w={bw}, h={bh}]")
        print(f"center                  = {center_pt}")

        if visualize:
            cv2.imshow("mask", largest_mask)
            cv2.imshow("annotated", vis)
            cv2.imshow("crop_edges_overlay", crop_edges_overlay)
            cv2.imshow("rectified_edges_overlay", rectified_edges_overlay)
            # cv2.imshow("cutout_black_path", cutout_black)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    inp_dir = r"C:/Users/shann/Workouts (Python & R)/Python Workouts/ScanBoX_Dir/Damage_Inspect/output/yolov11m_segside5-train1aaaa_out"
    out_path = r"C:/Users/shann/Workouts (Python & R)/Python Workouts/ScanBoX_Dir/Damage_Inspect/demo_out"
    eval_damage_codex(inp_dir, out_path, visualize=True, draw_boundary_only=True)
