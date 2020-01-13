# References:
# [1] Introduction to SURF (Speeded-Up Robust Features)
#     https://docs.opencv.org/trunk/df/dd2/tutorial_py_surf_intro.html
# [2] Feature Matching + Homography to find Objects
#     https://docs.opencv.org/trunk/d1/de0/tutorial_py_feature_homography.html

import os
import cv2
import numpy as np
    
def process_matching_images(hr_img_path, lr_img_path, path_to_save, fast_matching=True, process_as_grid=True, index=0, focal_length_lr=16, focal_length_hr=50):
    # Constans:
    # Sample sizes
    LR_SAMPLE_SIZE = 64
    HR_SAMPLE_SIZE = 128
    
    # Matching factors:
    # The minimum number of matches to pass the image pair
    MIN_MATCH_COUNT = 4
    # The lower the number the more matching keypoints would be found
    SURF_REDUCED_THRESHOLD = 5000
    # The Distance matching ratio
    MATCHING_RATIO = 0.7
    # Maximum allowed reprojection error to treat a point pair as an inlier (used in the RANSAC method only)
    RANSAC_REPROJECTION_THERESHOLD = 5.0

    
    # Read and convert to grayscale
    hr_img = cv2.imread(hr_img_path)
    lr_img = cv2.imread(lr_img_path)
    gray_hr = cv2.cvtColor(hr_img, cv2.COLOR_BGR2GRAY)
    gray_lr = cv2.cvtColor(lr_img, cv2.COLOR_BGR2GRAY)
    
    # Create SURF object
    surf = cv2.xfeatures2d.SURF_create()
    
    # If fast matching is enabled apply SURF optimization
    if(fast_matching):
        surf.setHessianThreshold(SURF_REDUCED_THRESHOLD)
        surf.setUpright(True)

    # Detect keypoints and compute keypointer descriptors
    kpts_hr, descs_hr = surf.detectAndCompute(gray_hr, None)
    kpts_lr, descs_lr = surf.detectAndCompute(gray_lr, None)
    
    # Create flann matcher.
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    matcher = cv2.FlannBasedMatcher(index_params, {})

    # Finds the k best matches for each descriptor from a query set. 2 means values to unpack.
    matches = matcher.knnMatch(descs_hr, descs_lr, 2)
    
    # Sort by their distance.
    matches = sorted(matches, key=lambda x:x[0].distance)

    # Ratio test, to get good matches.
    good = [m1 for (m1, m2) in matches if m1.distance < MATCHING_RATIO * m2.distance]

    # Check matches
    if len(good)>MIN_MATCH_COUNT:
        # queryIndex for the small object, trainIndex for the scene
        src_pts = np.float32([ kpts_hr[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kpts_lr[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        
        # Find homography matrix in cv2.RANSAC using good match points
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJECTION_THERESHOLD)

        # Transform image according to perspective
        h, w = hr_img.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)
        perspectiveM = cv2.getPerspectiveTransform(np.float32(dst),pts)
        found = cv2.warpPerspective(lr_img, perspectiveM, (w, h), cv2.INTER_NEAREST)

        # Scenario 1: Process matched images as grid of smaller images
        if(process_as_grid):
            # Calculate matching area original size
            aspect_ratio = hr_img.shape[1]/hr_img.shape[0]
            original_width = int(np.round(dst[2, 0, 0] - dst[1, 0, 0]))
            original_height = int(np.round((dst[2, 0, 0] - dst[1, 0, 0]) / aspect_ratio))

            # Resize upscaled in cv2.warpPerspective image to original size
            aligned_image = cv2.resize(found, (original_width, original_height), cv2.INTER_NEAREST)

            # Crop and prepare images for downsapmling
            aligned_image = aligned_image[(original_height % LR_SAMPLE_SIZE):original_height, 0:(original_width - (original_width % LR_SAMPLE_SIZE))]
            #hr_sample = int(np.floor(LR_SAMPLE_SIZE * hr_img.shape[1] / original_width))
            hr_sample = int(np.round(LR_SAMPLE_SIZE * hr_img.shape[1] / original_width))
            hr_img = hr_img[(hr_img.shape[0] % hr_sample):hr_img.shape[0], 0:(hr_img.shape[1] - (hr_img.shape[1] % hr_sample))]
            
            # Crop and save images by sample grid
            grid_height = int(aligned_image.shape[0] / LR_SAMPLE_SIZE)
            grid_width = int(aligned_image.shape[1] / LR_SAMPLE_SIZE)

            # Create directories for saving, if missing
            test_dir = f'{path_to_save}test_{focal_length_lr}_{focal_length_hr}/'
            train_dir = f'{path_to_save}train_{focal_length_lr}_{focal_length_hr}/'
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)
            if not os.path.exists(train_dir):
                os.makedirs(train_dir)

            for i in range(grid_height):
                for ii in range(grid_width):
                    img_hr = hr_img[(i * hr_sample):((i + 1) * hr_sample),(ii * hr_sample):((ii + 1) * hr_sample)]
                    img_lr = aligned_image[(i * LR_SAMPLE_SIZE):((i + 1) * LR_SAMPLE_SIZE),(ii * LR_SAMPLE_SIZE):((ii + 1) * LR_SAMPLE_SIZE)]

                    if (img_hr.shape[0] != 0 and img_hr.shape[1] != 0):
                        cv2.imwrite(f"{test_dir}{index}_{i}_{ii}.png", cv2.resize(img_hr, (HR_SAMPLE_SIZE,HR_SAMPLE_SIZE), cv2.INTER_NEAREST))
                        cv2.imwrite(f"{train_dir}{index}_{i}_{ii}.png", img_lr)
            
            return int(grid_height * grid_width)
        
        # Scenario 2: Create an image pair for match visualization (for thesis purposes)
        else:
             # Print the found features parameters to console
            print(f"Size of found SURF image descriptor: {surf.descriptorSize()}")
            print(f"Number of found HR image keypoints: {len(kpts_hr)}")
            print(f"Number of found LR image keypoints: {len(kpts_lr)}")
            print(f"Number of matched keypoints: {len(good)}")	
        
            canvas = lr_img.copy()
            keypoints_hr = hr_img.copy()
            keypoints_lr = lr_img.copy()

            # Draw keypoints on HR and LR images
            cv2.drawKeypoints(gray_hr, kpts_hr, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, outImage=keypoints_hr)
            cv2.drawKeypoints(gray_lr, kpts_lr, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, outImage=keypoints_lr)
            
            # Draw polylines on LR image to visualize the HR image matching area
            cv2.polylines(canvas, [np.int32(dst)], True, (0,255,0), 3, cv2.LINE_AA)

            # Draw matches between found good keypoints
            matched = cv2.drawMatches(hr_img, kpts_hr, canvas, kpts_lr, good, None)
            
            # Save images
            cv2.imwrite(path_to_save + "surf_keypoints_hr.jpg", keypoints_hr)
            cv2.imwrite(path_to_save + "surf_keypoints_lr.jpg", keypoints_lr)
            cv2.imwrite(path_to_save + "found.png", found)
            cv2.imwrite(path_to_save + "matched.png", matched)
            
            return 1
    else:
        print(f"Not enough matches are found. Found: {len(good)}. Required: {MIN_MATCH_COUNT}.")
        
        return 0
        
