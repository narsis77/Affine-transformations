import cv2
import numpy as np
import math


def apply_affine_transform(image, choice):
    height, width = image.shape[:2]

    if choice == 1:  # Identity Transformation
        affine_matrix = np.float32([[1, 0, 0], [0, 1, 0]])
    elif choice == 2:  # Scaling Transformation
        scale_factor = 2.0  # Change this as needed
        affine_matrix = np.float32([[scale_factor, 0, 0], [0, scale_factor, 0]])
    elif choice == 3:  # Rotation Transformation
        angle = -45  # Change this as needed
        theta = math.radians(angle)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        affine_matrix = np.float32([[cos_theta, sin_theta, 0], [-sin_theta, cos_theta, 0]])
    elif choice == 4:  # Translation Transformation
        tx, ty = 50, 30  # Change this as needed
        affine_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    elif choice == 5:  # Shear (Vertical) Transformation
        shear_factor = 0.5  # Change this as needed
        affine_matrix = np.float32([[1, 0, 0], [shear_factor, 1, 0]])
    elif choice == 6:  # Shear (Horizontal) Transformation
        shear_factor = -0.5  # Change this as needed
        affine_matrix = np.float32([[1, shear_factor, 0], [0, 1, 0]])

    else:
        print("Invalid choice!")
        return None

    if choice == 2 or choice == 4:
        transformed_image = cv2.warpAffine(image, affine_matrix, (width, height))
        return transformed_image
    else:
        # Apply transformation to find the bounding box of the transformed image
        corners = np.array([[0, 0], [0, height], [width, 0], [width, height]])
        transformed_corners = cv2.transform(corners.reshape(-1, 1, 2), affine_matrix)
        min_x = np.min(transformed_corners[:, 0, 0])
        max_x = np.max(transformed_corners[:, 0, 0])
        min_y = np.min(transformed_corners[:, 0, 1])
        max_y = np.max(transformed_corners[:, 0, 1])

        # Adjust the transformation matrix to ensure the entire original image is preserved
        affine_matrix[0, 2] += -min_x
        affine_matrix[1, 2] += -min_y

        transformed_image = cv2.warpAffine(image, affine_matrix, (int(max_x - min_x), int(max_y - min_y)))
        return transformed_image


def main():
    image_path = 'p.jpg'  # Replace with your image path
    image = cv2.imread(image_path)

    while True:
        print("\nChoose an affine transformation:")
        print("1. Identity")
        print("2. Scaling")
        print("3. Rotation")
        print("4. Translation")
        print("5. Shear (Vertical)")
        print("6. Shear (Horizontal)")
        print("0. Exit")

        choice = int(input("Enter your choice: "))
        if choice == 0:
            break

        transformed_image = apply_affine_transform(image, choice)
        if transformed_image is not None:
            cv2.imshow("Transformed Image", transformed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
