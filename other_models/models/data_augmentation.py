import PIL
import numpy as np
from PIL import Image

from evoline.img.processor import value_to_class
from evoline.img.rotate_crop import Point, getBounds, getRotatedRectanglePoints


class ImageTooSmall(Exception):

    def __init__(self, max_shape, generated_shape) -> None:
        super().__init__(
            f'Max shape expected {max_shape}, got {generated_shape}, change your image center or lower the padding')


class CenteredCropGenerator:

    def __init__(self, padding,
                 img_shape,
                 patch_size=(16, 16),
                 road_threshold=0.25,
                 fill_mode='reflect',
                 color_change=0.0,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rot90=False,
                 rot_angle=0) -> None:
        super().__init__()

        assert (padding >= 0)
        self.padding = padding
        self.patch_size = patch_size
        self.img_shape = img_shape
        assert (all(img > patch for img, patch in zip(img_shape, patch_size)))
        self.foreground_threshold = road_threshold
        self.fill_mode = fill_mode
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rot90 = rot90
        self.color_change = color_change
        self.rot_angle = np.clip(rot_angle, 0, 89)  # otherwise redundant with other parameters

        self.proba_threshold = 0.5

    def get_generated_shape(self):

        return self.padding * 2 + self.patch_size[0], self.padding * 2 + self.patch_size[1], 3

    def pad_patch(self, patch):
        if self.fill_mode:
            padding = (self.padding, self.padding)
            if len(patch.shape) < 3:
                return np.lib.pad(patch, (padding, padding), self.fill_mode)
            else:
                return np.lib.pad(patch, (padding, padding, (0, 0)), self.fill_mode)
        else:
            return patch

    def create_patches_with_padding(self, img):
        list_patches = []
        imgwidth = img.shape[1]
        imgheight = img.shape[0]
        for i in range(self.padding, imgheight + self.padding, self.patch_size[0] + 2 * self.padding):
            for j in range(self.padding, imgwidth + self.padding, self.patch_size[1] + 2 * self.padding):
                list_patches.append(img[i - self.padding: i + self.patch_size[0] + self.padding,
                                    j - self.padding: j + self.patch_size[1] + self.padding])
        return list_patches

    def generate_crop(self, img, gt_img, center_x, center_y):
        # shape = img.shape
        # assert(shape == self.img_shape)
        generated_shape = self.get_generated_shape()
        for max_s, generated in zip(self.img_shape, generated_shape):
            if generated > max_s:
                raise ImageTooSmall(self.img_shape, generated_shape)

        # Sample a random window from the image
        # center = np.random.randint(self.padding, shape[0] - window_size // 2, 2)
        sub_image = img[
                    center_x - self.padding - self.patch_size[0] // 2:center_x + self.padding + self.patch_size[0] // 2,
                    center_y - self.padding - self.patch_size[1] // 2:center_y + self.padding + self.patch_size[1] // 2]
        gt_sub_image = gt_img[center_x - self.patch_size[0] // 2:center_x + self.patch_size[0] // 2,
                       center_y - self.patch_size[1] // 2:center_y + self.patch_size[1] // 2]

        # The label does not depend on the image rotation/flip (provided that the rotation is in steps of 90°)
        label = value_to_class(np.mean(gt_sub_image))

        # Image augmentation
        # Random flip
        if self.vertical_flip and np.random.rand() < self.proba_threshold:
            # Flip vertically
            sub_image = np.flipud(sub_image)
        if self.horizontal_flip and np.random.rand() < self.proba_threshold:
            # Flip horizontally
            sub_image = np.fliplr(sub_image)

        if self.rot90:
            # random 90 degree rotation
            sub_image = np.rot90(sub_image, np.random.choice(4))

        if self.color_change and np.random.rand() < self.proba_threshold:
            sub_image = np.clip(sub_image * (1 + np.random.uniform(-self.color_change, self.color_change)), 0, 1)

        return sub_image, label

    def generate_batches(self, original_imgs, gt_labels, batch_size):

        new_imgs = np.empty((original_imgs.shape[0],
                             original_imgs.shape[1] + 2 * self.padding,
                             original_imgs.shape[2] + 2 * self.padding,
                             original_imgs.shape[3]))

        new_gts = np.empty((gt_labels.shape[0],
                            gt_labels.shape[1] + 2 * self.padding,
                            gt_labels.shape[2] + 2 * self.padding))

        for i in range(original_imgs.shape[0]):
            new_imgs[i] = self.pad_patch(original_imgs[i])
            new_gts[i] = self.pad_patch(gt_labels[i])

        original_imgs = new_imgs
        gt_labels = new_gts
        generated_shape = self.get_generated_shape()

        while 1:
            images_batch = np.empty((batch_size, *generated_shape))
            labels_batch = np.empty((batch_size, 2))

            for i in range(batch_size):
                img_idx = np.random.choice(original_imgs.shape[0])
                img = original_imgs[img_idx]
                gt = gt_labels[img_idx]
                if self.rot_angle and np.random.rand() < self.proba_threshold:
                    while 1:
                        try:
                            angle = np.random.randint(1, self.rot_angle)
                            base = Point(0, 0)

                            points = getRotatedRectanglePoints(np.pi * (angle / 180), base,
                                                               self.padding * 2 + self.patch_size[0],
                                                               self.padding * 2 + self.patch_size[1])
                            limits = getBounds(points)
                            width = np.random.randint(0, self.img_shape[1] + 2 * self.padding - limits.right)
                            height = np.random.randint(np.abs(limits.upper),
                                                       self.img_shape[0] + 2 * self.padding - limits.lower)

                            bounds = getBounds(getRotatedRectanglePoints(np.pi * (angle / 180), Point(width, height),
                                                                         self.padding * 2 + self.patch_size[0],
                                                                         self.padding * 2 + self.patch_size[1]))

                            boundary_image = img[int(round(bounds.left)): int(round(bounds.right)),
                                             int(round(bounds.upper)):int(round(bounds.lower)),
                                             :]
                            rotated_boundary = np.asarray(
                                PIL.Image.fromarray(np.uint8(boundary_image * 255)).rotate(angle,
                                                                                           resample=Image.BICUBIC)).astype(
                                'float32') / 255.0

                            boundary_gt = gt[int(round(bounds.left)): int(round(bounds.right)),
                                          int(round(bounds.upper)):int(round(bounds.lower))]

                            rotated_boundary_gt = np.asarray(
                                PIL.Image.fromarray(np.uint8(boundary_gt * 255)).rotate(angle)).astype(
                                'float32') / 255.0

                            center_x = rotated_boundary.shape[0] // 2
                            center_y = center_x
                            generated_image, label = self.generate_crop(rotated_boundary, rotated_boundary_gt, center_x,
                                                                        center_y)
                            break

                        except:
                            pass

                else:

                    center_x = np.random.randint(self.padding + self.patch_size[0] // 2,
                                                 self.img_shape[0] + self.padding - self.patch_size[0] // 2, 1)[0]
                    center_y = np.random.randint(self.padding + self.patch_size[1] // 2,
                                                 self.img_shape[1] + self.padding - self.patch_size[1] // 2, 1)[0]
                    generated_image, label = self.generate_crop(img, gt, center_x, center_y)

                images_batch[i] = generated_image
                # already one hot-encoded
                labels_batch[i] = label

            yield (images_batch, labels_batch)


class ImageGeneratorLabelOneToOne:
    def __init__(self,
                 img_shape,
                 width,
                 height,
                 road_threshold=0.5,
                 color_change=0.0,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rot90=False,
                 rot_angle=0) -> None:
        super().__init__()

        self.img_shape = img_shape
        self.width = width
        self.height = height
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rot90 = rot90
        self.color_change = color_change
        self.rot_angle = np.clip(rot_angle, 0, 89)  # otherwise redundant with other parameters
        self.road_threshold = road_threshold
        self.proba_threshold = 0.5

    def get_generated_shape(self):

        return self.width, self.height, 3

    def generate_crop(self, img, gt_img, base_x, base_y):
        # shape = img.shape
        # assert(shape == self.img_shape)
        generated_shape = self.get_generated_shape()
        for max_s, generated in zip(self.img_shape, generated_shape):
            if generated > max_s:
                raise ImageTooSmall(self.img_shape, generated_shape)

        # Sample a random window from the image
        # base = np.random.randint(self.padding, shape[0] - window_size // 2, 2)
        sub_image = img[
                    base_x:base_x + self.height,
                    base_y:base_y + self.width]
        gt_sub_image = gt_img[
                       base_x:base_x + self.height,
                       base_y:base_y + self.width]

        # The label does not depend on the image rotation/flip (provided that the rotation is in steps of 90°)

        # Image augmentation
        # Random flip
        if self.vertical_flip and np.random.rand() < self.proba_threshold:
            # Flip vertically
            sub_image = np.flipud(sub_image)
            gt_sub_image = np.flipud(gt_sub_image)

        if self.horizontal_flip and np.random.rand() < self.proba_threshold:
            # Flip horizontally
            sub_image = np.fliplr(sub_image)
            gt_sub_image = np.fliplr(gt_sub_image)

        if self.rot90:
            # random 90 degree rotation
            rotation = np.random.choice(4)
            sub_image = np.rot90(sub_image, rotation)
            gt_sub_image = np.rot90(gt_sub_image, rotation)

        if self.color_change and np.random.rand() < self.proba_threshold:
            sub_image = np.clip(sub_image * (1 + np.random.uniform(-self.color_change, self.color_change)), 0, 1)

        return sub_image, gt_sub_image

    def generate_batches(self, original_imgs, gt_labels, batch_size):

        generated_shape = self.get_generated_shape()

        while 1:
            images_batch = np.empty((batch_size, *generated_shape))
            labels_batch = np.empty((batch_size, generated_shape[0], generated_shape[1], 1))

            for i in range(batch_size):
                img_idx = np.random.choice(original_imgs.shape[0])
                img = original_imgs[img_idx]
                gt = gt_labels[img_idx]
                if self.rot_angle and np.random.rand() < self.proba_threshold:
                    while 1:
                        try:
                            angle = np.random.randint(1, self.rot_angle)
                            base = Point(0, 0)

                            points = getRotatedRectanglePoints(np.pi * (angle / 180), base,
                                                               self.height,
                                                               self.width)
                            limits = getBounds(points)
                            width = np.random.randint(0, self.img_shape[1] - limits.right)
                            height = np.random.randint(np.abs(limits.upper),
                                                       self.img_shape[0] - limits.lower)

                            bounds = getBounds(getRotatedRectanglePoints(np.pi * (angle / 180), Point(width, height),
                                                                         self.height,
                                                                         self.width))

                            boundary_image = img[int(round(bounds.left)): int(round(bounds.right)),
                                             int(round(bounds.upper)):int(round(bounds.lower)),
                                             :]
                            rotated_boundary = np.asarray(
                                PIL.Image.fromarray(np.uint8(boundary_image * 255)).rotate(angle,
                                                                                           resample=Image.BICUBIC)).astype(
                                'float32') / 255.0

                            boundary_gt = gt[int(round(bounds.left)): int(round(bounds.right)),
                                          int(round(bounds.upper)):int(round(bounds.lower))]

                            rotated_boundary_gt = np.asarray(
                                PIL.Image.fromarray(np.uint8(boundary_gt * 255)).rotate(angle)).astype(
                                'float32') / 255.0

                            base_x = rotated_boundary.shape[0] // 2 - self.height//2
                            base_y = rotated_boundary.shape[1] // 2 - self.width//2
                            generated_image, gt_crop = self.generate_crop(rotated_boundary, rotated_boundary_gt,
                                                                            base_x,
                                                                            base_y)
                            break

                        except Exception as e:
                            # print(e)
                            pass


                else:

                    base_x = np.random.randint(0, self.img_shape[0] - self.height, 1)[0]
                    base_y = np.random.randint(0, self.img_shape[1] - self.width, 1)[0]
                    generated_image, gt_crop = self.generate_crop(img, gt, base_x, base_y)

                images_batch[i] = generated_image
                labels_batch[i] = np.expand_dims((gt_crop > self.road_threshold) * 1, axis=2)

            yield (images_batch, labels_batch.astype(np.uint8))
