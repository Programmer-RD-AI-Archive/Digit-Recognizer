from Model import *


# Transformations
class PreProccessing:
    def __init__(
        self,
        random_vertical_flip: bool = False,
        color_jitter: bool = False,
        random_grayscale: bool = False,
        random_horizontal_flip: bool = False,
        random_rotation: bool = False,
    ) -> None:
        self.color_jitter = color_jitter
        self.random_grayscale = random_grayscale
        self.random_horizontal_flip = random_horizontal_flip
        self.random_rotation = random_rotation
        self.random_vertical_flip = random_vertical_flip
        self.compose_list = []

    def color_jitter_pp(self):
        self.compose_list.append(ColorJitter(0.125, 0.125, 0.125, 0.125))

    def random_grayscale_pp(self):
        self.compose_list.append(RandomGrayscale())

    def random_horizontal_flip_pp(self):
        self.compose_list.append(RandomHorizontalFlip())

    def random_rotation_pp(self):
        self.compose_list.append(RandomRotation(180))

    def random_vertical_flip_pp(self):
        self.compose_list.append(RandomVerticalFlip())

    def preproccess(self, img):
        if self.color_jitter:
            self.color_jitter_pp()
        if self.random_grayscale:
            self.random_grayscale_pp()
        if self.random_horizontal_flip:
            self.random_horizontal_flip_pp()
        if self.random_rotation:
            self.random_rotation_pp()
        if self.random_vertical_flip:
            self.random_vertical_flip_pp()
        try:
            transformation = Compose(self.compose_list)
            img = np.array(transformation(Image.fromarray(img)))
        except:
            pass
        img = img / 255.0
        return img


# for _ in range(50):
#     pp = PreProccessing()
#     plt.figure(figsize=(10, 6))
#     plt.imshow(pp.preproccess(np.array(cv2.imread("./tests/images.jpeg"))))
#     plt.savefig("./test.png")
#     plt.close()
#     time.sleep(2.5)
# print(np.array(cv2.imread("./tests/images.jpeg")).dtype)
# print(type(cv2.imread("./tests/images.jpeg")))
