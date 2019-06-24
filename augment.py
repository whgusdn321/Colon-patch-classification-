import imgaug #https://github.com/aleju/imgaug
from imgaug import augmenters as iaa
import imgaug as ia
ia.seed(100)
#data_size = [3150,3150]
input_size = [2048,2048]
def train_augementors():
    shape_augs = [

        # iaa.Affine(
        #     # scale images to 80-120% of their size, individually per axis
        #     scale={"x": (0.8, 1.2),
        #            "y": (0.8, 1.2)},
        #     # translate by -A to +A percent (per axis)
        #     translate_percent={"x": (-0.01, 0.01),
        #                        "y": (-0.01, 0.01)},
        #     rotate=(-179, 179),  # rotate by -179 to +179 degrees
        #     shear=(-5, 5),  # shear by -5 to +5 degrees
        #     order=[0],  # use nearest neighbour
        #     backend='cv2'  # opencv for fast processing
        # ),
        # iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        # iaa.Flipud(0.2),  # vertically flip 20% of all images
        # iaa.CropToFixedSize(input_size[0],
        #                     input_size[1],
        #                     position='center',
        #                     deterministic=True)
        # iaa.Affine(
        #     # scale images to 80-120% of their size, individually per axis
        #     scale={"x": (0.9, 1.1),
        #            "y": (0.9, 1.1)},
        #     # translate by -A to +A percent (per axis)
        #     order=[0],  # use nearest neighbour
        #     backend='cv2'  # opencv for fast processing
        # ),
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Flipud(0.2),  # vertically flip 20% of all images

    ]
    #
    input_augs = [
        iaa.OneOf([
            iaa.GaussianBlur((0, 3.0)),  # gaussian blur with random sigma
            iaa.MedianBlur(k=(3, 5)),  # median with random kernel siz;lk;l/es
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        ]),
        iaa.Sequential([
            iaa.Add((-26, 26)),
            iaa.AddToHueAndSaturation((-20, 20)),
            iaa.LinearContrast((0.75, 1.25), per_channel=1.0),
        ], random_order=True),
    ]
    return shape_augs, input_augs

def infer_augmentors():
    shape_augs = [
        iaa.CropToFixedSize(input_size[0],
                            input_size[1],
                            position='center',
                            deterministic=True)
    ]
    return shape_augs