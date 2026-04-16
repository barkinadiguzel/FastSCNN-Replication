class CityscapesConfig:
    NUM_CLASSES = 19
    IGNORE_INDEX = 255

    INPUT_SIZE = (1024, 2048)

    MEAN = (0.485, 0.456, 0.406)
    STD  = (0.229, 0.224, 0.225)

    OUTPUT_STRIDE = 8
