import itertools as it


class Config:
    WORKER_THREADS = 8

    LEARNING_RATE = 1e-4
    DECAY_RATE = 0.997
    GAMMA = 0.95
    N_STEP_RETURN = 4
    FRAME_REPEAT = 8
    BATCH_SIZE = 64
    QUEUE_LEN = 1000

    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 128
    IMAGE_CHANNELS = 3
    IMAGE_CROP_BOX = (0, 130, 640, 400)
    IMAGE_GRAY_LEVEL = 8

    # buttons: TURN_LEFT
    #          TURN_RIGHT
    # 		   MOVE_FORWARD
    AVAILABLE_ACTION_BUTTONS = 3
    ACTION_SPACE = [list(a) for a in it.product([0, 1], repeat=AVAILABLE_ACTION_BUTTONS)
                    if a[0] != 1 or a[1] != 1]
    N_ACTIONS = len(ACTION_SPACE)
