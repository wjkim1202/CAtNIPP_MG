BATCH_SIZE = 256 #8
INPUT_DIM = 4
EMBEDDING_DIM = 128
SAMPLE_SIZE = 200
K_SIZE = 20

SAMPLE_LENGTH = 0.2

ADAPTIVE_AREA = True
ADAPTIVE_TH = 0.4

USE_GPU = True
USE_GPU_GLOBAL = True
CUDA_DEVICE = [0] #[0, 1, 2, 3]
NUM_GPU = 1
NUM_META_AGENT = 6 #6  32
LR = 1e-4

################################################################
GAMMA = 0.9
BUDGET_RANGE = (8, 10)  # (6, 8)   (2, 4)  (4, 6)  (6, 8) (8, 10)
MULTI_GAMMA = [0.6, 0.8 , 0.9, 0.99] # [0.2, 0.6, 0.8, 0.9]   # [0, 0.2, 0.6, 0.8 , 0.9, 0.99]   #[0, 0.1, 0.5, 0.9, 0.99]  # [0.1, 0.5, 0.9, 0.99] # [0.1, 0.5, 0.9, 0.99]  None
if MULTI_GAMMA is not None:
    GAMMA = MULTI_GAMMA[-1]
SCALE_GAMMA = False
FIT_GAMMA = False


RANDOM_GAMMA = True
SPECIFIC_GAMMA = 0
SEED = 100

RESET_EVERY_EPISODE = False

GAMMANET= False

################################################################
DECAY_STEP = 32
SUMMARY_WINDOW = 8
FOLDER_NAME = 'ipp'
FOLDER_NAME = FOLDER_NAME + str(GAMMA) + '_bud' + str(BUDGET_RANGE[0])
if MULTI_GAMMA is not None:
    FOLDER_NAME = FOLDER_NAME + '_MG' + str(len(MULTI_GAMMA))
    if SCALE_GAMMA:
        FOLDER_NAME = FOLDER_NAME + '_SG'
elif GAMMANET:
    FOLDER_NAME = FOLDER_NAME + '_GN'

if FIT_GAMMA:
    FOLDER_NAME = FOLDER_NAME + '_FG'
elif RANDOM_GAMMA:
    FOLDER_NAME = FOLDER_NAME + '_RG'
elif SPECIFIC_GAMMA:
    FOLDER_NAME = FOLDER_NAME + '_SG' + str(SPECIFIC_GAMMA)


# if RESET_EVERY_EPISODE:
#     FOLDER_NAME = FOLDER_NAME + '_REE'

FOLDER_NAME = FOLDER_NAME + '_S' + str(SEED)

model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'
LOAD_MODEL = False
SAVE_IMG_GAP = 100000000
