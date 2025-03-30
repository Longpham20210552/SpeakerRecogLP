# Wave path
TRAIN_WAV_DIR = '/home/admin/Desktop/read_25h_2/train'
DEV_WAV_DIR = '/home/admin/Desktop/read_25h_2/dev'
TEST_WAV_DIR = 'test_wavs'

# Feature path
TRAIN_FEAT_DIR = 'TIMIT_UNKNOWN/TRAIN'
VALID_FEAT_DIR = 'TIMIT_UNKNOWN/VALID'
TEST_FEAT_DIR = 'vietnam_mfcc2'
OUT_FEAT_DIR = 'TIMIT_UNKNOWN/unknown(valid)'
# Context window size
NUM_WIN_SIZE = 400 #10

# Settings for feature extraction
USE_LOGSCALE = True
USE_DELTA = False
USE_SCALE = False
SAMPLE_RATE = 16000
FILTER_BANK = 60
NUM_PREVIOUS_FRAME = 99
NUM_NEXT_FRAME = 100