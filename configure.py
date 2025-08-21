# Wave path
TRAIN_WAV_DIR = '/home/admin/Desktop/read_25h_2/train'
DEV_WAV_DIR = '/home/admin/Desktop/read_25h_2/dev'
TEST_WAV_DIR = 'test_wavs'

# Feature path
TRAIN_FEAT_DIR = 'LP_clean_unknown_split1/TRAIN'
VALID_FEAT_DIR = 'LP_clean_unknown_split1/VALID'
TEST_FEAT_DIR = 'LP_clean_unknown_split1/TEST'
OUT_FEAT_DIR = 'LP_clean_unknown_split1/unknown(valid)'
TEST_FEAT_DIR_ENROLL = 'LP_test'
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