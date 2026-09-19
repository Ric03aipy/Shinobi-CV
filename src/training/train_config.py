from shinobi_cv.config import DATA_PATH

DATA_FILE = DATA_PATH / "signs.csv" # 126 featrues (21 points x 3 dimensions x 2 hands) + 1 (class)
BLIND_DATA_FILE = DATA_PATH / "blind_test.csv" # Data collected in a separate session, though same person, same environment, different light and slighly more motion
