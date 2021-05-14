import os
from datetime import datetime

# Setup the summary writer.

ROOT_LOG_DIR = "logs"
RUN_ID = datetime.now().strftime("%Y-%m-%d-%H:%M%Z")
RUN_DIR = os.path.join(ROOT_LOG_DIR, RUN_ID)
