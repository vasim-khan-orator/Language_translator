import time


class SilenceManager:

    def __init__(self, silence_threshold=1.0):

        self.silence_threshold = silence_threshold

        self.last_activity_time = time.time()


    # -----------------------------
    # UPDATE ACTIVITY
    # -----------------------------
    def update_activity(self):

        self.last_activity_time = time.time()


    # -----------------------------
    # CHECK SILENCE
    # -----------------------------
    def is_silence_detected(self):

        current_time = time.time()

        silence_duration = (
            current_time - self.last_activity_time
        )

        return silence_duration >= self.silence_threshold


    # -----------------------------
    # RESET TIMER
    # -----------------------------
    def reset(self):

        self.last_activity_time = time.time()