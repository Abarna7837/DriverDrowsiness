import pygame
from pathlib import Path


ALARM_PATH = Path("assets") / "alarm.wav"


def _init_mixer():
    try:
        pygame.mixer.init(frequency=44100, size=-16, channels=2)
    except Exception:
        try:
            pygame.mixer.init()
        except Exception:
            return False
    return True


_MIXER_READY = _init_mixer()


def play_alarm():
    if not _MIXER_READY:
        return

    if not ALARM_PATH.exists():
        return

    try:
        sound = pygame.mixer.Sound(str(ALARM_PATH))
        sound.play()
    except Exception:
        return