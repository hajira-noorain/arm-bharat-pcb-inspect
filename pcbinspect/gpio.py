"""Andon-style stack light + buzzer on the Pi's GPIO header (gpiozero), with a mock elsewhere.

Wiring (BCM numbering, each LED through a 330 ohm resistor to GND; active buzzer direct):
  GPIO17 green   = PASS
  GPIO27 amber   = NEEDS REVIEW
  GPIO22 red     = FAIL
  GPIO23 buzzer  = short beep on REVIEW, long beep on FAIL
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)
PINS = {"green": 17, "amber": 27, "red": 22, "buzzer": 23}


class StackLight:
    def __init__(self, enabled: bool = True):
        self.mock = True
        self.state = "off"
        if not enabled:
            return
        try:
            from gpiozero import LED, Buzzer
            self.leds = {k: LED(PINS[k]) for k in ("green", "amber", "red")}
            self.buzzer = Buzzer(PINS["buzzer"])
            self.mock = False
        except Exception as exc:  # not a Pi, or gpiozero missing
            log.info("GPIO unavailable (%s): using mock stack light", exc)

    def show(self, verdict: str) -> None:
        colour = {"PASS": "green", "REVIEW": "amber", "FAIL": "red"}.get(verdict, "off")
        self.state = colour
        if self.mock:
            log.info("stack light: %s", colour)
            return
        for k, led in self.leds.items():
            led.on() if k == colour else led.off()
        if verdict in ("REVIEW", "FAIL"):
            secs = 0.15 if verdict == "REVIEW" else 0.8
            threading.Thread(target=self.buzzer.beep, kwargs={"on_time": secs, "off_time": 0.1,
                                                               "n": 1, "background": False},
                             daemon=True).start()

    def off(self) -> None:
        self.state = "off"
        if not self.mock:
            for led in self.leds.values():
                led.off()
            self.buzzer.off()
