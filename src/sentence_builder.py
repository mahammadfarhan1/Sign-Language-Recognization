"""Turn a stream of per-frame letter predictions into words and sentences.

Fingerspelling has no built-in "letter boundary", so the rules are:
  * A letter is typed once the same letter has been shown continuously for
    `hold_seconds`.
  * To type the same letter twice in a row (e.g. the "LL" in HELLO), drop your
    hand out of view (or change sign) between the two.
  * Keeping the hand out of view for `space_after_seconds` types a space,
    which ends the current word.
"""


class SentenceBuilder:
    def __init__(self, hold_seconds: float = 1.0, space_after_seconds: float = 2.0):
        self.hold_seconds = hold_seconds
        self.space_after_seconds = space_after_seconds
        self.text = ""
        self._candidate: str | None = None
        self._candidate_since = 0.0
        self._committed = False
        self._no_hand_since: float | None = None

    def update(self, letter: str | None, now: float) -> float:
        """Feed the latest smoothed letter (None = no hand) and the current time.

        Returns the hold progress of the current letter in [0, 1].
        """
        if letter is None:
            if self._no_hand_since is None:
                self._no_hand_since = now
            elif (
                now - self._no_hand_since >= self.space_after_seconds
                and self.text
                and not self.text.endswith(" ")
            ):
                self.text += " "
            self._candidate = None
            return 0.0

        self._no_hand_since = None
        if letter != self._candidate:
            self._candidate = letter
            self._candidate_since = now
            self._committed = False
            return 0.0
        if self._committed:
            return 1.0

        progress = (now - self._candidate_since) / self.hold_seconds
        if progress >= 1.0:
            self.text += letter
            self._committed = True
            return 1.0
        return progress

    def add_space(self) -> None:
        if self.text and not self.text.endswith(" "):
            self.text += " "

    def backspace(self) -> None:
        self.text = self.text[:-1]

    def clear(self) -> None:
        self.text = ""
