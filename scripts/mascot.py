"""Tally, the count courier: the small character in the corner of the patient query.

Tally is our own drawing. It stands for what crosses hospital walls in this project: a variant
name goes out, counts come back, and no patient record ever travels. So Tally is a courier.
When a query runs it sets off (a hop, dots behind it), comes back holding the number it
fetched (the highest share of healthy carriers any hospital reported), reacts to the call
for a moment, then stands idle, blinking now and then and glancing at the chart.

Every pose is 4 rows by 12 columns, drawn with box drawing and a few plain symbols that every
terminal font carries, in the palette of query_drawing.py: the body in greys, the call's colour
only on the mark and the words that state the call. One short line of speech at most.

The poses, in order after a query, at FPS frames a second:

    thinking   ticks 0-1     hop, dots           "asking the other hospitals"
    carrying   ticks 2-3     the number held     "counts back, no records"
    verdict    ticks 4-8     happy / alert / shrug, one per call
    idle       from tick 9   blink at one tick in twelve, a glance at another

The web page (docs/demo/template.html) draws the same frames: export() hands them over as
data, and the page mirrors pose() only. A frame is a list of 4 lines; a line is a list of
(text, style key) pairs; a style key names a colour in STYLES, "" for plain spaces.
"""

from __future__ import annotations

from rich.text import Text

from query_drawing import BRIGHT, CALLS, FAINT, LABEL, PLAIN

NAME = "Tally"
FPS = 2.5  # frames a second: calm, well under anything that flashes

ART_WIDTH, SPEECH_WIDTH, GAP = 12, 26, 2
WIDTH = SPEECH_WIDTH + GAP + ART_WIDTH  # every line of the rendered mascot is exactly this wide
ROWS = 4

THINKING_TICKS, CARRYING_TICKS, VERDICT_TICKS = 2, 2, 5
IDLE_CYCLE, BLINK_AT, GLANCE_AT = 12, 5, 10

VERDICT_MOOD = {"LIKELY HARMLESS": "happy", "KEEP FLAGGED": "alert", "FREQUENCY SAYS NOTHING": "shrug"}

# ---------------------------------------------------------------------------
# The drawing
# ---------------------------------------------------------------------------
_TOP = [("   ", ""), ("╭──────╮", "box"), (" ", "")]
_BOTTOM = [("   ", ""), ("╰┬────┬╯", "box"), (" ", "")]
_FEET = [("    ", ""), ("╵", "box"), ("    ", ""), ("╵", "box"), ("  ", "")]
_BLANK = [(" " * ART_WIDTH, "")]


def _face(left: str, right: str, arm_left: str = " ", arm_right: str = " ") -> list[tuple[str, str]]:
    """The row with the eyes, and an arm on either side when the pose has them."""
    return [("  ", ""), (arm_left, "arm" if arm_left != " " else ""), ("│", "box"), (" ", ""), (left, "eye"), ("  ", ""),
            (right, "eye"), (" ", ""), ("│", "box"), (arm_right, "arm" if arm_right != " " else "")]


FRAMES: dict[str, list[list[list[tuple[str, str]]]]] = {
    "idle": [
        [_TOP, _face("•", "•"), _BOTTOM, _FEET],
        [_TOP, _face("-", "-"), _BOTTOM, _FEET],  # blink
        [_TOP, [("   ", ""), ("│", "box"), ("•", "eye"), ("  ", ""), ("•", "eye"), ("  ", ""), ("│", "box"), (" ", "")], _BOTTOM, _FEET],  # a glance at the chart
    ],
    "thinking": [
        [_TOP, [(" ", ""), ("·", "dot"), (" ", ""), ("│", "box"), (" ", ""), ("•", "eye"), ("  ", ""), ("•", "eye"), (" ", ""), ("│", "box"), (" ", "")], _BOTTOM, _FEET],
        [_BLANK, [("·", "dot"), ("  ", ""), ("╭──────╮", "box"), (" ", "")], _face("•", "•"), _BOTTOM],  # a hop
    ],
    "carrying": [
        [_TOP, _face("•", "•"), [("   ", ""), ("│", "box"), ("", "number"), ("│", "box"), (" ", "")], _BOTTOM],
    ],
    "happy": [
        [_TOP, _face("^", "^", "\\", "/"), _BOTTOM, _FEET],
        [_TOP, _face("^", "^", "─", "─"), _BOTTOM, _FEET],
    ],
    "alert": [
        [[("   ", ""), ("╭──────╮", "box"), ("!", "mark")], _face("•", "•"), _BOTTOM, _FEET],
    ],
    "shrug": [
        [[("   ", ""), ("╭──────╮", "box"), ("?", "mark")], _face("•", "•", "/", "\\"), _BOTTOM, _FEET],
    ],
}

SPEECH: dict[str, tuple[str, str]] = {  # what Tally says in each mood, and the style key of the words
    "idle": ("", ""),
    "thinking": ("asking the other hospitals", "quiet"),
    "carrying": ("counts back, no records", "quiet"),
    "happy": ("likely harmless", "call"),
    "alert": ("keep flagged", "call"),
    "shrug": ("cannot tell", "call"),
}


def styles(call: str | None) -> dict[str, str]:
    """Style key -> Rich style. The call's colour is the only colour, and only where the call is stated."""
    colour = CALLS[call].colour if call else LABEL
    return {"": "", "box": LABEL, "eye": PLAIN, "dot": FAINT, "arm": PLAIN, "mark": colour,
            "number": f"bold {BRIGHT}", "quiet": LABEL, "call": colour}


# ---------------------------------------------------------------------------
# Which pose, when
# ---------------------------------------------------------------------------
def pose(since: int, call: str | None) -> tuple[str, int]:
    """(mood, frame) this many ticks after the last query. No call on screen: idle."""
    if call is None:
        return idle_pose(since)
    if since < THINKING_TICKS:
        return "thinking", since % 2
    since -= THINKING_TICKS
    if since < CARRYING_TICKS:
        return "carrying", 0
    since -= CARRYING_TICKS
    if since < VERDICT_TICKS:
        mood = VERDICT_MOOD[call]
        return mood, (since // 2) % len(FRAMES[mood])  # a two-frame mood changes every other tick
    return idle_pose(since - VERDICT_TICKS)


def idle_pose(since: int) -> tuple[str, int]:
    at = since % IDLE_CYCLE
    return "idle", 1 if at == BLINK_AT else 2 if at == GLANCE_AT else 0


def lines(since: int, call: str | None, number: str = "") -> list[list[tuple[str, str]]]:
    """The four lines as (text, style key) pairs, each exactly WIDTH wide: the speech, right-justified,
    then the art. The number Tally carries is centred in its hands."""
    mood, frame = pose(since, call)
    art = FRAMES[mood][frame]
    speech, speech_style = SPEECH[mood]
    out = []
    for row, art_line in enumerate(art):
        line: list[tuple[str, str]] = []
        if row == 1 and speech:
            line += [(" " * (SPEECH_WIDTH - len(speech)), ""), (speech, speech_style), (" " * GAP, "")]
        else:
            line.append((" " * (SPEECH_WIDTH + GAP), ""))
        for text, style in art_line:
            if style == "number":
                text = number.center(6) if len(number) < 6 else number[:6]
            line.append((text, style))
        out.append(line)
    return out


def render(since: int, call: str | None, number: str = "") -> Text:
    """The mascot as Rich text, ROWS lines of WIDTH cells."""
    palette = styles(call)
    text = Text(no_wrap=True, end="")
    for row, line in enumerate(lines(since, call, number)):
        if row:
            text.append("\n")
        for chunk, style in line:
            text.append(chunk, palette[style] or None)
    return text


def export() -> dict:
    """Everything the web page needs to draw the same frames: the art, the words, and the timing."""
    return {
        "name": NAME, "fps": FPS, "rows": ROWS, "width": WIDTH, "speech_width": SPEECH_WIDTH, "art_width": ART_WIDTH, "gap": GAP,
        "ticks": [THINKING_TICKS, CARRYING_TICKS, VERDICT_TICKS], "idle": [IDLE_CYCLE, BLINK_AT, GLANCE_AT],
        "verdict_mood": VERDICT_MOOD,
        "frames": {mood: [[[list(pair) for pair in line] for line in frame] for frame in frames] for mood, frames in FRAMES.items()},
        "speech": {mood: list(words) for mood, words in SPEECH.items()},
    }
