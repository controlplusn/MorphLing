from manim import *

# ── Palette ───────────────────────────────────────────────────────────────────
BG = "#0D1117"
PANEL = "#161B22"
BORDER = "#30363D"
ACCENT = "#58A6FF"       # blue  – highlight / active
SUCCESS = "#3FB950"       # green – done
MUTED = "#8B949E"       # grey  – inactive
WHITE = "#E6EDF3"
YELLOW = "#D29922"
ORANGE = "#F0883E"
PURPLE   = "#A371F7"

STEPS = [
    "1 Normalization",
    "2 Caching",
    "3 Morph Tagging",
    "4 Seq. Assembly",
    "5 BPE Fallback",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def pipeline_bar(active: int, done: set) -> VGroup:
    boxes = VGroup()
    for i, label in enumerate(STEPS):
        if i in done:
            fg, bg_col, bord = SUCCESS, "#0D2410", SUCCESS
        elif i == active:
            fg, bg_col, bord = WHITE, "#0C2A4A", ACCENT
        else:
            fg, bg_col, bord = MUTED, PANEL, BORDER

        box = RoundedRectangle(
            corner_radius=0.12, width=2.55, height=0.55,
            fill_color=bg_col, fill_opacity=1,
            stroke_color=bord, stroke_width=2
        )
        txt = Text(label, font="Consolas", font_size=14, color=fg)
        txt.move_to(box)
        grp = VGroup(box, txt)
        boxes.add(grp)

    boxes.arrange(RIGHT, buff=0.18)
    boxes.move_to(ORIGIN).to_edge(UP, buff=0.22)
    return boxes


def divider() -> Line:
    return Line(
        start=LEFT * 6.8, end=RIGHT * 6.8,
        stroke_color=BORDER, stroke_width=1
    ).next_to(pipeline_bar(0, set()), DOWN, buff=0.18)


def section_title(text: str) -> Text:
    return Text(text, font="Consolas", font_size=26, color=ACCENT)


# ── Scene 0 – Pipeline Overview ───────────────────────────────────────────────

class PipelineOverview(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("MorphLing  ·  Tokenization Pipeline",
                     font="Consolas", font_size=30, color=WHITE)
        title.to_edge(UP, buff=0.45)

        subtitle = Text("Input → 5 sequential stages → Token sequence",
                        font="Consolas", font_size=18, color=MUTED)
        subtitle.next_to(title, DOWN, buff=0.18)

        # Input label
        input_lbl = Text("Raw Text", font="Consolas",
                         font_size=20, color=YELLOW)
        input_lbl.shift(LEFT * 5.5 + UP * 0.5)

        # Build step cards
        cards = VGroup()
        descriptions = [
            "NFKC + accent strip",
            "LFU memo cache",
            "Affix → special tags",
            "Root + tags → seq.",
            "OOV → BPE subwords",
        ]
        colors_list = [ACCENT, "#A371F7", SUCCESS, ORANGE, YELLOW]

        for i, (lbl, desc, col) in enumerate(zip(STEPS, descriptions, colors_list)):
            box = RoundedRectangle(
                corner_radius=0.2, width=2.5, height=1.1,
                fill_color=PANEL, fill_opacity=1,
                stroke_color=col, stroke_width=2.5
            )
            num  = Text(lbl[:2], font="Consolas", font_size=22, color=col)
            name = Text(lbl[2:], font="Consolas", font_size=17, color=WHITE)
            desc_t = Text(desc, font="Consolas", font_size=13, color=MUTED)
            VGroup(num, name, desc_t).arrange(DOWN, buff=0.06).move_to(box)
            cards.add(VGroup(box, num, name, desc_t))

        cards.arrange(RIGHT, buff=0.28)
        cards.move_to(ORIGIN + DOWN * 0.4)

        # Arrows between cards
        arrows = VGroup()
        for i in range(len(cards) - 1):
            a = Arrow(
                cards[i].get_right(), cards[i+1].get_left(),
                buff=0.08, stroke_width=2.5,
                color=BORDER, max_tip_length_to_length_ratio=0.25
            )
            arrows.add(a)

        self.play(FadeIn(title, shift=DOWN*0.2))
        self.play(FadeIn(subtitle))
        self.wait(0.3)

        for i, card in enumerate(cards):
            self.play(FadeIn(card, scale=0.92), run_time=0.35)
            if i < len(arrows):
                self.play(GrowArrow(arrows[i]), run_time=0.2)

        self.wait(1.2)
        self.play(FadeOut(VGroup(title, subtitle, cards, arrows)))


# ── Scene 1 – Normalization ───────────────────────────────────────────────────

class Normalization(Scene):
    def construct(self):
        self.camera.background_color = BG

        bar = pipeline_bar(active=0, done=set())
        div = Line(LEFT*6.8, RIGHT*6.8, stroke_color=BORDER, stroke_width=1
                   ).next_to(bar, DOWN, buff=0.18)

        self.add(bar, div)

        heading = Text("Phase 1: Normalization", font="Consolas",
                       font_size=26, color=ACCENT)
        heading.next_to(div, DOWN, buff=0.28).to_edge(LEFT, buff=0.5)

        # TODO: Improve text padding
        purpose = Text("Goal: clean messy/fancy text so the engine can parse it reliably",
                       font="Consolas", font_size=16, color=MUTED)
        purpose.next_to(heading, DOWN, buff=0.12).to_edge(LEFT, buff=0.5)

        self.play(FadeIn(heading), FadeIn(purpose), run_time=0.5)
        self.wait(0.3)

        # ── Two sub-step badges ──────────────────────────────────────────────
        def badge(text, col):
            box = RoundedRectangle(corner_radius=0.15, width=2.8, height=0.55,
                                   fill_color=PANEL, fill_opacity=1,
                                   stroke_color=col, stroke_width=2)
            lbl = Text(text, font="Consolas", font_size=17, color=col)
            lbl.move_to(box)
            return VGroup(box, lbl)

        b1 = badge("NFKC Normalization", ACCENT)
        b2 = badge("Accent Stripping", ORANGE)

        b1.next_to(purpose, DOWN, buff=0.32).to_edge(LEFT, buff=0.5)
        b2.next_to(b1, RIGHT, buff=0.55)
        arr_sub = Arrow(
            start=b1.get_right(), end=b2.get_left(),
            buff=0.1, stroke_width=2, color=BORDER,
            max_tip_length_to_length_ratio=0.25
        )
        sub_row = VGroup(b1, arr_sub, b2)

        self.play(FadeIn(b1), run_time=0.3)
        self.play(GrowArrow(arr_sub), FadeIn(b2), run_time=0.3)
        self.wait(0.2)

        ex_title = Text("Example", font="Consolas",
                        font_size=17, color=MUTED)
        ex_title.next_to(b2, DOWN, buff=0.4).to_edge(LEFT, buff=0.5)

        # "ｍａｌａｋｉｎｇ" word can't be read by the font, this is ok for now.
        raw_str  = 'Input:   "Punô ng luhà ang m a l a k i n g matá"'
        nfkc_str = 'NFKC:    "Punô ng luhà ang malaking matá."'
        strip_str = 'Output:  "Puno ng luha ang malaking mata."'

        def code_line(s, col=WHITE):
            return Text(s, font="Consolas", font_size=17, color=col)

        raw_t  = code_line(raw_str,  YELLOW)
        nfkc_t = code_line(nfkc_str, ACCENT)
        out_t  = code_line(strip_str, SUCCESS)

        raw_t.next_to(ex_title, DOWN, buff=0.20).to_edge(LEFT, buff=0.5)
        nfkc_t.next_to(raw_t, DOWN, buff=0.45).to_edge(LEFT, buff=0.5)
        out_t.next_to(nfkc_t, DOWN, buff=0.45).to_edge(LEFT, buff=0.5)

        self.play(FadeIn(ex_title))
        self.play(FadeIn(raw_t))
        self.wait(0.4)

        fw_note = Text(
            "-> m a l a k i n g = wide Unicode letters e.g. malaking",
            font="Consolas", font_size=15, color=YELLOW
        )
        fw_note.next_to(raw_t, DOWN, buff=0.1).to_edge(LEFT, buff=1.0)

        self.play(FadeIn(fw_note), run_time=0.3)
        self.wait(0.3)

        # TODO: Improve highlight effect
        # Highlight b1 (NFKC active)
        self.play(b1[0].animate.set_stroke(color=WHITE, width=3), run_time=0.2)
        self.play(FadeIn(nfkc_t))
        self.play(b1[0].animate.set_stroke(color=ACCENT, width=2), run_time=0.2)
        self.wait(0.3)

        # Accent note — placed below nfkc line
        acc_note = Text(
            "-> Accents stripped: ô->o   à->a   á->a   (diacritics removed)",
            font="Consolas", font_size=15, color=ORANGE
        )
        acc_note.next_to(nfkc_t, DOWN, buff=0.1).to_edge(LEFT, buff=1.0)

        # TODO: Improve highlight effect
        # Highlight b2 (Active Stripping active)
        self.play(b2[0].animate.set_stroke(color=WHITE, width=3), run_time=0.2)
        self.play(FadeIn(acc_note), run_time=0.3)
        self.play(b2[0].animate.set_stroke(color=ORANGE, width=2), run_time=0.2)
        self.wait(0.2)
        self.play(FadeIn(out_t))
        self.wait(0.5)

        # Key takeaway
        takeaway = Text(
            "Uniform text — dictionary lookups now succeed",
            font="Consolas", font_size=17, color=SUCCESS
        )
        takeaway.next_to(out_t, DOWN, buff=0.4).to_edge(LEFT, buff=0.5)
        self.play(FadeIn(takeaway, shift=UP*0.1))
        self.wait(1.0)

        # Return to pipeline
        content = VGroup(heading, purpose, sub_row, ex_title,
                         raw_t, fw_note, nfkc_t, acc_note, out_t, takeaway)
        self.play(FadeOut(content))

        # Update bar: step 0 done
        bar_done = pipeline_bar(active=1, done={0})
        div_done = Line(LEFT*6.8, RIGHT*6.8, stroke_color=BORDER, stroke_width=1
                        ).next_to(bar_done, DOWN, buff=0.18)

        self.play(
            Transform(bar, bar_done),
            Transform(div, div_done),
            run_time=0.5
        )
        self.wait(0.8)


# ── Combined render target ────────────────────────────────────────────────────

class NormalizationFull(Scene):
    def construct(self):
        PipelineOverview.construct(self)
        Normalization.construct(self)