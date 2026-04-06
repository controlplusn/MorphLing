"""
SentencePiece Visualization — Manim animation for thesis defense.

Requirements:
    pip install manim

Run BPE version:
    manim -pql sentencepiece_viz.py SentencePieceBPE

Run Unigram version:
    manim -pql sentencepiece_viz.py SentencePieceUnigram

Use -pqh for high quality when doing the final export.
"""

from manim import *

# ── Palette ───────────────────────────────────────────────────────────────────
BG           = "#0f1117"
PANEL_BG     = "#1a1d27"
ACCENT_TEAL  = "#1D9E75"
ACCENT_PURP  = "#7F77DD"
ACCENT_CORAL = "#D85A30"
ACCENT_AMBER = "#BA7517"
GRAY_LIGHT   = "#B4B2A9"
GRAY_DIM     = "#5F5E5A"
WHITE        = "#F1EFE8"

# ── Layout constants ──────────────────────────────────────────────────────────
# Manim default frame: y in [-4.0, 4.0]
PIPE_Y      = UP * 3.3   # pipeline strip (top of frame)
CONTENT_TOP = UP * 2.0   # just below the pipeline
CONTENT_MID = UP * 0.4   # vertical centre of content area
CONTENT_BOT = DOWN * 1.6 # lower content area

STEP_LABELS = [
    "Raw Input",
    "Normalization",
    "Seed Vocab",
    "Algorithm",
    "Final Vocab",
    "Encode/Decode",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_pipeline(active_idx=-1, algo_label="BPE / Unigram"):
    steps = STEP_LABELS.copy()
    steps[3] = algo_label

    boxes  = VGroup()
    arrows = VGroup()
    labels = VGroup()

    n       = len(steps)
    box_w   = 1.88
    box_h   = 0.48
    gap     = 0.14
    total_w = n * box_w + (n - 1) * gap
    start_x = -total_w / 2 + box_w / 2

    for i, label in enumerate(steps):
        x      = start_x + i * (box_w + gap)
        active = (i == active_idx)

        rect = RoundedRectangle(
            width=box_w, height=box_h,
            corner_radius=0.08,
            fill_color=ACCENT_TEAL if active else PANEL_BG,
            fill_opacity=1,
            stroke_color=ACCENT_TEAL if active else GRAY_DIM,
            stroke_width=2.5 if active else 1.0,
        ).move_to([x, 0, 0])

        txt = Text(
            label, font_size=14,
            color=WHITE if active else GRAY_LIGHT,
            weight=BOLD if active else NORMAL,
        ).move_to(rect.get_center())

        boxes.add(rect)
        labels.add(txt)

        if i < n - 1:
            mid_x = x + box_w / 2 + gap / 2
            arr = Arrow(
                start=[mid_x - gap / 2 + 0.02, 0, 0],
                end  =[mid_x + gap / 2 - 0.02, 0, 0],
                buff=0, stroke_width=1.5,
                color=GRAY_DIM, tip_length=0.12,
            )
            arrows.add(arr)

    return VGroup(boxes, arrows, labels)


def make_pipe_at(active_idx, algo_label="BPE / Unigram"):
    pipe = make_pipeline(active_idx, algo_label)
    pipe.move_to(PIPE_Y)
    return pipe


def token_box(text, color=ACCENT_TEAL, font_size=22):
    w   = max(len(text) * 0.20 + 0.45, 0.68)
    rect = RoundedRectangle(
        width=w, height=0.48,
        corner_radius=0.07,
        fill_color=color, fill_opacity=0.18,
        stroke_color=color, stroke_width=1.5,
    )
    txt = Text(text, font_size=font_size, color=color, weight=BOLD)
    txt.move_to(rect.get_center())
    return VGroup(rect, txt)


# ── Base Scene ────────────────────────────────────────────────────────────────

class SentencePieceBase(Scene):
    ALGO_LABEL = "BPE / Unigram"
    ALGO_COLOR = ACCENT_TEAL

    def setup(self):
        self.camera.background_color = BG
        self._pipe_mob = None

    # ── Pipeline helpers ──────────────────────────────────────────────────────

    def show_pipeline(self, active_idx, run_t=0.45, algo_label=None):
        label    = algo_label if algo_label is not None else self.ALGO_LABEL
        new_pipe = make_pipe_at(active_idx, label)
        if self._pipe_mob is None:
            self.play(FadeIn(new_pipe, shift=DOWN * 0.15), run_time=run_t)
        else:
            self.play(ReplacementTransform(self._pipe_mob, new_pipe), run_time=run_t)
        self._pipe_mob = new_pipe

    def clear_content(self, *mobs, run_t=0.35):
        valid = [m for m in mobs if m is not None]
        if valid:
            self.play(*[FadeOut(m) for m in valid], run_time=run_t)

    # ── Title ─────────────────────────────────────────────────────────────────

    def play_title(self):
        title = Text("SentencePiece Tokenizer", font_size=40, color=WHITE, weight=BOLD)
        sub   = Text(f"Algorithm: {self.ALGO_LABEL}", font_size=24, color=self.ALGO_COLOR)
        sub.next_to(title, DOWN, buff=0.35)
        grp = VGroup(title, sub).move_to(ORIGIN)
        self.play(FadeIn(grp, shift=UP * 0.25), run_time=0.7)
        self.wait(1.2)
        self.play(FadeOut(grp), run_time=0.45)

    # ── Pipeline overview ─────────────────────────────────────────────────────

    def play_pipeline_overview(self):
        self.show_pipeline(-1)
        lbl = Text("End-to-end tokenization pipeline", font_size=19, color=GRAY_LIGHT)
        lbl.move_to(CONTENT_TOP + DOWN * 0.3)
        self.play(FadeIn(lbl), run_time=0.4)
        self.wait(1.3)
        self.clear_content(lbl)

    # ── Step 1: Raw Input ─────────────────────────────────────────────────────

    def play_raw_input(self):
        self.show_pipeline(0)

        raw = Text('"the cat sat on the mat"', font_size=32, color=ACCENT_AMBER)
        raw.move_to(CONTENT_MID + UP * 0.4)

        note = Text("No pre-splitting — raw text goes in as-is",
                    font_size=19, color=GRAY_LIGHT)
        note.next_to(raw, DOWN, buff=0.5)

        self.play(Write(raw), run_time=0.8)
        self.play(FadeIn(note), run_time=0.4)
        self.wait(1.2)
        self.clear_content(raw, note)

    # ── Step 2: Normalization ─────────────────────────────────────────────────

    def play_normalization(self):
        self.show_pipeline(1)

        before = Text('"the cat sat on the mat"', font_size=25, color=ACCENT_AMBER)
        before.move_to(CONTENT_TOP + DOWN * 0.4)

        arrow = Arrow(
            start=before.get_bottom() + DOWN * 0.08,
            end  =before.get_bottom() + DOWN * 0.65,
            buff=0, stroke_width=2, color=GRAY_LIGHT, tip_length=0.15,
        )

        after = Text("▁the  ▁cat  ▁sat  ▁on  ▁the  ▁mat",
                     font_size=25, color=ACCENT_TEAL)
        after.next_to(arrow, DOWN, buff=0.1)

        note = Text("Spaces → ▁  (encodes word boundaries)",
                    font_size=18, color=GRAY_LIGHT)
        note.next_to(after, DOWN, buff=0.48)

        self.play(FadeIn(before), run_time=0.4)
        self.play(GrowArrow(arrow), run_time=0.35)
        self.play(TransformFromCopy(before, after), run_time=0.65)
        self.play(FadeIn(note), run_time=0.35)
        self.wait(1.3)
        self.clear_content(before, arrow, after, note)

    # ── Step 3: Seed Vocabulary ───────────────────────────────────────────────

    def play_seed_vocab(self):
        self.show_pipeline(2)

        layers = [
            ("Special tokens:",     "<unk>  <s>  </s>",              GRAY_LIGHT),
            ("Every character:",    "▁  t  h  e  c  a  s  o  n  m", ACCENT_PURP),
            ("Frequent substrings:","▁the  ▁cat  ▁sat  ▁on  ▁mat",  ACCENT_CORAL),
        ]

        rows = VGroup()
        for lname, ex, color in layers:
            l = Text(lname, font_size=19, color=GRAY_LIGHT, weight=BOLD)
            e = Text(ex,    font_size=19, color=color)
            rows.add(VGroup(l, e).arrange(RIGHT, buff=0.28))

        rows.arrange(DOWN, buff=0.40, aligned_edge=LEFT)
        rows.move_to(CONTENT_MID + UP * 0.1)

        note = Text("Draft lookup table — much larger than final vocab",
                    font_size=17, color=GRAY_LIGHT)
        note.next_to(rows, DOWN, buff=0.48)

        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.1) for r in rows],
                               lag_ratio=0.2), run_time=0.7)
        self.play(FadeIn(note), run_time=0.35)
        self.wait(1.4)
        self.clear_content(rows, note)

    # ── Step 5: Final Vocabulary ──────────────────────────────────────────────

    def play_final_vocab(self):
        self.show_pipeline(4)

        heading = Text("Final vocabulary",
                       font_size=21, color=WHITE, weight=BOLD)
        heading.move_to(CONTENT_TOP + DOWN * 0.2)

        tokens = self.get_final_tokens()
        tmobs  = VGroup(*[token_box(t, self.ALGO_COLOR) for t in tokens])
        tmobs.arrange(RIGHT, buff=0.22)
        tmobs.move_to(CONTENT_MID + UP * 0.2)

        note = Text(f"{len(tokens)} tokens shown  (real models: 32k – 64k)",
                    font_size=17, color=GRAY_LIGHT)
        note.next_to(tmobs, DOWN, buff=0.48)

        self.play(FadeIn(heading), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(t, shift=UP * 0.1) for t in tmobs],
                               lag_ratio=0.12), run_time=0.9)
        self.play(FadeIn(note), run_time=0.35)
        self.wait(1.4)
        self.clear_content(heading, tmobs, note)

    # ── Step 6: Encode / Decode ───────────────────────────────────────────────

    def play_encode_decode(self):
        self.show_pipeline(5)

        input_txt = Text('"the cat sat on the mat"', font_size=23, color=ACCENT_AMBER)
        input_txt.move_to(CONTENT_TOP + DOWN * 0.15)

        tokens = self.get_final_tokens()
        tmobs  = VGroup(*[token_box(t, self.ALGO_COLOR, font_size=19) for t in tokens])
        tmobs.arrange(RIGHT, buff=0.18)
        tmobs.move_to(CONTENT_MID + UP * 0.25)

        enc_arr = Arrow(
            start=input_txt.get_bottom() + DOWN * 0.05,
            end  =tmobs.get_top()        + UP   * 0.05,
            buff=0, color=GRAY_DIM, stroke_width=1.5, tip_length=0.14,
        )

        ids    = self.get_token_ids()
        id_txt = Text("Token IDs:  " + "  ".join(str(i) for i in ids),
                      font_size=20, color=ACCENT_PURP)
        id_txt.next_to(tmobs, DOWN, buff=0.40)

        dec_note = Text("Decode: IDs  →  tokens  →  original text",
                        font_size=17, color=GRAY_LIGHT)
        dec_note.next_to(id_txt, DOWN, buff=0.36)

        self.play(FadeIn(input_txt), run_time=0.35)
        self.play(GrowArrow(enc_arr), run_time=0.3)
        self.play(LaggedStart(*[FadeIn(t) for t in tmobs], lag_ratio=0.1),
                  run_time=0.7)
        self.play(FadeIn(id_txt), run_time=0.35)
        self.play(FadeIn(dec_note), run_time=0.35)
        self.wait(1.5)
        self.clear_content(input_txt, enc_arr, tmobs, id_txt, dec_note)

    # ── Outro ─────────────────────────────────────────────────────────────────

    def play_outro(self):
        self.clear_content(self._pipe_mob)
        self._pipe_mob = None
        done = Text(f"SentencePiece  ·  {self.ALGO_LABEL}",
                    font_size=28, color=self.ALGO_COLOR, weight=BOLD)
        done.move_to(ORIGIN)
        self.play(FadeIn(done, shift=UP * 0.2), run_time=0.6)
        self.wait(1.2)
        self.play(FadeOut(done), run_time=0.45)

    # ── Hooks for subclasses ──────────────────────────────────────────────────

    def get_final_tokens(self):
        raise NotImplementedError

    def get_token_ids(self):
        raise NotImplementedError

    def play_algorithm_step(self):
        raise NotImplementedError

    # ── Main construct ────────────────────────────────────────────────────────

    def construct(self):
        self.play_title()
        self.play_pipeline_overview()
        self.play_raw_input()
        self.play_normalization()
        self.play_seed_vocab()
        self.play_algorithm_step()
        self.play_final_vocab()
        self.play_encode_decode()
        self.play_outro()


# ── BPE Scene ─────────────────────────────────────────────────────────────────

class SentencePieceBPE(SentencePieceBase):
    ALGO_LABEL = "BPE"
    ALGO_COLOR = ACCENT_PURP

    def get_final_tokens(self):
        return ["▁the", "▁cat", "▁sat", "▁on", "▁mat"]

    def get_token_ids(self):
        return [13, 14, 15, 16, 13, 17]

    def play_algorithm_step(self):
        self.show_pipeline(3, algo_label="BPE")

        heading = Text("BPE — bottom-up merging", font_size=23,
                       color=ACCENT_PURP, weight=BOLD)
        heading.move_to(CONTENT_TOP + DOWN * 0.1)
        self.play(FadeIn(heading), run_time=0.4)

        # State 0: raw characters
        chars  = ["▁t", "h", "e", "▁c", "a", "t", "▁s", "a", "t"]
        c_mobs = VGroup(*[token_box(c, GRAY_LIGHT, font_size=19) for c in chars])
        c_mobs.arrange(RIGHT, buff=0.10)
        c_mobs.move_to(CONTENT_MID + UP * 0.55)

        sub0 = Text("Start: every character is its own token",
                    font_size=17, color=GRAY_LIGHT)
        sub0.next_to(c_mobs, DOWN, buff=0.28)

        self.play(LaggedStart(*[FadeIn(m) for m in c_mobs], lag_ratio=0.06),
                  run_time=0.7)
        self.play(FadeIn(sub0), run_time=0.3)
        self.wait(0.8)

        # Merge 1: ▁t + h → ▁th
        note1 = Text("Most frequent pair:  ▁t + h  →  ▁th",
                     font_size=17, color=ACCENT_PURP)
        note1.next_to(sub0, DOWN, buff=0.26)
        self.play(FadeIn(note1), run_time=0.35)

        m1_g = VGroup(*[token_box(c, ACCENT_PURP if c == "▁th" else GRAY_LIGHT,
                                  font_size=19)
                        for c in ["▁th", "e", "▁c", "a", "t", "▁s", "a", "t"]])
        m1_g.arrange(RIGHT, buff=0.10)
        m1_g.move_to(CONTENT_MID + UP * 0.55)
        self.play(ReplacementTransform(c_mobs, m1_g), run_time=0.55)
        self.wait(0.6)

        # Merge 2: ▁th + e → ▁the
        note2 = Text("Next pair:  ▁th + e  →  ▁the",
                     font_size=17, color=ACCENT_PURP)
        note2.move_to(note1.get_center())
        self.play(ReplacementTransform(note1, note2), run_time=0.35)

        m2_g = VGroup(*[token_box(c, ACCENT_TEAL if c == "▁the" else GRAY_LIGHT,
                                  font_size=19)
                        for c in ["▁the", "▁c", "a", "t", "▁s", "a", "t"]])
        m2_g.arrange(RIGHT, buff=0.12)
        m2_g.move_to(CONTENT_MID + UP * 0.55)
        self.play(ReplacementTransform(m1_g, m2_g), run_time=0.55)
        self.wait(0.6)

        dots = Text("… merging continues until vocab size reached …",
                    font_size=16, color=GRAY_DIM)
        dots.next_to(note2, DOWN, buff=0.26)
        self.play(FadeIn(dots), run_time=0.35)
        self.wait(1.0)

        self.clear_content(heading, m2_g, sub0, note2, dots)


# ── Unigram Scene ─────────────────────────────────────────────────────────────

class SentencePieceUnigram(SentencePieceBase):
    ALGO_LABEL = "Unigram LM"
    ALGO_COLOR = ACCENT_CORAL

    def get_final_tokens(self):
        return ["▁the", "▁cat", "▁sat", "▁on", "▁mat"]

    def get_token_ids(self):
        return [13, 14, 15, 16, 13, 17]

    def play_algorithm_step(self):
        self.show_pipeline(3, algo_label="Unigram LM")

        heading = Text("Unigram LM — top-down pruning", font_size=23,
                       color=ACCENT_CORAL, weight=BOLD)
        heading.move_to(CONTENT_TOP + DOWN * 0.1)
        self.play(FadeIn(heading), run_time=0.4)

        # Large initial vocab
        all_toks = ["▁the", "▁cat", "▁sat", "▁on", "▁mat",
                    "▁ca",  "▁sa",  "th",   "he",  "at",
                    "t",    "h",    "e",    "c",   "a",   "s",  "o", "n"]

        big_grp = VGroup(*[token_box(t, GRAY_LIGHT, font_size=15) for t in all_toks])
        big_grp.arrange_in_grid(rows=2, cols=9, buff=(0.08, 0.16))
        big_grp.move_to(CONTENT_MID + UP * 0.5)

        sub0 = Text("Start: large seed vocab (all candidates)",
                    font_size=17, color=GRAY_LIGHT)
        sub0.next_to(big_grp, DOWN, buff=0.26)

        self.play(LaggedStart(*[FadeIn(m, shift=UP * 0.05) for m in big_grp],
                               lag_ratio=0.04), run_time=0.8)
        self.play(FadeIn(sub0), run_time=0.3)
        self.wait(0.8)

        # Prune round
        note1 = Text("EM scores each token → remove lowest log-prob",
                     font_size=17, color=ACCENT_CORAL)
        note1.next_to(sub0, DOWN, buff=0.26)
        self.play(FadeIn(note1), run_time=0.35)

        prune_idxs = list(range(10, 18))
        strikes = VGroup(*[
            Line(big_grp[i].get_left(), big_grp[i].get_right(),
                 color=RED, stroke_width=2)
            for i in prune_idxs
        ])
        self.play(LaggedStart(*[Create(s) for s in strikes], lag_ratio=0.06),
                  run_time=0.55)
        self.play(
            *[FadeOut(big_grp[i]) for i in prune_idxs],
            FadeOut(strikes),
            run_time=0.45,
        )
        self.wait(0.5)

        dots = Text("… pruning repeats until target vocab size …",
                    font_size=16, color=GRAY_DIM)
        dots.next_to(note1, DOWN, buff=0.26)
        self.play(FadeIn(dots), run_time=0.35)
        self.wait(1.0)

        remaining = VGroup(*[big_grp[i] for i in range(len(all_toks))
                             if i not in prune_idxs])
        self.clear_content(heading, remaining, sub0, note1, dots)