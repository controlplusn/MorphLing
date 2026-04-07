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
PURPLE = "#A371F7"

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


# ── Scene 2 – Caching ─────────────────────────────────────────────────────────
class Caching(Scene):
    def construct(self):
        self.camera.background_color = BG
 
        bar = pipeline_bar(active=1, done={0})
        div = Line(LEFT*6.8, RIGHT*6.8, stroke_color=BORDER, stroke_width=1
                   ).next_to(bar, DOWN, buff=0.18)
        self.add(bar, div)
 
        heading = Text("Phase 2: Caching", font="Consolas",
                       font_size=26, color=PURPLE)
        heading.next_to(div, DOWN, buff=0.25).to_edge(LEFT, buff=0.5)
 
        purpose = Text(
            "Goal: avoid re-analyzing the same word twice  (speed up pretraining)",
            font="Consolas", font_size=15, color=MUTED
        )
        purpose.next_to(heading, DOWN, buff=0.10).to_edge(LEFT, buff=0.5)
 
        self.play(FadeIn(heading), FadeIn(purpose), run_time=0.5)
        self.wait(0.2)

        # Two mechanism
        def badge(text, col, w=3.2):
            box = RoundedRectangle(corner_radius=0.15, width=w, height=0.55,
                                   fill_color=PANEL, fill_opacity=1,
                                   stroke_color=col, stroke_width=2)
            lbl = Text(text, font="Consolas", font_size=16, color=col)
            lbl.move_to(box)
            return VGroup(box, lbl)
 
        bm = badge("Memoization Dict", PURPLE)
        bl = badge("LFU Cache  (cap=3)", ACCENT)
 
        bm.next_to(purpose, DOWN, buff=0.28).to_edge(LEFT, buff=0.5)
        bl.next_to(bm, RIGHT, buff=0.5)
 
        bm_note = Text("word -> root saved on first analysis",
                       font="Consolas", font_size=13, color=MUTED)
        bm_note.next_to(bm, DOWN, buff=0.08).to_edge(LEFT, buff=0.5)
 
        bl_note = Text("evicts least-used entry when full",
                       font="Consolas", font_size=13, color=MUTED)
        bl_note.next_to(bl, DOWN, buff=0.08).align_to(bl, LEFT)
 
        self.play(FadeIn(bm), FadeIn(bl), run_time=0.4)
        self.play(FadeIn(bm_note), FadeIn(bl_note), run_time=0.3)
        self.wait(0.3)

        # Cache table
        col_w = [2.4, 1.6, 1.4]

        def make_cell(text, col=WHITE, w=2.8, h=0.42, bg=PANEL, border=BORDER):
            box = Rectangle(width=w, height=h,
                            fill_color=bg, fill_opacity=1,
                            stroke_color=border, stroke_width=1.5)
            lbl = Text(text, font="Consolas", font_size=14, color=col)
            lbl.move_to(box)
            return VGroup(box, lbl)
 
        def make_row(word, root, freq, wc=WHITE, rc=SUCCESS, fc=ACCENT,
                     bg=PANEL, border=BORDER):
            c1 = make_cell(word, wc, col_w[0], bg=bg, border=border)
            c2 = make_cell(root, rc, col_w[1], bg=bg, border=border)
            c3 = make_cell(freq, fc, col_w[2], bg=bg, border=border)
            row = VGroup(c1, c2, c3)
            row.arrange(RIGHT, buff=0)
            return row
 
        hdr = make_row("Word", "Root", "Freq",
                       wc=MUTED, rc=MUTED, fc=MUTED,
                       bg="#1C2128", border=ACCENT)
        hdr.next_to(bm_note, DOWN, buff=0.32).to_edge(LEFT, buff=0.5)
        self.play(FadeIn(hdr))

        # Right Panel
        RIGHT_ANCHOR = hdr.get_right() + RIGHT * 0.6

        # Vertical slots (top → bottom) in the right column:
        #   slot_input   : current input word
        #   slot_status  : HIT / MISS / FULL label
        #   slot_evict   : eviction warning (only shown when needed)
        slot_input_y = 1.0
        slot_status_y = 0.2
        slot_evict_y = -0.6

        def right_text(txt, col, font_size=16):
            t = Text(txt, font="Consolas", font_size=font_size, color=col)
            t.next_to(RIGHT_ANCHOR, RIGHT, buff=0.1)
            return t


        # Simulation data
        # Steps: word, root, resulting cache state [(word,root,freq)], note, evict?
        CACHE_CAP = 3
        steps = [
            ("kumakain",    "kain",  "Cache miss -> stem -> save"),
            ("naglalakad",  "lakad", "Cache miss -> stem -> save"),
            ("kumakain",    "kain",  "Cache HIT  -> freq +1"),
            ("umuulan",     "ulan",  "Cache miss -> stem -> save  [FULL]"),
            ("naglalakad",  "lakad", "Cache HIT  -> freq +1"),
            ("napakaganda", "ganda", "Cache miss -> EVICT umuulan (freq=1) -> save"),
            ("kumakain",    "kain",  "Cache HIT  -> freq +1"),
        ]

        cache        = []
        row_mobjects = []
        input_mob    = None
        status_mob   = None
        evict_mob    = None
        arrow_mob    = None # input → table arrow

        for step_i, (word, root, note_txt) in enumerate(steps):
            new_input = Text(f'Input:  "{word}"',
                             font="Consolas", font_size=16, color=YELLOW)
            
            # Absolute anchor
            new_input.next_to(RIGHT, buff=0.8)
            new_input.set_y(hdr.get_top()[1])
 
            swap_anims = [FadeIn(new_input, run_time=0.5)]
            if input_mob:
                swap_anims.append(FadeOut(input_mob, run_time=0.4))
            if arrow_mob:
                swap_anims.append(FadeOut(arrow_mob, run_time=0.3))
            if status_mob:
                swap_anims.append(FadeOut(status_mob, run_time=0.3))
            if evict_mob:
                swap_anims.append(FadeOut(evict_mob, run_time=0.3))
                evict_mob = None
            self.play(*swap_anims)
            input_mob = new_input
            status_mob = None
            self.wait(1.0)


            # Determine HIT or MISS
            hit_idx = next((i for i, r in enumerate(cache) if r[0] == word), None)

            if hit_idx is not None:
                # HIT
                cache[hit_idx][2] += 1
 
                # Arrow: input → matching row
                target_row = row_mobjects[hit_idx]
                arr = Arrow(
                    start=new_input.get_bottom(),
                    end=target_row.get_right() + RIGHT * 0.05,
                    buff=0.08, stroke_width=2.5, color=SUCCESS,
                    max_tip_length_to_length_ratio=0.3
                )
                self.play(GrowArrow(arr), run_time=0.5)
                arrow_mob = arr
                self.wait(0.3)
 
                # Flash row green
                self.play(
                    target_row[0].animate.set_fill(color="#0D3320"),
                    run_time=0.5
                )
                self.wait(0.8)
                self.play(
                    target_row[0].animate.set_fill(color=PANEL),
                    run_time=0.4
                )
 
                # Update freq
                new_row = make_row(cache[hit_idx][0], cache[hit_idx][1],
                                   str(cache[hit_idx][2]))
                new_row.move_to(target_row)
                self.play(Transform(row_mobjects[hit_idx], new_row), run_time=0.6)
                self.wait(0.4)

            else:
                # MISS
                if len(cache) >= CACHE_CAP:
                    evict_entry = cache[min(range(len(cache)),
                                           key=lambda i: cache[i][2])]
                    evict_idx   = cache.index(evict_entry)
 
                    evict_txt = (f'Cache full!\n'
                                 f'Evicting "{evict_entry[0]}"'
                                 f'  (freq={evict_entry[2]})')
                    evict_mob = Text(evict_txt, font="Consolas",
                                     font_size=13, color=ORANGE)
                    evict_mob.next_to(new_input, DOWN, buff=0.3)
 
                    self.play(FadeIn(evict_mob), run_time=0.5)
                    self.wait(1.5)
 
                    # Arrow: input → evicted row
                    target_evict = row_mobjects[evict_idx]
                    arr_evict = Arrow(
                        start=new_input.get_bottom(),
                        end=target_evict.get_right() + RIGHT * 0.05,
                        buff=0.08, stroke_width=2.5, color=ORANGE,
                        max_tip_length_to_length_ratio=0.3
                    )
                    self.play(GrowArrow(arr_evict), run_time=0.5)
                    arrow_mob = arr_evict
                    self.wait(0.3)
 
                    # Flash evicted row red
                    self.play(
                        target_evict[0].animate.set_fill(color="#3D0A0A"),
                        run_time=0.5
                    )
                    self.wait(1.0)
                    self.play(
                        FadeOut(target_evict),
                        FadeOut(arr_evict),
                        FadeOut(evict_mob),
                        run_time=0.5
                    )
                    arrow_mob = None
                    evict_mob = None
 
                    cache.pop(evict_idx)
                    row_mobjects.pop(evict_idx)
 
                    shift_anims = [row_mobjects[j].animate.shift(UP * 0.42)
                                   for j in range(evict_idx, len(row_mobjects))]
                    if shift_anims:
                        self.play(*shift_anims, run_time=0.5)
                    self.wait(0.4)
 
                # Add new row
                cache.append([word, root, 1])
                new_row = make_row(word, root, "1")
                if row_mobjects:
                    new_row.next_to(row_mobjects[-1], DOWN, buff=0)
                else:
                    new_row.next_to(hdr, DOWN, buff=0)
 
                # Arrow: input → new row destination
                arr_new = Arrow(
                    start=new_input.get_bottom(),
                    end=new_row.get_right() + RIGHT * 0.05,
                    buff=0.08, stroke_width=2.5, color=ACCENT,
                    max_tip_length_to_length_ratio=0.3
                )
                self.play(
                    FadeIn(new_row, shift=RIGHT * 0.2),
                    GrowArrow(arr_new),
                    run_time=0.7
                )
                arrow_mob = arr_new
                row_mobjects.append(new_row)
                self.wait(0.5)
            
            # Status note
            note_color = SUCCESS if "HIT" in note_txt else (
                ORANGE if "EVICT" in note_txt else ACCENT
            )
            new_status = Text(note_txt, font="Consolas",
                              font_size=13, color=note_color)
            new_status.next_to(new_input, DOWN, buff=0.25)
 
            self.play(FadeIn(new_status), run_time=0.4)
            status_mob = new_status
            self.wait(1.8)

        # Takeaway
        fade_out = []
        for m in [input_mob, status_mob, evict_mob, arrow_mob]:
            if m:
                fade_out.append(FadeOut(m))
        if fade_out:
            self.play(*fade_out)
 
        takeaway = Text(
            "High-freq words served instantly — no redundant stemming",
            font="Consolas", font_size=16, color=SUCCESS
        )
        last_ref = row_mobjects[-1] if row_mobjects else hdr
        takeaway.next_to(last_ref, DOWN, buff=0.35).to_edge(LEFT, buff=0.5)
        self.play(FadeIn(takeaway, shift=UP * 0.1))
        self.wait(1.0)

        # Cleanup
        all_content = VGroup(heading, purpose, bm, bl, bm_note, bl_note,
                             hdr, takeaway, *row_mobjects)
        self.play(FadeOut(all_content))
 
        bar_done = pipeline_bar(active=2, done={0, 1})
        div_done = Line(LEFT*6.8, RIGHT*6.8, stroke_color=BORDER, stroke_width=1
                        ).next_to(bar_done, DOWN, buff=0.18)
        self.play(Transform(bar, bar_done), Transform(div, div_done), run_time=0.5)
        self.wait(0.8)


# ── Scene 3 – Morphological Tagging ──────────────────────────────────────────
class MorphTagging(Scene):
    def construct(self):
        self.camera.background_color = BG
 
        bar = pipeline_bar(active=2, done={0, 1})
        div = Line(LEFT*6.8, RIGHT*6.8, stroke_color=BORDER, stroke_width=1
                   ).next_to(bar, DOWN, buff=0.18)
        self.add(bar, div)
 
        # SECTION: Morph Header
        heading = Text("Phase 3: Morphological Tagging", font="Consolas",
                       font_size=24, color=SUCCESS)
        heading.next_to(div, DOWN, buff=0.25).to_edge(LEFT, buff=0.5)
 
        purpose = Text(
            "Goal: replace stripped affixes with special tokens so the LLM keeps grammar",
            font="Consolas", font_size=14, color=MUTED
        )
        purpose.next_to(heading, DOWN, buff=0.10).to_edge(LEFT, buff=0.5)
 
        self.play(FadeIn(heading), FadeIn(purpose), run_time=0.5)
        self.wait(0.5)

        # Overview badge
        def badge(text, col, w=3.6):
            box = RoundedRectangle(corner_radius=0.15, width=w, height=0.52,
                                   fill_color=PANEL, fill_opacity=1,
                                   stroke_color=col, stroke_width=2)
            lbl = Text(text, font="Consolas", font_size=15, color=col)
            lbl.move_to(box)
            return VGroup(box, lbl)
 
        s1 = badge("Step 1: Strip word -> find root", ACCENT)
        s2 = badge("Step 2: BPE-encode root", PURPLE)
        s3 = badge("Step 3: Affixes -> special tokens", ORANGE)
 
        s1.next_to(purpose, DOWN, buff=0.28).to_edge(LEFT, buff=0.5)
        a12 = Arrow(s1.get_right(), s1.get_right() + RIGHT*0.4,
                    buff=0.05, stroke_width=2, color=BORDER,
                    max_tip_length_to_length_ratio=0.3)
        s2.next_to(a12, RIGHT, buff=0.05)
        a23 = Arrow(s2.get_right(), s2.get_right() + RIGHT*0.4,
                    buff=0.05, stroke_width=2, color=BORDER,
                    max_tip_length_to_length_ratio=0.3)
        s3.next_to(a23, RIGHT, buff=0.05)
 
        # NOTE: Fixed transition time for a smoother transition
        self.play(FadeIn(s1, shift=RIGHT * 0.2), run_time=0.8)
        self.wait(0.6)
        self.play(GrowArrow(a12), FadeIn(s2), run_time=0.8)
        self.wait(0.6)
        self.play(GrowArrow(a23), FadeIn(s3), run_time=0.8)
        self.wait(1.3)

        # Divider line
        ex_div = Line(LEFT*6.3, RIGHT*6.3, stroke_color=BORDER, stroke_width=1)
        ex_div.next_to(s1, DOWN, buff=0.30).to_edge(LEFT, buff=0.5)
 
        ex_lbl = Text('Example word:  "tinatawagan"',
                      font="Consolas", font_size=17, color=YELLOW)
        ex_lbl.next_to(ex_div, DOWN, buff=0.20).to_edge(LEFT, buff=0.5)

        # NOTE: Slower transition time
        self.play(FadeIn(ex_div), FadeIn(ex_lbl), run_time=0.4)
        self.wait(1.2)

        # SECTION: Strip -> root
        self.play(s1[0].animate.set_stroke(color=WHITE, width=3), run_time=0.5)
        self.wait(0.8)
 
        # Layout: prefix/infix markers + root + suffix, left-aligned
        strip_label = Text("Stripping:", font="Consolas", font_size=15, color=MUTED)
        strip_label.next_to(ex_lbl, DOWN, buff=0.28).to_edge(LEFT, buff=0.5)
 
        # Build the decomposition visually as colored tokens in a row
        def token_chip(text, col, bg_col):
            box = RoundedRectangle(corner_radius=0.1, width=len(text)*0.18+0.4,
                                   height=0.42,
                                   fill_color=bg_col, fill_opacity=1,
                                   stroke_color=col, stroke_width=1.8)
            lbl = Text(text, font="Consolas", font_size=16, color=col)
            lbl.move_to(box)
            return VGroup(box, lbl)
 
        chip_in = token_chip("-in-",   ORANGE,  "#2A1A00")   # infix
        chip_redup = token_chip("ta-",   PURPLE,  "#1E0A2A")   # reduplication
        chip_root = token_chip("tawag",  SUCCESS, "#0A2010")   # root
        chip_suf = token_chip("-an",    YELLOW,  "#2A1E00")   # suffix
 
        decomp_row = VGroup(chip_in, chip_redup, chip_root, chip_suf)
        decomp_row.arrange(RIGHT, buff=0.18)
        decomp_row.next_to(strip_label, RIGHT, buff=0.3)
 
        # Labels below each chip
        lbl_in = Text("infix",   font="Consolas", font_size=12, color=ORANGE)
        lbl_redup = Text("redup",   font="Consolas", font_size=12, color=PURPLE)
        lbl_root = Text("root",    font="Consolas", font_size=12, color=SUCCESS)
        lbl_suf = Text("suffix",  font="Consolas", font_size=12, color=YELLOW)
 
        lbl_in.next_to(chip_in,    DOWN, buff=0.08)
        lbl_redup.next_to(chip_redup, DOWN, buff=0.08)
        lbl_root.next_to(chip_root,  DOWN, buff=0.08)
        lbl_suf.next_to(chip_suf,   DOWN, buff=0.08)
 
        chip_labels = VGroup(lbl_in, lbl_redup, lbl_root, lbl_suf)
 
        self.play(FadeIn(strip_label), run_time=0.5)
        for chip, lbl in zip(decomp_row, chip_labels):
            self.play(FadeIn(chip, shift=UP*0.1), FadeIn(lbl), run_time=0.6)
            self.wait(0.5)
 
        self.play(s1[0].animate.set_stroke(color=ACCENT, width=2), run_time=0.5)
        self.wait(0.8)
 
        # SECTION: BPE Encode root
        self.play(s2[0].animate.set_stroke(color=WHITE, width=3), run_time=0.5)
        self.wait(0.2)
 
        bpe_label = Text("BPE encodes root:", font="Consolas", font_size=15, color=MUTED)
        bpe_label.next_to(strip_label, DOWN, buff=0.65).to_edge(LEFT, buff=0.5)
 
        # Arrow from root chip down to BPE result
        bpe_arrow = Arrow(
            start=chip_root.get_bottom(),
            end=chip_root.get_bottom() + DOWN * 0.55,
            buff=0.05, stroke_width=2, color=SUCCESS,
            max_tip_length_to_length_ratio=0.35
        )
 
        # "tawag" is common -> kept as single token
        bpe_result = token_chip("[tawag]", SUCCESS, "#0A2010")
        bpe_result.next_to(bpe_label, RIGHT, buff=0.3)
 
        bpe_note = Text("common root -> kept as 1 token",
                        font="Consolas", font_size=13, color=MUTED)
        bpe_note.next_to(bpe_result, RIGHT, buff=0.3)
 
        self.play(GrowArrow(bpe_arrow), run_time=0.5)
        self.play(FadeIn(bpe_label), FadeIn(bpe_result), run_time=0.4)
        self.play(FadeIn(bpe_note), run_time=0.8)
        self.play(s2[0].animate.set_stroke(color=PURPLE, width=2), run_time=0.5)
        self.wait(0.8)
 
        # SECTION: Affixes -> special tokens
        self.play(s3[0].animate.set_stroke(color=WHITE, width=3), run_time=0.5)
        self.wait(0.2)
 
        tag_label = Text("Affixes -> tokens:", font="Consolas", font_size=15, color=MUTED)
        tag_label.next_to(bpe_label, DOWN, buff=0.55).to_edge(LEFT, buff=0.5)
 
        # Map chips to their special tokens
        tag_map = [
            (chip_in,    "##INFIX",  ORANGE),
            (chip_redup, "##REDUP",  PURPLE),
            (chip_suf,   "##SUFFIX", YELLOW),
        ]
 
        tag_chips = VGroup()
        tag_arrows = VGroup()
        for src_chip, tag_txt, col in tag_map:
            tc = token_chip(tag_txt, col, PANEL)
            ta = Arrow(
                start=src_chip.get_bottom(),
                end=tc.get_top(),
                buff=0.08, stroke_width=1.8, color=col,
                max_tip_length_to_length_ratio=0.3
            )
            tag_chips.add(tc)
            tag_arrows.add(ta)
 
        tag_chips.arrange(RIGHT, buff=0.3)
        tag_chips.next_to(tag_label, RIGHT, buff=0.3)
 
        # Reposition arrows now that tag_chips are placed
        for i, (src_chip, tag_txt, col) in enumerate(tag_map):
            tag_arrows[i].become(Arrow(
                start=src_chip.get_bottom(),
                end=tag_chips[i].get_top(),
                buff=0.08, stroke_width=1.5, color=col,
                max_tip_length_to_length_ratio=0.3
            ))
 
        self.play(FadeIn(tag_label), run_time=0.6)
        for tc, ta in zip(tag_chips, tag_arrows):
            self.play(GrowArrow(ta), FadeIn(tc, shift=DOWN*0.1), run_time=0.8)
            self.wait(0.3)
 
        self.play(s3[0].animate.set_stroke(color=ORANGE, width=2), run_time=0.5)
        self.wait(0.8)
 
        # ── Final assembled token sequence ────────────────────────────────────
        final_label = Text("Final token sequence:", font="Consolas",
                           font_size=15, color=MUTED)
        final_label.next_to(tag_label, DOWN, buff=0.55).to_edge(LEFT, buff=0.5)

        # NOTE: Fixed ordering of tokenization output   
        # Tokens in order: ##INFIX, ##REDUP, [tawag], ##SUFFIX
        f1 = token_chip("[_tawag]",  SUCCESS, "#0A2010")
        f2 = token_chip("##REDUP",  PURPLE, PANEL)
        f3 = token_chip("in##INFIX",  ORANGE, PANEL)
        f4 = token_chip("an##SUFFIX", YELLOW, PANEL)
 
        final_seq = VGroup(f1, f2, f3, f4)
        final_seq.arrange(RIGHT, buff=0.22)
        final_seq.next_to(final_label, RIGHT, buff=0.3)
 
        # Small connector arrows between final tokens
        conn_arrows = VGroup()
        for i in range(len(final_seq) - 1):
            ca = Arrow(
                final_seq[i].get_right(), final_seq[i+1].get_left(),
                buff=0.06, stroke_width=1.5, color=BORDER,
                max_tip_length_to_length_ratio=0.4
            )
            conn_arrows.add(ca)
 
        self.play(FadeIn(final_label), run_time=0.8)
        self.wait(0.3)  
        for ft in final_seq:
            self.play(FadeIn(ft, scale=0.9), run_time=0.5)
        self.play(*[GrowArrow(ca) for ca in conn_arrows], run_time=0.6)
        self.wait(0.8)
 
        # Takeaway
        takeaway = Text(
            "Grammar is preserved as tokens — the LLM sees structure, not just characters",
            font="Consolas", font_size=14, color=SUCCESS
        )
        takeaway.next_to(final_label, DOWN, buff=0.38).to_edge(LEFT, buff=0.5)
        self.play(FadeIn(takeaway, shift=UP*0.1))
        self.wait(1.5)
 
        # ── cleanup + advance bar ─────────────────────────────────────────────
        content = VGroup(
            heading, purpose,
            s1, a12, s2, a23, s3,
            ex_div, ex_lbl,
            strip_label, decomp_row, chip_labels,
            bpe_arrow, bpe_label, bpe_result, bpe_note,
            tag_label, tag_chips, tag_arrows,
            final_label, final_seq, conn_arrows,
            takeaway,
        )
        self.play(FadeOut(content))
 
        bar_done = pipeline_bar(active=3, done={0, 1, 2})
        div_done = Line(LEFT*6.8, RIGHT*6.8, stroke_color=BORDER, stroke_width=1
                        ).next_to(bar_done, DOWN, buff=0.18)
        self.play(Transform(bar, bar_done), Transform(div, div_done), run_time=0.5)
        self.wait(0.8)


# SECTION: Sequence Assembly
class SequenceAssembly(Scene):
    def construct(self):
        self.camera.background_color = BG
 
        bar = pipeline_bar(active=3, done={0, 1, 2})
        div = Line(LEFT * 6.8, RIGHT * 6.8, stroke_color=BORDER, stroke_width=1
                   ).next_to(bar, DOWN, buff=0.18)
        self.add(bar, div)
 
        # ── Header ────────────────────────────────────────────────────────────
        heading = Text("Phase 4: Sequence Assembly", font="Consolas",
                       font_size=24, color=ORANGE)
        heading.next_to(div, DOWN, buff=0.25).to_edge(LEFT, buff=0.5)
 
        purpose = Text(
            "Goal: hook all pieces together in the exact order, with invisible glue",
            font="Consolas", font_size=14, color=MUTED
        )
        purpose.next_to(heading, DOWN, buff=0.10).to_edge(LEFT, buff=0.5)
 
        self.play(FadeIn(heading), FadeIn(purpose), run_time=0.5)
        self.wait(0.3)
 
        # ── Helper: token chip (reused style) ─────────────────────────────────
        def token_chip(text, col, bg_col=PANEL, extra_w=0.0):
            w = len(text) * 0.165 + 0.5 + extra_w
            box = RoundedRectangle(
                corner_radius=0.1, width=w, height=0.44,
                fill_color=bg_col, fill_opacity=1,
                stroke_color=col, stroke_width=1.8
            )
            lbl = Text(text, font="Consolas", font_size=16, color=col)
            lbl.move_to(box)
            return VGroup(box, lbl)
 
        def glue_dot():
            d = Dot(radius=0.07, color=MUTED)
            return d
 
        # ── Example word ──────────────────────────────────────────────────────
        ex_lbl = Text('Input word:  "Tinatawagan"  (capitalized)',
                      font="Consolas", font_size=16, color=YELLOW)
        ex_lbl.next_to(purpose, DOWN, buff=0.30).to_edge(LEFT, buff=0.5)
        self.play(FadeIn(ex_lbl))
        self.wait(0.3)
 
        # ── Step 1 – Capitalisation tracking ──────────────────────────────────
        step1_lbl = Text("Step 1 — Capital 'T' detected → lowercase + ##CAPITAL tag",
                         font="Consolas", font_size=13, color=MUTED)
        step1_lbl.next_to(ex_lbl, DOWN, buff=0.28).to_edge(LEFT, buff=0.5)
        self.play(FadeIn(step1_lbl))
 
        cap_chip = token_chip("##CAPITAL", ACCENT, "#0C2A4A")
        lower_chip = token_chip('"tinatawagan"', WHITE, PANEL)
 
        cap_row = VGroup(cap_chip, lower_chip)
        cap_row.arrange(RIGHT, buff=0.35)
        cap_row.next_to(step1_lbl, DOWN, buff=0.18).to_edge(LEFT, buff=0.5)
 
        arrow_cap = Arrow(
            cap_chip.get_right(), lower_chip.get_left(),
            buff=0.08, stroke_width=2, color=BORDER,
            max_tip_length_to_length_ratio=0.3
        )
        cap_note = Text("sticky note: first letter was uppercase",
                        font="Consolas", font_size=12, color=ACCENT)
        cap_note.next_to(cap_chip, DOWN, buff=0.08).align_to(cap_chip, LEFT)
 
        self.play(FadeIn(cap_chip, shift=UP * 0.1), run_time=0.4)
        self.play(FadeIn(cap_note), run_time=0.3)
        self.play(GrowArrow(arrow_cap), FadeIn(lower_chip), run_time=0.5)
        self.wait(0.5)
 
        # ── Step 2 – Pieces from morph tagging ────────────────────────────────
        step2_lbl = Text("Step 2 — Pieces from Phase 3 (morph tags + BPE root)",
                         font="Consolas", font_size=13, color=MUTED)
        step2_lbl.next_to(cap_row, DOWN, buff=0.30).to_edge(LEFT, buff=0.5)
        self.play(FadeIn(step2_lbl))
 
        p_root   = token_chip("[_tawag]",  SUCCESS, "#0A2010")
        p_redup  = token_chip("##REDUP",  PURPLE, PANEL)
        p_infix  = token_chip("in##INFIX",  ORANGE, PANEL)
        p_suffix = token_chip("an##SUFFIX", YELLOW, PANEL)
        p_cap = token_chip("##CAPITAL", ACCENT, "#0C2A4A")
 
        pieces = VGroup(p_root, p_redup, p_infix, p_suffix, p_cap)
        pieces.arrange(RIGHT, buff=0.22)
        pieces.next_to(step2_lbl, DOWN, buff=0.16).to_edge(LEFT, buff=0.5)
 
        for p in pieces:
            self.play(FadeIn(p, scale=0.9), run_time=0.3)
        self.wait(0.4)
 
        # ── Step 3 – Invisible glue binding ───────────────────────────────────
        step3_lbl = Text(
            "Step 3 — Bind with invisible non-printable Unicode 🔗  (O(1) lookup)",
            font="Consolas", font_size=13, color=MUTED
        )
        step3_lbl.next_to(pieces, DOWN, buff=0.32).to_edge(LEFT, buff=0.5)
        self.play(FadeIn(step3_lbl))
        self.wait(0.2)
 
        # Show glue dots appearing between each chip in the final sequence
        glue_note = Text(
            "🔗 = hidden barcode — computer knows it's a grammar tag, not plain text",
            font="Consolas", font_size=12, color=MUTED
        )
        glue_note.next_to(step3_lbl, DOWN, buff=0.08).to_edge(LEFT, buff=0.7)
        self.play(FadeIn(glue_note), run_time=0.4)
        self.wait(0.4)
 
        # ── Final assembled sequence ───────────────────────────────────────────
        final_lbl = Text("Final assembled sequence:", font="Consolas",
                         font_size=14, color=MUTED)
        final_lbl.next_to(glue_note, DOWN, buff=0.32).to_edge(LEFT, buff=0.5)
        self.play(FadeIn(final_lbl))
 
        # Build final chips: ##CAPITAL  🔗  ##INFIX  🔗  [tawag]  🔗  ##SUFFIX
        f_cap = token_chip("##CAPITAL", ACCENT,   "#0C2A4A")
        f_redup = token_chip("##REDUP", PURPLE, PANEL)
        f_infix = token_chip("in##INFIX",   ORANGE,   PANEL)
        f_root = token_chip("[_tawag]",   SUCCESS,  "#0A2010")
        f_suffix = token_chip("an##SUFFIX",  YELLOW,   PANEL)
 
        final_tokens = [f_cap, f_redup, f_infix, f_root, f_suffix]
 
        # Interleave with glue dots
        seq_group = VGroup()
        for i, ft in enumerate(final_tokens):
            seq_group.add(ft)
            if i < len(final_tokens) - 1:
                g = glue_dot()
                seq_group.add(g)
 
        seq_group.arrange(RIGHT, buff=0.14)
        seq_group.next_to(final_lbl, RIGHT, buff=0.3)
 
        # Animate chips + dots appearing in order
        for mob in seq_group:
            if isinstance(mob, Dot):
                self.play(FadeIn(mob), run_time=0.2)
            else:
                self.play(FadeIn(mob, scale=0.9), run_time=0.4)
        self.wait(0.5)
 
        # ── Takeaway ──────────────────────────────────────────────────────────
        takeaway = Text(
            "Order preserved · Capital restored · Tags unambiguous → LLM-ready",
            font="Consolas", font_size=14, color=SUCCESS
        )
        takeaway.next_to(seq_group, DOWN, buff=0.32).to_edge(LEFT, buff=0.5)
        self.play(FadeIn(takeaway, shift=UP * 0.1))
        self.wait(1.2)
 
        # ── Cleanup + advance bar ─────────────────────────────────────────────
        content = VGroup(
            heading, purpose, ex_lbl,
            step1_lbl, cap_chip, arrow_cap, lower_chip, cap_note,
            step2_lbl, pieces,
            step3_lbl, glue_note,
            final_lbl, seq_group,
            takeaway,
        )
        self.play(FadeOut(content))
 
        bar_done = pipeline_bar(active=4, done={0, 1, 2, 3})
        div_done = Line(LEFT * 6.8, RIGHT * 6.8, stroke_color=BORDER, stroke_width=1
                        ).next_to(bar_done, DOWN, buff=0.18)
        self.play(Transform(bar, bar_done), Transform(div, div_done), run_time=0.5)
        self.wait(0.8)



# ── Combined render target ────────────────────────────────────────────────────
class FullPipeline(Scene):
    def construct(self):
        PipelineOverview.construct(self)
        Normalization.construct(self)
        Caching.construct(self)
        MorphTagging.construct(self)

class CachingOnly(Scene):
    def construct(self):
        bar = pipeline_bar(active=1, done={0})
        div = Line(LEFT*6.8, RIGHT*6.8, stroke_color=BORDER, stroke_width=1
                   ).next_to(bar, DOWN, buff=0.18)
        self.add(bar, div)
        Caching.construct(self)

class MorphTaggingOnly(Scene):
    def construct(self):
        bar = pipeline_bar(active=2, done={0, 1})
        div = Line(LEFT*6.8, RIGHT*6.8, stroke_color=BORDER, stroke_width=1
                   ).next_to(bar, DOWN, buff=0.18)
        self.add(bar, div)
        MorphTagging.construct(self)

class SeqAssemblyOnly(Scene):
    def construct(self):
        bar = pipeline_bar(active=2, done={0, 1})
        div = Line(LEFT*6.8, RIGHT*6.8, stroke_color=BORDER, stroke_width=1
                   ).next_to(bar, DOWN, buff=0.18)
        self.add(bar, div)
        SequenceAssembly.construct(self)