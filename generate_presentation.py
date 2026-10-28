import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ------------------------------------------------------------------------------
# COLOR PALETTE (Purple & Blue Professional Academic Theme)
# ------------------------------------------------------------------------------
COLOR_PRIMARY_PURPLE = RGBColor(109, 40, 217)   # Deep Violet #6D28D9
COLOR_ACCENT_PURPLE  = RGBColor(139, 92, 246)   # Bright Violet #8B5CF6
COLOR_PRIMARY_BLUE   = RGBColor(29, 78, 216)    # Deep Royal Blue #1D4ED8
COLOR_ACCENT_BLUE    = RGBColor(37, 99, 235)    # Electric Blue #2563EB
COLOR_DARK_TEXT      = RGBColor(15, 23, 42)     # Slate 900 #0F172A
COLOR_MUTED_TEXT     = RGBColor(71, 85, 105)    # Slate 600 #475569
COLOR_CARD_BG        = RGBColor(248, 250, 252)  # Slate 50 #F8FAFC
COLOR_CARD_BORDER    = RGBColor(226, 232, 240)  # Slate 200 #E2E8F0
COLOR_WHITE          = RGBColor(255, 255, 255)
COLOR_SUCCESS_GREEN  = RGBColor(22, 163, 74)    # Emerald 600

def create_deck():
    prs = Presentation()
    # 16:9 Widescreen standard
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def set_speaker_notes(slide, notes_text):
        if notes_text:
            notes_slide = slide.notes_slide
            text_frame = notes_slide.notes_text_frame
            text_frame.text = notes_text

    def add_header(slide, title, category="QUANTUM REINFORCEMENT LEARNING • PROOF OF CONCEPT"):
        # Top banner background stripe
        banner = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.1))
        banner.fill.solid()
        banner.fill.fore_color.rgb = COLOR_WHITE
        banner.line.color.rgb = COLOR_CARD_BORDER
        banner.line.width = Pt(1)

        # Category pill/badge
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.12), Inches(10), Inches(0.3))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.name = "Calibri"
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = COLOR_ACCENT_PURPLE

        # Main Slide Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.38), Inches(11.5), Inches(0.6))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title
        p_title.font.name = "Calibri"
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_DARK_TEXT

    def add_card(slide, left, top, width, height, bg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1)
        return card

    def add_bullet_list(tf, items, font_size=15, bullet_char="▪"):
        for i, item in enumerate(items):
            p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
            p.text = f"{bullet_char}  {item}"
            p.font.name = "Calibri"
            p.font.size = Pt(font_size)
            p.font.color.rgb = COLOR_DARK_TEXT
            p.space_after = Pt(10)

    def add_table(slide, left, top, width, height, headers, rows):
        shape = slide.shapes.add_table(len(rows) + 1, len(headers), left, top, width, height)
        table = shape.table

        for j, h in enumerate(headers):
            cell = table.cell(0, j)
            cell.text = str(h)
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_PRIMARY_PURPLE
            p = cell.text_frame.paragraphs[0]
            p.font.name = "Calibri"
            p.font.bold = True
            p.font.size = Pt(13)
            p.font.color.rgb = COLOR_WHITE
            p.alignment = PP_ALIGN.CENTER

        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                cell = table.cell(i + 1, j)
                cell.text = str(val)
                cell.fill.solid()
                cell.fill.fore_color.rgb = COLOR_WHITE if i % 2 == 0 else COLOR_CARD_BG
                p = cell.text_frame.paragraphs[0]
                p.font.name = "Calibri"
                p.font.size = Pt(12)
                p.font.color.rgb = COLOR_DARK_TEXT
                p.alignment = PP_ALIGN.CENTER
        return shape

    # ==========================================================================
    # SLIDE 1: Title Slide
    # ==========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = RGBColor(15, 23, 42) # Dark theme for title
    bg1.line.fill.background()

    # Title box
    tb1 = s1.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(11.0), Inches(2.2))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Quantum-Enhanced Reinforcement Learning for Strategic Decision-Making"
    p1.font.name = "Calibri"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_WHITE

    p1_sub = tf1.add_paragraph()
    p1_sub.text = "Proof of Concept: Hybrid Quantum-Classical Policy in a Strategic Dice Game"
    p1_sub.font.name = "Calibri"
    p1_sub.font.size = Pt(20)
    p1_sub.font.color.rgb = COLOR_ACCENT_PURPLE
    p1_sub.space_before = Pt(14)

    # Bullet container
    card1 = add_card(s1, Inches(1.2), Inches(4.3), Inches(10.9), Inches(2.2),
                     bg_color=RGBColor(30, 41, 59), border_color=RGBColor(51, 65, 85))
    tb_b = s1.shapes.add_textbox(Inches(1.5), Inches(4.5), Inches(10.3), Inches(1.8))
    add_bullet_list(tb_b.text_frame, [
        "End-to-end implementation of game environment, classical baselines, and hybrid PQC agents.",
        "Systematic comparison of policy sizes (389 vs 1,825 vs 5,697 parameters).",
        "Ablation studies isolating the roles of CNOT entanglement and learned rotation angles.",
        "Characterization of parameter efficiency vs. sample efficiency vs. simulation overhead."
    ], font_size=14)
    for p in tb_b.text_frame.paragraphs:
        p.font.color.rgb = RGBColor(226, 232, 240)

    set_speaker_notes(s1, "Cieľom tejto časti bolo vytvoriť kontrolovateľné prostredie, v ktorom sa dá hybridný model implementovať, overiť a porovnať s klasickým riešením. Kocková hra slúži ako proof of concept pred presunom k aplikačne relevantnejšej úlohe.")

    # ==========================================================================
    # SLIDE 2: Research Questions
    # ==========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Core Research Questions")
    
    questions = [
        ("1. Feasibility", "Can a hybrid quantum-classical policy reliably navigate the full sequential decision space of a competitive game?"),
        ("2. Parameter Efficiency", "Can parameterized quantum circuits achieve competitive win rates with significantly fewer trainable parameters?"),
        ("3. Sample Efficiency", "Does quantum representation require fewer environment interactions (matches/decisions) to converge?"),
        ("4. Role of Quantum Components", "Do CNOT entanglement gates and learned circuit rotations provide a demonstrable functional advantage?"),
        ("5. Computational Overhead", "What is the wall-clock execution penalty on classical CPU simulators vs standard MLPs?")
    ]
    for idx, (q_title, q_desc) in enumerate(questions):
        y_pos = Inches(1.4 + idx * 1.05)
        add_card(s2, Inches(1.0), y_pos, Inches(11.3), Inches(0.9))
        tb = s2.shapes.add_textbox(Inches(1.2), y_pos + Inches(0.1), Inches(10.9), Inches(0.7))
        tf = tb.text_frame
        p = tf.paragraphs[0]
        p.text = f"{q_title}: "
        p.font.bold = True
        p.font.size = Pt(16)
        p.font.color.rgb = COLOR_PRIMARY_PURPLE
        run = p.add_run()
        run.text = q_desc
        run.font.bold = False
        run.font.color.rgb = COLOR_DARK_TEXT
        run.font.size = Pt(15)

    # Highlight box at bottom
    add_card(s2, Inches(1.0), Inches(6.6), Inches(11.3), Inches(0.6), bg_color=RGBColor(243, 232, 255), border_color=COLOR_ACCENT_PURPLE)
    tb_w = s2.shapes.add_textbox(Inches(1.2), Inches(6.65), Inches(10.9), Inches(0.5))
    pw = tb_w.text_frame.paragraphs[0]
    pw.text = "Crucial distinction: Parameter efficiency ≠ Sample efficiency ≠ Computational acceleration."
    pw.font.bold = True
    pw.font.size = Pt(13)
    pw.font.color.rgb = COLOR_PRIMARY_PURPLE
    pw.alignment = PP_ALIGN.CENTER

    set_speaker_notes(s2, "Tieto otázky treba hodnotiť oddelene. Malý model nemusí potrebovať menej skúseností a menší počet parametrov nemusí znamenať rýchlejší tréning.")

    # ==========================================================================
    # SLIDE 3: Environment & Rules
    # ==========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Environment: Strategic Multi-Dice Game")
    
    # Left Card: Rules
    add_card(s3, Inches(1.0), Inches(1.4), Inches(6.0), Inches(5.6))
    tb_l = s3.shapes.add_textbox(Inches(1.2), Inches(1.6), Inches(5.6), Inches(5.2))
    tf_l = tb_l.text_frame
    p_lt = tf_l.paragraphs[0]
    p_lt.text = "Game Dynamics & Risk Trade-offs"
    p_lt.font.bold = True
    p_lt.font.size = Pt(18)
    p_lt.font.color.rgb = COLOR_PRIMARY_BLUE
    p_lt.space_after = Pt(14)
    add_bullet_list(tf_l, [
        "2 players, 6 fair six-sided dice, target goal: 4,000 points.",
        "Each roll yields scoring subsets (singles, triplets, straights).",
        "Strategic decision branch at each legal scoring subset:\n   a) Bank current accumulated turn points and pass.\n   b) Continue rolling remaining unbanked dice.",
        "Hot Dice rule: Scoring all 6 dice restores all 6 dice to roll.",
        "Farkle penalty: Rolling zero scoring dice forfeits all turn points.",
        "Core challenge: Balances greedy short-term point gain vs. catastrophic risk of busting."
    ], font_size=13)

    # Right Card: Scoring Table
    add_card(s3, Inches(7.3), Inches(1.4), Inches(5.0), Inches(5.6))
    tb_rt = s3.shapes.add_textbox(Inches(7.5), Inches(1.6), Inches(4.6), Inches(0.5))
    prt = tb_rt.text_frame.paragraphs[0]
    prt.text = "Scoring Combinations Reference"
    prt.font.bold = True
    prt.font.size = Pt(18)
    prt.font.color.rgb = COLOR_PRIMARY_PURPLE

    headers_score = ["Selection", "Points"]
    rows_score = [
        ["Single 1", "100 pts"],
        ["Single 5", "50 pts"],
        ["Three 1s (111)", "1,000 pts"],
        ["Three of a kind (222 - 666)", "200 - 600 pts"],
        ["Small straight (2-3-4-5-6)", "750 pts"],
        ["Full straight (1-2-3-4-5-6)", "1,500 pts"]
    ]
    add_table(s3, Inches(7.5), Inches(2.3), Inches(4.6), Inches(4.2), headers_score, rows_score)

    set_speaker_notes(s3, "Agent nerozhoduje iba o tom, či pokračovať. Vyberá aj konkrétne kocky, ktoré odloží. Môže teda získať menej okamžitých bodov výmenou za iný počet kociek pri ďalšom hode.")

    # ==========================================================================
    # SLIDE 4: RL Formulation
    # ==========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "RL Formulation & Policy Representation")

    # Left card: Policy architecture
    add_card(s4, Inches(1.0), Inches(1.4), Inches(5.6), Inches(5.6))
    tb4_l = s4.shapes.add_textbox(Inches(1.2), Inches(1.6), Inches(5.2), Inches(5.2))
    tf4_l = tb4_l.text_frame
    p4_1 = tf4_l.paragraphs[0]
    p4_1.text = "State-Action Scoring Architecture"
    p4_1.font.bold = True
    p4_1.font.size = Pt(17)
    p4_1.font.color.rgb = COLOR_PRIMARY_BLUE
    p4_1.space_after = Pt(10)
    add_bullet_list(tf4_l, [
        "Variable Action Spaces: Legal dice subsets change dynamically at every throw.",
        "Candidate-Scoring Mechanism: Policy evaluates candidate pairs f_θ(s, a) ∈ ℝ rather than a fixed categorical output head.",
        "Normalized Action Selection:\nπ_θ(a | s) = exp(f_θ(s, a)) / ∑ exp(f_θ(s, a'))",
        "Inference: Greedy selection argmax f_θ(s, a).",
        "Exploration: Categorical sampling from softmax during training.",
        "Key insight: Selection probability π(a|s) reflects relative preference, not final win probability."
    ], font_size=13)

    # Right card: 22 Features breakdown
    add_card(s4, Inches(6.9), Inches(1.4), Inches(5.4), Inches(5.6))
    tb4_r = s4.shapes.add_textbox(Inches(7.1), Inches(1.6), Inches(5.0), Inches(5.2))
    tf4_r = tb4_r.text_frame
    p4_2 = tf4_r.paragraphs[0]
    p4_2.text = "22 State-Action Input Features"
    p4_2.font.bold = True
    p4_2.font.size = Pt(17)
    p4_2.font.color.rgb = COLOR_PRIMARY_PURPLE
    p4_2.space_after = Pt(10)
    add_bullet_list(tf4_r, [
        "Global Match Context:\n  • Normalized player score\n  • Opponent score\n  • Accumulated turn points",
        "Action Specification:\n  • Points gained by proposed subset\n  • Number of dice locked\n  • Remaining dice to roll if continuing",
        "Decision Type Flag:\n  • Binary indicator: bank vs. continue",
        "Histogram Featurization (12 features):\n  • Exact counts of faces 1-6 rolled\n  • Exact counts of faces 1-6 chosen"
    ], font_size=13)

    set_speaker_notes(s4, "Počet legálnych akcií sa medzi stavmi mení. Preto sieť nemá pevný výstupný vektor všetkých možných akcií, ale rovnakou funkciou ohodnotí každého dostupného kandidáta.")

    # ==========================================================================
    # SLIDE 5: Training Algorithm
    # ==========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Monte Carlo REINFORCE with Learned Baseline")

    add_card(s5, Inches(1.0), Inches(1.4), Inches(11.3), Inches(5.6))
    tb5 = s5.shapes.add_textbox(Inches(1.3), Inches(1.6), Inches(10.7), Inches(5.2))
    tf5 = tb5.text_frame
    p5_h = tf5.paragraphs[0]
    p5_h.text = "Mathematical Formulation & Optimization"
    p5_h.font.bold = True
    p5_h.font.size = Pt(18)
    p5_h.font.color.rgb = COLOR_PRIMARY_PURPLE
    p5_h.space_after = Pt(12)

    add_bullet_list(tf5, [
        "Terminal Reward Signal: R = +1 for match win, -1 for loss (no intermediate handcrafted reward shaping).",
        "Advantage Formulation: A_t = R - V_φ(s_t), reducing trajectory-level variance.",
        "Policy Objective Function: L_π = - ∑_t stopgrad(A_t) · log π_θ(a_t | s_t) - β · ∑_t H(π_θ(· | s_t)).",
        "Learned State Critic Network: V_φ(s) : 12 inputs → 64 → 32 → 1 scalar value (2,945 parameters).",
        "Critic Loss: Mean Squared Error L_V = (1/T) ∑_t (V_φ(s_t) - R)² directly targeting final game outcome.",
        "Deployment Independence: The critic is discarded after training; only policy f_θ is deployed."
    ], font_size=14)

    set_speaker_notes(s5, "Napriek označeniu kritik nejde o bootstrapovaný actor-critic. Hodnota sa učí priamo z konečného výsledku zápasu, nie z cieľa obsahujúceho odhad hodnoty nasledujúceho stavu.")

    # ==========================================================================
    # SLIDE 6: Experimental Roadmap
    # ==========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Experimental Roadmap & Methodology")

    headers_road = ["Phase", "Objective & Methodology", "Outcome / Milestone"]
    rows_road = [
        ["1. Environment & Opponents", "Develop game engine, heuristics (250-800 bank thresholds) and v2 reference bot.", "Reliable testbed with reproducible seeds."],
        ["2. Training Refinement", "Compare scalar moving average baseline vs. learned state-value critic baseline.", "Learned baseline established as standard."],
        ["3. Baseline Selection", "Explore classical MLP sizes (Large 5.7k, Medium 1.8k, Compact 657 params).", "Medium identified as minimal robust classical anchor."],
        ["4. 1st Hybrid Family", "Fair benchmark: 4-qubit circuit vs. 4x4x4 MLP replacement with identical encoder/readout.", "Hybrid matches Medium and beats direct MLP clone."],
        ["5. Compact Hybrid Family", "Introduce feature bypass and ZZ correlators; reduce policy to 389 parameters.", "Matches 1,825-param classical performance."],
        ["6. Sample Efficiency & Ablations", "Evaluate 5 intermediate checkpoints (500 to 10k); test frozen circuit weights.", "Isolates representational capacity vs. quantum learning."]
    ]
    add_table(s6, Inches(1.0), Inches(1.5), Inches(11.3), Inches(5.3), headers_road, rows_road)

    set_speaker_notes(s6, "Vývoj bol exploračný. Nie všetky historické výsledky vznikli rovnakým protokolom, preto ich nemožno bez označenia zlúčiť do jednej porovnávacej tabuľky.")

    # ==========================================================================
    # SLIDE 7: Historical Training Selection
    # ==========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Historical Exploration: Baseline Selection")

    add_card(s7, Inches(1.0), Inches(1.4), Inches(5.0), Inches(5.6))
    tb7_l = s7.shapes.add_textbox(Inches(1.2), Inches(1.6), Inches(4.6), Inches(5.2))
    tf7_l = tb7_l.text_frame
    p7 = tf7_l.paragraphs[0]
    p7.text = "Learned vs. Moving Baseline"
    p7.font.bold = True
    p7.font.size = Pt(17)
    p7.font.color.rgb = COLOR_PRIMARY_BLUE
    p7.space_after = Pt(10)
    add_bullet_list(tf7_l, [
        "Tested continued training from checkpoint v2 over 10,000 matches.",
        "Learned critic achieved +8.1 percentage points improvement against v2.",
        "Head-to-head match-up: Learned critic won 57.3% of direct games.",
        "Outcome: Learned baseline selected for all future comparative experiments."
    ], font_size=13)

    headers_crit = ["Seed", "Learned vs v2", "Scalar vs v2", "Direct Match"]
    rows_crit = [
        ["Seed 42", "59.2 %", "54.9 %", "54.1 %"],
        ["Seed 43", "58.4 %", "50.5 %", "59.6 %"],
        ["Seed 44", "63.0 %", "51.0 %", "58.2 %"],
        ["Average", "60.2 %", "52.1 %", "57.3 %"]
    ]
    add_table(s7, Inches(6.3), Inches(1.8), Inches(6.0), Inches(3.2), headers_crit, rows_crit)

    add_card(s7, Inches(6.3), Inches(5.3), Inches(6.0), Inches(1.7), bg_color=RGBColor(254, 242, 242), border_color=RGBColor(252, 165, 165))
    tb7_w = s7.shapes.add_textbox(Inches(6.5), Inches(5.4), Inches(5.6), Inches(1.5))
    p7_w = tb7_w.text_frame.paragraphs[0]
    p7_w.text = "Methodological Disclaimer:\nThese models were fine-tuned from pre-trained checkpoints, not initialized tabula rasa. Their absolute performance cannot be directly merged into the clean scratch-training benchmarks."
    p7_w.font.size = Pt(12)
    p7_w.font.color.rgb = RGBColor(153, 27, 27)

    set_speaker_notes(s7, "Tento experiment zdôvodňuje voľbu tréningového postupu. Jeho silné výsledky však nie sú rovnocenným architektonickým porovnaním s novými modelmi trénovanými od nuly.")

    # ==========================================================================
    # SLIDE 8: Classical Baseline Selection
    # ==========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Selection of the Classical Anchor Baseline")

    # Left: Table and points
    add_card(s8, Inches(0.8), Inches(1.4), Inches(5.6), Inches(5.6))
    tb8 = s8.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(5.2), Inches(2.2))
    tf8 = tb8.text_frame
    p8 = tf8.paragraphs[0]
    p8.text = "Scaling Down Classical MLPs"
    p8.font.bold = True
    p8.font.size = Pt(17)
    p8.font.color.rgb = COLOR_PRIMARY_BLUE
    p8.space_after = Pt(6)
    add_bullet_list(tf8, [
        "Medium eliminates 68% of parameters compared to Large without statistically significant performance degradation.",
        "Compact MLP (657 params) suffers catastrophic drop (39.8% win rate).",
        "Medium (1,825 params) serves as the primary rigorous classical reference."
    ], font_size=12)

    headers_b = ["Model", "Architecture", "Params", "Win Rate vs v2"]
    rows_b = [
        ["Large", "22 → 64 → 64 → 1", "5,697", "52.2 ± 1.9 %"],
        ["Medium", "22 → 32 → 32 → 1", "1,825", "51.9 ± 1.5 %"],
        ["Compact MLP", "22 → 16 → 16 → 1", "657", "39.8 ± 2.8 %"]
    ]
    add_table(s8, Inches(1.0), Inches(3.9), Inches(5.2), Inches(2.8), headers_b, rows_b)

    # Right: Embed architecture_comparison.png or equal_depth_comparison.png
    img_path = "architecture_comparison.png"
    if not os.path.exists(img_path):
        img_path = "equal_depth_comparison.png"
    if os.path.exists(img_path):
        s8.shapes.add_picture(img_path, Inches(6.7), Inches(1.6), width=Inches(5.8))
    else:
        add_card(s8, Inches(6.7), Inches(1.6), Inches(5.8), Inches(5.2))
        tb_img = s8.shapes.add_textbox(Inches(7.0), Inches(3.5), Inches(5.2), Inches(1.0))
        tb_img.text_frame.text = "[Insert architecture_comparison.png]"

    set_speaker_notes(s8, "Výber baseline nebol založený iba na jednej veľkej sieti. Už klasickým zmenšením sa podarilo odstrániť veľkú časť parametrov. Preto neskoršiu úsporu hybridu primárne porovnávame s medium.")

    # ==========================================================================
    # SLIDE 9: Common Comparative Protocol
    # ==========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Controlled Experimental Protocol")

    add_card(s9, Inches(1.0), Inches(1.4), Inches(5.5), Inches(5.6))
    tb9_l = s9.shapes.add_textbox(Inches(1.2), Inches(1.6), Inches(5.1), Inches(5.2))
    tf9_l = tb9_l.text_frame
    p9_l = tf9_l.paragraphs[0]
    p9_l.text = "Training Protocol (Identical for All)"
    p9_l.font.bold = True
    p9_l.font.size = Pt(17)
    p9_l.font.color.rgb = COLOR_PRIMARY_PURPLE
    p9_l.space_after = Pt(10)
    add_bullet_list(tf9_l, [
        "Strict initialization tabula rasa from uniform random seeds (42, 43, 44).",
        "Identical input representation: 22 normalized state-action features.",
        "Shared critic architecture: 12 → 64 → 32 → 1 MLP.",
        "Training budget: Exactly 10,000 matches per seed.",
        "Curriculum opponent mix: v2 bot + heuristic thresholds 250, 500, 800.",
        "Action selection: Softmax sampling during training; strategic filter disabled."
    ], font_size=13)

    add_card(s9, Inches(6.8), Inches(1.4), Inches(5.5), Inches(5.6))
    tb9_r = s9.shapes.add_textbox(Inches(7.0), Inches(1.6), Inches(5.1), Inches(5.2))
    tf9_r = tb9_r.text_frame
    p9_r = tf9_r.paragraphs[0]
    p9_r.text = "Evaluation Standard"
    p9_r.font.bold = True
    p9_r.font.size = Pt(17)
    p9_r.font.color.rgb = COLOR_PRIMARY_BLUE
    p9_r.space_after = Pt(10)
    add_bullet_list(tf9_r, [
        "Greedy policy deployment: argmax candidate logit.",
        "1,000 matches per benchmark opponent.",
        "Symmetric evaluation: Exactly 500 games played as 1st player, 500 games as 2nd player.",
        "Benchmarked against: Reference v2, Bank250, Bank500, Bank650, Bank800.",
        "Single-threaded execution on standard CPU hardware.",
        "Critical detail: Trajectories diverge organically across seeds as policies make distinct choices."
    ], font_size=13)

    set_speaker_notes(s9, "Pri rovnakom tréningovom seede majú modely rovnaký rozpis súperov a začínajúcich hráčov. Nemajú však rovnaké trajektórie: rozdielne rozhodnutia vedú k rozdielnym stavom a dĺžkam zápasov.")

    # ==========================================================================
    # SLIDE 10: Quantum Circuit Block
    # ==========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Parameterized Quantum Circuit (PQC) Architecture")

    add_card(s10, Inches(1.0), Inches(1.4), Inches(6.0), Inches(5.6))
    tb10_l = s10.shapes.add_textbox(Inches(1.2), Inches(1.6), Inches(5.6), Inches(5.2))
    tf10_l = tb10_l.text_frame
    p10_1 = tf10_l.paragraphs[0]
    p10_1.text = "Circuit Specifications"
    p10_1.font.bold = True
    p10_1.font.size = Pt(17)
    p10_1.font.color.rgb = COLOR_PRIMARY_PURPLE
    p10_1.space_after = Pt(10)
    add_bullet_list(tf10_l, [
        "4 Qubits initialized in ground state |0000⟩.",
        "2 Variational Layers with repeated data re-uploading.",
        "Total 24 trainable rotation angles (θ ∈ ℝ²⁴).",
        "Layer Pipeline:\n   1. Input Encoding: R_y(π · u_i) where u ∈ ℝ⁴.\n   2. Trainable single-qubit rotations: R_y(θ_y) and R_z(θ_z).\n   3. Entanglement layer: Circular CNOT ring (0→1→2→3→0).\n   4. Trainable output rotation: R_x(θ_x).",
        "Exact expectation values ⟨Z_i⟩ evaluated via state-vector simulation (no shot noise)."
    ], font_size=13)

    add_card(s10, Inches(7.3), Inches(1.4), Inches(5.0), Inches(5.6))
    tb10_r = s10.shapes.add_textbox(Inches(7.5), Inches(1.6), Inches(4.6), Inches(5.2))
    tf10_r = tb10_r.text_frame
    p10_2 = tf10_r.paragraphs[0]
    p10_2.text = "Mathematical Characterization"
    p10_2.font.bold = True
    p10_2.font.size = Pt(17)
    p10_2.font.color.rgb = COLOR_PRIMARY_BLUE
    p10_2.space_after = Pt(10)
    add_bullet_list(tf10_r, [
        "Data re-uploading enables high-frequency Fourier components despite shallow depth.",
        "Linear Entanglement Topology: Captures pairwise qubit correlations efficiently.",
        "Analytic gradients computed via PyTorch autograd through state-vector amplitudes.",
        "Classical Simulation: Implemented via custom PyTorch complex128 tensors (no quantum hardware claims)."
    ], font_size=13)

    set_speaker_notes(s10, "Ide o malý parametrizovaný kvantový obvod, ktorý je klasicky simulovaný. Nepoužívame kvantový hardvér a výsledok nemá byť interpretovaný ako hardvérové kvantové zrýchlenie.")

    # ==========================================================================
    # SLIDE 11: First Hybrid Family
    # ==========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "First Hybrid Family: Controlled Architectural Test")

    add_card(s11, Inches(1.0), Inches(1.4), Inches(11.3), Inches(1.8))
    tb11_t = s11.shapes.add_textbox(Inches(1.2), Inches(1.5), Inches(10.9), Inches(1.5))
    tf11_t = tb11_t.text_frame
    p11_t = tf11_t.paragraphs[0]
    p11_t.text = "Shared Classical Trunk with Modular Core"
    p11_t.font.bold = True
    p11_t.font.size = Pt(16)
    p11_t.font.color.rgb = COLOR_PRIMARY_BLUE
    p11_t.space_after = Pt(6)
    add_bullet_list(tf11_t, [
        "Common Classical Encoder: 22 inputs → 32 hidden → 4 latent features (Tanh activation).",
        "Common Linear Readout Head: 4 inputs → 1 scalar logit.",
        "Direct substitution of the central 4-dimensional latent transformation to isolate quantum utility."
    ], font_size=13)

    headers_f1 = ["Model Name", "Internal Latent Block Structure", "Policy Parameters", "Quantum Params"]
    rows_f1 = [
        ["Medium (Classical Baseline)", "Full MLP: 22 → 32 → 32 → 1", "1,825", "0"],
        ["Direct Classical Replacement", "MLP Block: 4 → 4 → 4 (Tanh)", "913", "0"],
        ["Hybrid (with CNOT)", "4-Qubit PQC with Entanglement Ring", "897", "24"],
        ["Hybrid (without CNOT)", "4-Qubit PQC with Separable Rotations", "897", "24"]
    ]
    add_table(s11, Inches(1.0), Inches(3.6), Inches(11.3), Inches(3.2), headers_f1, rows_f1)

    set_speaker_notes(s11, "Klasický blok je kontrolná náhrada kvantového bloku pri zachovaní spoločného encoderu a readoutu. Nie je to však dôkaz, že reprezentuje všetky možné klasické alternatívy.")

    # ==========================================================================
    # SLIDE 12: Results: First Hybrid Family
    # ==========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Empirical Results: First Hybrid Family")

    headers_res1 = ["Opponent", "Medium (1,825)", "Classical (913)", "Hybrid w/ CNOT (897)", "Hybrid w/o CNOT (897)"]
    rows_res1 = [
        ["Reference v2", "51.9 ± 1.38 %", "34.7 ± 0.17 %", "52.8 ± 3.14 %", "52.2 ± 4.90 %"],
        ["Bank 250", "48.5 ± 0.50 %", "32.3 ± 0.00 %", "51.9 ± 2.35 %", "49.8 ± 6.41 %"],
        ["Bank 500", "54.7 ± 2.08 %", "43.0 ± 0.06 %", "56.3 ± 1.61 %", "55.4 ± 2.53 %"],
        ["Bank 650", "56.4 ± 1.70 %", "40.3 ± 0.06 %", "58.4 ± 1.71 %", "57.2 ± 2.97 %"],
        ["Bank 800", "64.8 ± 0.25 %", "51.0 ± 0.00 %", "65.6 ± 2.33 %", "64.7 ± 3.30 %"]
    ]
    add_table(s12, Inches(1.0), Inches(1.5), Inches(11.3), Inches(3.6), headers_res1, rows_res1)

    add_card(s12, Inches(1.0), Inches(5.4), Inches(11.3), Inches(1.6))
    tb12_b = s12.shapes.add_textbox(Inches(1.2), Inches(5.5), Inches(10.9), Inches(1.4))
    tf12_b = tb12_b.text_frame
    p12_bt = tf12_b.paragraphs[0]
    p12_bt.text = "Key Takeaways from First Family:"
    p12_bt.font.bold = True
    p12_bt.font.size = Pt(14)
    p12_bt.font.color.rgb = COLOR_PRIMARY_PURPLE
    add_bullet_list(tf12_b, [
        "Hybrids clearly outperformed the identical-size classical bottleneck replacement (52.8% vs 34.7% vs v2).",
        "Both hybrid models successfully matched the 1,825-parameter Medium baseline while cutting parameters by ~51%.",
        "Absence of Entanglement Advantage: No statistically significant difference between CNOT and non-CNOT variants."
    ], font_size=12)

    set_speaker_notes(s12, "Oba kvantové varianty výrazne prekonali túto konkrétnu klasickú náhradu. Rozdiely oproti medium sú menšie a tri seedy nepostačujú na silné všeobecné závery.")

    # ==========================================================================
    # SLIDE 13: Compact Hybrid Architecture
    # ==========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Compact Hybrid Architecture: 389 Parameters")

    # Left: Block Diagram description
    add_card(s13, Inches(1.0), Inches(1.4), Inches(5.5), Inches(5.6))
    tb13_l = s13.shapes.add_textbox(Inches(1.2), Inches(1.6), Inches(5.1), Inches(5.2))
    tf13_l = tb13_l.text_frame
    p13_1 = tf13_l.paragraphs[0]
    p13_1.text = "Architectural Pipeline"
    p13_1.font.bold = True
    p13_1.font.size = Pt(17)
    p13_1.font.color.rgb = COLOR_PRIMARY_BLUE
    p13_1.space_after = Pt(10)
    add_bullet_list(tf13_l, [
        "1. Compact Classical Encoder:\n   x (22) → h = tanh(W₁x + b₁) ∈ ℝ⁸",
        "2. Projection to Circuit:\n   u = tanh(W₂h + b₂) ∈ ℝ⁴",
        "3. Quantum Circuit Block:\n   Evaluates 4 single-qubit Z expectations + 4 adjacent two-qubit correlators:\n   q(u) = (⟨Z₀⟩...⟨Z₃⟩, ⟨Z₀Z₁⟩...⟨Z₃Z₀⟩) ∈ ℝ⁸",
        "4. Classical Bypass & Concatenation:\n   Merges encoder features with quantum features:\n   [h; q(u)] ∈ ℝ¹⁶",
        "5. Final Readout Head:\n   16 → 8 → 1 logit output (389 total parameters)."
    ], font_size=13)

    # Right: Parameter breakdown table
    add_card(s13, Inches(6.8), Inches(1.4), Inches(5.5), Inches(5.6))
    tb13_r = s13.shapes.add_textbox(Inches(7.0), Inches(1.6), Inches(5.1), Inches(1.0))
    p13_2 = tb13_r.text_frame.paragraphs[0]
    p13_2.text = "Parameter Allocation"
    p13_2.font.bold = True
    p13_2.font.size = Pt(17)
    p13_2.font.color.rgb = COLOR_PRIMARY_PURPLE

    headers_comp = ["Layer Component", "Input → Output", "Param Count"]
    rows_comp = [
        ["Classical Encoder", "22 → 8", "184"],
        ["Circuit Projection", "8 → 4", "36"],
        ["Trainable Quantum Rotations", "4 Qubits, 2 Layers", "24"],
        ["Readout Layer 1", "16 → 8", "136"],
        ["Output Linear Head", "8 → 1", "9"],
        ["Total Policy Parameters", "-", "389"]
    ]
    add_table(s13, Inches(7.0), Inches(2.4), Inches(5.1), Inches(4.2), headers_comp, rows_comp)

    set_speaker_notes(s13, "Klasická obchádzka umožňuje readoutu používať priamo osem príznakov encoderu aj osem výstupov kvantového bloku. Oproti prvej rodine sa zmenilo viac častí naraz, preto prípadné zlepšenie nemožno pripísať iba korelačným meraniam.")

    # ==========================================================================
    # SLIDE 14: Results: Compact Family
    # ==========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Empirical Results: Compact Family (389 Params)")

    headers_cres = ["Opponent", "Compact Quantum (389)", "Compact w/o CNOT (389)", "Compact Classical (405)"]
    rows_cres = [
        ["Reference v2", "54.9 ± 2.81 %", "53.8 ± 3.09 %", "38.8 ± 6.78 %"],
        ["Bank 250", "55.5 ± 2.46 %", "55.6 ± 2.34 %", "36.7 ± 7.11 %"],
        ["Bank 500", "60.6 ± 2.42 %", "59.0 ± 1.39 %", "45.7 ± 4.42 %"],
        ["Bank 650", "60.2 ± 3.03 %", "59.2 ± 3.16 %", "43.1 ± 4.82 %"],
        ["Bank 800", "67.6 ± 0.26 %", "67.4 ± 2.31 %", "53.9 ± 4.91 %"]
    ]
    add_table(s14, Inches(1.0), Inches(1.5), Inches(11.3), Inches(3.6), headers_cres, rows_cres)

    add_card(s14, Inches(1.0), Inches(5.4), Inches(11.3), Inches(1.6))
    tb14_b = s14.shapes.add_textbox(Inches(1.2), Inches(5.5), Inches(10.9), Inches(1.4))
    tf14_b = tb14_b.text_frame
    p14_bt = tf14_b.paragraphs[0]
    p14_bt.text = "Key Findings:"
    p14_bt.font.bold = True
    p14_bt.font.size = Pt(14)
    p14_bt.font.color.rgb = COLOR_PRIMARY_PURPLE
    add_bullet_list(tf14_b, [
        "Compact Quantum policies achieved superior performance across all opponents with only 389 parameters.",
        "Classical counterpart with identical bypass (405 params) underperformed by >16 percentage points.",
        "CNOT entanglement again provided no noticeable win rate benefit over separable single-qubit rotations."
    ], font_size=12)

    set_speaker_notes(s14, "Klasický variant používa rovnaký encoder, obchádzku a readout. Rozdiel je vo vnútornom bloku, ktorý vytvára štyri hodnoty a ich susedné súčiny.")

    # ==========================================================================
    # SLIDE 15: Parameter Efficiency Analysis
    # ==========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    add_header(s15, "Parameter Efficiency Comparison")

    headers_p = ["Architecture", "Total Parameters", "Ratio vs. Compact Quantum", "Win Rate vs v2"]
    rows_p = [
        ["Original v2 / Large MLP", "5,697", "14.65× more", "52.2 ± 1.9 %"],
        ["Primary Medium Baseline", "1,825", "4.69× more", "51.9 ± 1.5 %"],
        ["Compact Classical MLP", "405", "1.04× more", "38.8 ± 6.8 %"],
        ["Compact Quantum Policy", "389", "1.00× (Anchor)", "54.9 ± 2.8 %"]
    ]
    add_table(s15, Inches(1.0), Inches(1.6), Inches(11.3), Inches(3.2), headers_p, rows_p)

    add_card(s15, Inches(1.0), Inches(5.1), Inches(11.3), Inches(1.9))
    tb15_b = s15.shapes.add_textbox(Inches(1.2), Inches(5.2), Inches(10.9), Inches(1.7))
    tf15_b = tb15_b.text_frame
    p15_b = tf15_b.paragraphs[0]
    p15_b.text = "Critical Evaluation of Parameter Savings:"
    p15_b.font.bold = True
    p15_b.font.size = Pt(15)
    p15_b.font.color.rgb = COLOR_PRIMARY_PURPLE
    p15_b.space_after = Pt(6)
    add_bullet_list(tf15_b, [
        "78.7% parameter reduction compared to the Medium baseline (389 vs 1,825).",
        "93.2% parameter reduction compared to the Large reference architecture (389 vs 5,697).",
        "Scope clarification: Savings strictly apply to the deployed decision policy; critic (2,945 params) was identical during training across all runs."
    ], font_size=13)

    set_speaker_notes(s15, "Voči pôvodnej veľkej architektúre možno hovoriť o menej než desatine parametrov. Podstatnejšie je však porovnanie s medium, pretože aj klasická optimalizácia už veľkosť výrazne znížila. Kritik je spoločný tréningový komponent a pri nasadení sa nepoužíva.")

    # ==========================================================================
    # SLIDE 16: Sample Efficiency Setup
    # ==========================================================================
    s16 = prs.slides.add_slide(blank_layout)
    add_header(s16, "Sample Efficiency: Checkpoint Evaluation Setup")

    add_card(s16, Inches(1.0), Inches(1.4), Inches(5.5), Inches(5.6))
    tb16_l = s16.shapes.add_textbox(Inches(1.2), Inches(1.6), Inches(5.1), Inches(5.2))
    tf16_l = tb16_l.text_frame
    p16_l = tf16_l.paragraphs[0]
    p16_l.text = "Checkpointing Methodology"
    p16_l.font.bold = True
    p16_l.font.size = Pt(17)
    p16_l.font.color.rgb = COLOR_PRIMARY_BLUE
    p16_l.space_after = Pt(10)
    add_bullet_list(tf16_l, [
        "Single continuous training run per seed (up to 10,000 matches).",
        "Snapshots saved at 5 strategic checkpoints:\n  • 500 matches\n  • 1,000 matches\n  • 2,000 matches\n  • 5,000 matches\n  • 10,000 matches (final)",
        "Zero data leakage: Checkpoints were strictly evaluated offline after training completion (not used for early stopping)."
    ], font_size=13)

    headers_evalm = ["Model Variant", "Total Policy Params", "Trainable Params"]
    rows_evalm = [
        ["Medium Classical", "1,825", "1,825"],
        ["Compact Quantum", "389", "389"],
        ["Compact Frozen PQC", "389", "365 (24 frozen)"]
    ]
    add_table(s16, Inches(6.8), Inches(1.8), Inches(5.5), Inches(2.6), headers_evalm, rows_evalm)

    add_card(s16, Inches(6.8), Inches(4.7), Inches(5.5), Inches(2.3))
    tb16_rb = s16.shapes.add_textbox(Inches(7.0), Inches(4.8), Inches(5.1), Inches(2.0))
    tf16_rb = tb16_rb.text_frame
    p16_rb = tf16_rb.paragraphs[0]
    p16_rb.text = "Tracking Multi-Faceted Costs:"
    p16_rb.font.bold = True
    p16_rb.font.size = Pt(15)
    p16_rb.font.color.rgb = COLOR_PRIMARY_PURPLE
    p16_rb.space_after = Pt(6)
    add_bullet_list(tf16_rb, [
        "1. Training Matches: Standard episode count.",
        "2. Candidate Decisions: Total turns agent decided.",
        "3. Evaluated State-Action Pairs: Number of forward-passes.",
        "4. Wall-clock Execution Time: Elapsed CPU seconds."
    ], font_size=12)

    set_speaker_notes(s16, "Nešlo o päť nezávislých tréningov. Ide o päť bodov tej istej tréningovej trajektórie. Checkpointy sa nepoužívali na predčasné zastavenie ani výber najlepšieho modelu.")

    # ==========================================================================
    # SLIDE 17: Learning Curve by Matches (Embed mean_learning_curve.png)
    # ==========================================================================
    s17 = prs.slides.add_slide(blank_layout)
    add_header(s17, "Performance Evolution Across Training Matches")

    # Left: Image
    img_mean = "mean_learning_curve.png"
    if os.path.exists(img_mean):
        s17.shapes.add_picture(img_mean, Inches(0.8), Inches(1.5), width=Inches(6.6))
    else:
        add_card(s17, Inches(0.8), Inches(1.5), Inches(6.6), Inches(5.4))
        tb = s17.shapes.add_textbox(Inches(1.0), Inches(3.5), Inches(6.2), Inches(1.0))
        tb.text_frame.text = "[Insert mean_learning_curve.png]"

    # Right: Analysis
    add_card(s17, Inches(7.7), Inches(1.5), Inches(4.8), Inches(5.5))
    tb17 = s17.shapes.add_textbox(Inches(7.9), Inches(1.7), Inches(4.4), Inches(5.1))
    tf17 = tb17.text_frame
    p17_t = tf17.paragraphs[0]
    p17_t.text = "Intermediate Budget Dynamics"
    p17_t.font.bold = True
    p17_t.font.size = Pt(17)
    p17_t.font.color.rgb = COLOR_PRIMARY_PURPLE
    p17_t.space_after = Pt(10)
    add_bullet_list(tf17, [
        "At 5,000 matches (mid-budget):\n  • Compact Quantum: 53.00 ± 2.31 %\n  • Compact Frozen: 52.27 ± 0.71 %\n  • Medium Classical: 34.80 ± 0.00 %",
        "Hybrids attained adult performance level 5,000 matches earlier than Medium.",
        "Early instability: High variance and performance dips observed at 1k-2k matches (Seed 43 collapse).",
        "Note on error bands: Shading represents ±1 SD across 3 seeds (not confidence intervals)."
    ], font_size=13)

    set_speaker_notes(s17, "Najzaujímavejší výsledok je pri rozpočte 5 000 zápasov. Zároveň by bolo zavádzajúce ukázať iba tento bod: pri 1 000 a 2 000 zápasoch sa objavili veľmi slabé checkpointy. Spojnice iba prepájajú uložené merania.")

    # ==========================================================================
    # SLIDE 18: Decision-Level Sample Efficiency
    # ==========================================================================
    s18 = prs.slides.add_slide(blank_layout)
    add_header(s18, "Interaction-Level Efficiency Comparison")

    headers_dec = ["Model & Checkpoint", "Win Rate vs v2", "Candidate Decisions", "Evaluated Pairs (s, a)"]
    rows_dec = [
        ["Medium (10,000 matches)", "51.93 %", "122,629", "783,289"],
        ["Compact Quantum (5,000 matches)", "53.00 %", "72,673 (~41% fewer)", "452,073 (~42% fewer)"],
        ["Compact Frozen (5,000 matches)", "52.27 %", "72,269 (~41% fewer)", "449,170 (~43% fewer)"]
    ]
    add_table(s18, Inches(1.0), Inches(1.6), Inches(11.3), Inches(2.8), headers_dec, rows_dec)

    add_card(s18, Inches(1.0), Inches(4.7), Inches(11.3), Inches(2.3))
    tb18 = s18.shapes.add_textbox(Inches(1.2), Inches(4.8), Inches(10.9), Inches(2.1))
    tf18 = tb18.text_frame
    p18 = tf18.paragraphs[0]
    p18.text = "Decision-Level Granularity vs. Episode Granularity"
    p18.font.bold = True
    p18.font.size = Pt(16)
    p18.font.color.rgb = COLOR_PRIMARY_BLUE
    p18.space_after = Pt(8)
    add_bullet_list(tf18, [
        "Accounting for variable game lengths: Comparing purely by match count can mask discrepancies caused by ultra-short or long games.",
        "Candidate decisions and forward evaluations confirm substantial sample savings: Hybrids achieved equal or superior win rates with ~41-43% fewer interactions.",
        "Caveat: This does not constitute an exact theoretical bound on minimum sample complexity."
    ], font_size=13)

    set_speaker_notes(s18, "Toto dopĺňa porovnanie podľa zápasov, pretože rôzne politiky vytvárajú rôzne dlhé zápasy. Stále nejde o presne nameraný minimálny počet skúseností potrebný na dosiahnutie cieľovej výhernosti.")

    # ==========================================================================
    # SLIDE 19: Learned vs Frozen Quantum Rotations
    # ==========================================================================
    s19 = prs.slides.add_slide(blank_layout)
    add_header(s19, "Ablation: Are Trainable Quantum Rotations Necessary?")

    add_card(s19, Inches(1.0), Inches(1.4), Inches(5.5), Inches(5.6))
    tb19_l = s19.shapes.add_textbox(Inches(1.2), Inches(1.6), Inches(5.1), Inches(5.2))
    tf19_l = tb19_l.text_frame
    p19_l = tf19_l.paragraphs[0]
    p19_l.text = "Frozen PQC Protocol"
    p19_l.font.bold = True
    p19_l.font.size = Pt(17)
    p19_l.font.color.rgb = COLOR_PRIMARY_PURPLE
    p19_l.space_after = Pt(10)
    add_bullet_list(tf19_l, [
        "24 Quantum rotation angles frozen at random initial values.",
        "Encoder, projection, and readout heads remained fully trainable.",
        "Backpropagation passed analytical gradients through the static circuit into the encoder.",
        "Weights verified explicitly unchanged throughout 10,000 matches."
    ], font_size=13)

    headers_froz = ["Model Variant", "Trainable Params", "Win Rate vs v2 (10k Checkpoint)"]
    rows_froz = [
        ["Compact Quantum (Learned)", "389", "54.90 ± 2.81 %"],
        ["Compact Frozen (Static PQC)", "365", "55.67 ± 3.59 %"]
    ]
    add_table(s19, Inches(6.8), Inches(1.8), Inches(5.5), Inches(2.2), headers_froz, rows_froz)

    add_card(s19, Inches(6.8), Inches(4.3), Inches(5.5), Inches(2.7), bg_color=RGBColor(245, 243, 255), border_color=COLOR_PRIMARY_PURPLE)
    tb19_rb = s19.shapes.add_textbox(Inches(7.0), Inches(4.5), Inches(5.1), Inches(2.3))
    tf19_rb = tb19_rb.text_frame
    p19_rb = tf19_rb.paragraphs[0]
    p19_rb.text = "Major Scientific Implication:"
    p19_rb.font.bold = True
    p19_rb.font.size = Pt(15)
    p19_rb.font.color.rgb = COLOR_PRIMARY_PURPLE
    p19_rb.space_after = Pt(6)
    add_bullet_list(tf19_rb, [
        "Zero statistical difference between updating rotation angles and leaving them frozen.",
        "The static circuit acts as a fixed high-dimensional non-linear reservoir / feature projection.",
        "The empirical success of the hybrid policy cannot be credited to active quantum parameter optimization."
    ], font_size=12)

    set_speaker_notes(s19, "Frozen neznamená konštantný výstup kvantového bloku. Menia sa jeho vstupy, pretože ich vytvára učený encoder. Rozdiel priemerov nie je dôkazom prevahy frozen variantu ani ekvivalencie oboch modelov.")

    # ==========================================================================
    # SLIDE 20: Instability Diagnostic
    # ==========================================================================
    s20 = prs.slides.add_slide(blank_layout)
    add_header(s20, "Diagnostic of Training Instability & Pathologies")

    headers_diag = ["Model & Checkpoint (Seed 43)", "Greedy Win Rate", "Sampled Win Rate", "Greedy Bank Action %"]
    rows_diag = [
        ["Quantum @ 1,000 matches", "2.7 %", "7.4 %", "1.45 %"],
        ["Quantum @ 2,000 matches", "3.8 %", "12.4 %", "~ 0.00 %"],
        ["Frozen @ 1,000 matches", "3.4 %", "8.4 %", "0.00 %"]
    ]
    add_table(s20, Inches(1.0), Inches(1.5), Inches(11.3), Inches(2.6), headers_diag, rows_diag)

    add_card(s20, Inches(1.0), Inches(4.4), Inches(11.3), Inches(2.6))
    tb20 = s20.shapes.add_textbox(Inches(1.2), Inches(4.5), Inches(10.9), Inches(2.4))
    tf20 = tb20.text_frame
    p20 = tf20.paragraphs[0]
    p20.text = "Pathology Analysis: Complete Aversion to Banking"
    p20.font.bold = True
    p20.font.size = Pt(16)
    p20.font.color.rgb = COLOR_PRIMARY_PURPLE
    p20.space_after = Pt(8)
    add_bullet_list(tf20, [
        "Reproducibility verification: Checkpoints independently re-evaluated; results matched raw logs exactly.",
        "Behavioral pathology: At checkpoints 1k-2k, Seed 43 policies completely stopped banking points (greedy bank frequency collapsed to 0%), rolling indefinitely until busting.",
        "Argmax effect: Stochastic sampling recovered partial performance (up to 12.4%), but underlying policy was fundamentally damaged.",
        "Self-recovery: All collapsed seeds naturally corrected this pathology by match 5,000."
    ], font_size=13)

    set_speaker_notes(s20, "Tieto hodnoty neboli preklepom. Nejde ani iba o dobre naučenú politiku, ktorú pokazil argmax: proti v2 bola slabá aj jej stochastická verzia. Podobná jednostrannosť sa objavila aj pri starších klasických modeloch, tie však často takmer vždy bankovali.")

    # ==========================================================================
    # SLIDE 21: Behavioral Check: Missed Immediate Wins
    # ==========================================================================
    s21 = prs.slides.add_slide(blank_layout)
    add_header(s21, "Behavioral Quality: Missed Immediate Wins")

    headers_err = ["Evaluated Policy Architecture", "Missed Immediate Win Opportunities", "Error Percentage"]
    rows_err = [
        ["Medium Classical (Final Checkpoint)", "84 / 1,642 states", "5.1 % error rate"],
        ["Compact Quantum (Final Checkpoint)", "200 / 1,847 states", "10.8 % error rate"]
    ]
    add_table(s21, Inches(1.0), Inches(1.6), Inches(11.3), Inches(2.2), headers_err, rows_err)

    add_card(s21, Inches(1.0), Inches(4.1), Inches(11.3), Inches(2.9))
    tb21 = s21.shapes.add_textbox(Inches(1.2), Inches(4.2), Inches(10.9), Inches(2.6))
    tf21 = tb21.text_frame
    p21 = tf21.paragraphs[0]
    p21.text = "High Win Rate ≠ Flawless Strategic Decisions"
    p21.font.bold = True
    p21.font.size = Pt(16)
    p21.font.color.rgb = COLOR_PRIMARY_BLUE
    p21.space_after = Pt(8)
    add_bullet_list(tf21, [
        "Scenario: Bankable points + accumulated score ≥ 4,000 points (immediate deterministic match win available).",
        "Observed behavior: Compact Quantum missed twice as many trivial winning banks as Medium (10.8% vs 5.1%), opting to greedily roll additional dice.",
        "Strategic Filter Disabled: These blunders were intentionally unmasked during benchmarking to assess genuine unconstrained neural representations.",
        "Insight: End-game rule enforcement masks representation defects; global win rate alone is an incomplete metric."
    ], font_size=13)

    set_speaker_notes(s21, "Celková výhernosť nezachytáva všetky vlastnosti správania. Diagnostika ukazuje priestor na zlepšenie aj pri najúspešnejšej architektúre. Ak by sme tieto chyby odstránili maskou, nemohli by sme tvrdiť, že sa ich samotná politika naučila nerobiť.")

    # ==========================================================================
    # SLIDE 22: Wall-Clock Execution Time (Embed final_training_time.png)
    # ==========================================================================
    s22 = prs.slides.add_slide(blank_layout)
    add_header(s22, "Computational Overhead: Wall-Clock Simulation Time")

    img_time = "final_training_time.png"
    if os.path.exists(img_time):
        s22.shapes.add_picture(img_time, Inches(0.8), Inches(1.5), width=Inches(6.4))
    else:
        add_card(s22, Inches(0.8), Inches(1.5), Inches(6.4), Inches(5.4))
        tb = s22.shapes.add_textbox(Inches(1.0), Inches(3.5), Inches(6.0), Inches(1.0))
        tb.text_frame.text = "[Insert final_training_time.png]"

    add_card(s22, Inches(7.5), Inches(1.5), Inches(5.0), Inches(5.5))
    tb22 = s22.shapes.add_textbox(Inches(7.7), Inches(1.7), Inches(4.6), Inches(5.1))
    tf22 = tb22.text_frame
    p22_t = tf22.paragraphs[0]
    p22_t.text = "Simulation Time Breakdown"
    p22_t.font.bold = True
    p22_t.font.size = Pt(17)
    p22_t.font.color.rgb = COLOR_PRIMARY_PURPLE
    p22_t.space_after = Pt(10)
    add_bullet_list(tf22, [
        "Medium Classical:\n  • 124.1 seconds (1.0× baseline)",
        "Compact Quantum (Learned):\n  • 1,618.5 seconds (13.0× slower)",
        "Compact Frozen PQC:\n  • 1,083.6 seconds (8.7× slower)",
        "Freezing rotations yielded ~33% speedup by skipping quantum backward pass.",
        "Takeaway: Parameter sparsity on classical simulators incurs massive matrix-overhead penalties."
    ], font_size=13)

    set_speaker_notes(s22, "Ide o namerané časy v konkrétnom CPU prostredí. Simulácia kvantového obvodu a výpočet gradientov zostávajú drahšie než malé klasické lineárne vrstvy. Tieto čísla nevypovedajú o rýchlosti budúcej realizácie na kvantovom hardvéri.")

    # ==========================================================================
    # SLIDE 23: Interpretation: What Drives Quantum Utility?
    # ==========================================================================
    s23 = prs.slides.add_slide(blank_layout)
    add_header(s23, "Dissecting the Quantum Factor: What Drives Performance?")

    add_card(s23, Inches(1.0), Inches(1.4), Inches(5.5), Inches(5.6))
    tb23_l = s23.shapes.add_textbox(Inches(1.2), Inches(1.6), Inches(5.1), Inches(5.2))
    tf23_l = tb23_l.text_frame
    p23_l = tf23_l.paragraphs[0]
    p23_l.text = "Empirical Observations"
    p23_l.font.bold = True
    p23_l.font.size = Pt(17)
    p23_l.font.color.rgb = COLOR_PRIMARY_BLUE
    p23_l.space_after = Pt(10)
    add_bullet_list(tf23_l, [
        "1. Hybrid variants significantly outperform matched-capacity classical MLP bottlenecks.",
        "2. Equivalent win rates achieved with vs without CNOT entanglement gates.",
        "3. Equivalent win rates achieved with learned vs frozen quantum angles.",
        "4. Classical encoder and readout head train successfully regardless of circuit updates."
    ], font_size=13)

    add_card(s23, Inches(6.8), Inches(1.4), Inches(5.5), Inches(5.6), bg_color=RGBColor(245, 243, 255), border_color=COLOR_PRIMARY_PURPLE)
    tb23_r = s23.shapes.add_textbox(Inches(7.0), Inches(1.6), Inches(5.1), Inches(5.2))
    tf23_r = tb23_r.text_frame
    p23_r = tf23_r.paragraphs[0]
    p23_r.text = "Scientific Interpretation"
    p23_r.font.bold = True
    p23_r.font.size = Pt(17)
    p23_r.font.color.rgb = COLOR_PRIMARY_PURPLE
    p23_r.space_after = Pt(10)
    add_bullet_list(tf23_r, [
        "Non-linear Trigonometric Reservoir: Data re-uploading and periodic Pauli measurements provide favorable inductive bias for feature transformation.",
        "No Entanglement Necessity: Separable single-qubit rotations provide sufficient representational richness.",
        "Hybrid Synergy: Performance stems from the joint architecture (encoder + bypass + reservoir), not standalone quantum mechanics."
    ], font_size=13)

    set_speaker_notes(s23, "Nevieme výsledok pripísať iba kvantovému bloku. Dôležitá môže byť kombinácia jeho funkčnej formy, učeného encoderu, klasickej obchádzky a readoutu. Pri variante bez CNOT možno transformáciu opísať aj nezávislými jednoqubitovými výpočtami.")

    # ==========================================================================
    # SLIDE 24: Limitations & Boundary of Claims
    # ==========================================================================
    s24 = prs.slides.add_slide(blank_layout)
    add_header(s24, "Threats to Validity & Experimental Limitations")

    limitations = [
        ("Statistical Scale", "Evaluation restricted to 3 random seeds due to computational simulation limits; captures broad trends but not tight confidence intervals."),
        ("Iterative Architecture Design", "Architectures evolved exploratory; initial exploratory runs differ from the final controlled comparative benchmark."),
        ("Indirect Opponent Benchmarking", "Win rates are measured against reference heuristic bots (v2, 250-800), not full round-robin cross-play between all checkpoints."),
        ("Simulator vs. Physical NISQ", "State-vector simulation is noiseless and exact (no decoherence, gate infidelity, or finite measurement shot noise)."),
        ("Classical Exploration Limits", "Failure of tested small MLPs does not rule out the existence of some optimal ultra-compact classical architecture.")
    ]
    for idx, (l_title, l_desc) in enumerate(limitations):
        y_pos = Inches(1.4 + idx * 1.05)
        add_card(s24, Inches(1.0), y_pos, Inches(11.3), Inches(0.9))
        tb = s24.shapes.add_textbox(Inches(1.2), y_pos + Inches(0.1), Inches(10.9), Inches(0.7))
        tf = tb.text_frame
        p = tf.paragraphs[0]
        p.text = f"{l_title}: "
        p.font.bold = True
        p.font.size = Pt(15)
        p.font.color.rgb = COLOR_PRIMARY_BLUE
        run = p.add_run()
        run.text = l_desc
        run.font.bold = False
        run.font.color.rgb = COLOR_DARK_TEXT
        run.font.size = Pt(14)

    set_speaker_notes(s24, "Tieto obmedzenia výsledok nerušia. Určujú, ako široko ho môžeme interpretovať a čo má preveriť ďalší experiment.")

    # ==========================================================================
    # SLIDE 25: Conclusion & Next Steps
    # ==========================================================================
    s25 = prs.slides.add_slide(blank_layout)
    add_header(s25, "Conclusions & Closing the Dice PoC")

    # Left: What was proven
    add_card(s25, Inches(1.0), Inches(1.4), Inches(5.5), Inches(4.3), bg_color=RGBColor(240, 253, 244), border_color=RGBColor(187, 247, 208))
    tb25_l = s25.shapes.add_textbox(Inches(1.2), Inches(1.6), Inches(5.1), Inches(3.9))
    tf25_l = tb25_l.text_frame
    p25_l = tf25_l.paragraphs[0]
    p25_l.text = "Proven Milestones (What Works)"
    p25_l.font.bold = True
    p25_l.font.size = Pt(17)
    p25_l.font.color.rgb = COLOR_SUCCESS_GREEN
    p25_l.space_after = Pt(10)
    add_bullet_list(tf25_l, [
        "End-to-end viable hybrid RL policy capable of navigating a complex sequential game.",
        "Competitive performance with only 389 policy parameters (78.7% reduction vs Medium).",
        "Demonstrated sample efficiency gains at intermediate training budgets (~41% fewer interactions).",
        "Rigorous verification of simulation and ablation mechanics."
    ], font_size=13)

    # Right: What was NOT proven
    add_card(s25, Inches(6.8), Inches(1.4), Inches(5.5), Inches(4.3), bg_color=RGBColor(254, 242, 242), border_color=RGBColor(254, 202, 202))
    tb25_r = s25.shapes.add_textbox(Inches(7.0), Inches(1.6), Inches(5.1), Inches(3.9))
    tf25_r = tb25_r.text_frame
    p25_r = tf25_r.paragraphs[0]
    p25_r.text = "Negative Findings & Honest Bounds"
    p25_r.font.bold = True
    p25_r.font.size = Pt(17)
    p25_r.font.color.rgb = RGBColor(185, 28, 28)
    p25_r.space_after = Pt(10)
    add_bullet_list(tf25_r, [
        "No general quantum advantage demonstrated.",
        "CNOT entanglement gates did not improve policy performance.",
        "Active learning of quantum rotation parameters was not required.",
        "Classical CPU state-vector simulation is 8-13× slower than small MLPs."
    ], font_size=13)

    # Bottom Banner: Next step
    add_card(s25, Inches(1.0), Inches(5.9), Inches(11.3), Inches(1.1), bg_color=RGBColor(245, 243, 255), border_color=COLOR_PRIMARY_PURPLE)
    tb25_bot = s25.shapes.add_textbox(Inches(1.2), Inches(6.0), Inches(10.9), Inches(0.9))
    p25_bot = tb25_bot.text_frame.paragraphs[0]
    p25_bot.text = "Next Strategic Horizon:"
    p25_bot.font.bold = True
    p25_bot.font.size = Pt(14)
    p25_bot.font.color.rgb = COLOR_PRIMARY_PURPLE
    p25_bot2 = tb25_bot.text_frame.add_paragraph()
    p25_bot2.text = "Transition methodology from toy benchmark to a domain-relevant decision problem with pre-registered protocols."
    p25_bot2.font.size = Pt(13)
    p25_bot2.font.color.rgb = COLOR_DARK_TEXT

    set_speaker_notes(s25, "Kockový experiment považujem za uzavretý proof of concept. Poskytol funkčnú implementáciu, rozumný klasický baseline, pozitívne výsledky aj dôležité negatívne zistenia. Ďalšou fázou už nemá byť nekonečné dolaďovanie tejto hry.")

    # ==========================================================================
    # APPENDIX SLIDES (A1 - A10)
    # ==========================================================================
    # A1: Hyperparameters
    sa1 = prs.slides.add_slide(blank_layout)
    add_header(sa1, "Technical Appendix A1: Exact Hyperparameter Settings", "TECHNICAL APPENDIX • REPRODUCIBILITY")
    headers_a1 = ["Hyperparameter", "Value / Configuration", "Target Component"]
    rows_a1 = [
        ["Policy Optimizer", "Adam", "Candidate Scoring Network"],
        ["Policy Learning Rate", "0.0003", "Policy Weights"],
        ["Critic Optimizer", "Adam", "State-Value Network"],
        ["Critic Learning Rate", "0.0005", "Critic Weights"],
        ["Entropy Regularization (β)", "0.01", "Action Softmax Exploration"],
        ["Gradient Norm Clipping", "1.0", "Global Gradient Stabilization"],
        ["Update Frequency", "Per Episode (Batch = Trajectory)", "Monte Carlo REINFORCE"],
        ["Reward Signal", "Terminal (+1 win / -1 loss)", "Zero handcrafted intermediate shaping"],
        ["Strategic Masking", "Disabled in main benchmark", "Raw logits evaluated"]
    ]
    add_table(sa1, Inches(1.5), Inches(1.5), Inches(10.3), Inches(5.3), headers_a1, rows_a1)
    set_speaker_notes(sa1, "Kompletné nastavenie hyperparametrov pre reprodukovateľnosť.")

    # A2: Parameter Breakdown
    sa2 = prs.slides.add_slide(blank_layout)
    add_header(sa2, "Technical Appendix A2: Exact Parameter Count Breakdown", "TECHNICAL APPENDIX • ARCHITECTURAL DETAILS")
    headers_a2 = ["Submodule Component", "Input Dim → Output Dim", "Trainable Params", "Type"]
    rows_a2 = [
        ["Classical Feature Encoder", "22 → 8", "184", "Classical Linear + Tanh"],
        ["Projection to Circuit", "8 → 4", "36", "Classical Linear + Tanh"],
        ["PQC Variational Rotations", "4 Qubits (Ry, Rz, Rx)", "24", "Quantum Rotation Gates"],
        ["Readout Layer 1", "16 → 8", "136", "Classical Linear + Tanh"],
        ["Readout Output Head", "8 → 1", "9", "Classical Linear Scalar"],
        ["Policy Total (Deployed)", "22 → 1", "389", "365 Classical + 24 Quantum"],
        ["State Value Critic", "12 → 64 → 32 → 1", "2,945", "Classical Baseline (Training only)"]
    ]
    add_table(sa2, Inches(1.5), Inches(1.5), Inches(10.3), Inches(4.5), headers_a2, rows_a2)
    add_card(sa2, Inches(1.5), Inches(6.1), Inches(10.3), Inches(1.0))
    tb_a2 = sa2.shapes.add_textbox(Inches(1.7), Inches(6.2), Inches(9.9), Inches(0.8))
    tb_a2.text_frame.text = "Note: Including the training critic, Medium totals 4,770 parameters and Quantum totals 3,334 parameters. The 78.7% parameter economy is strictly achieved on the deployed decision-making agent."
    tb_a2.text_frame.paragraphs[0].font.size = Pt(13)
    set_speaker_notes(sa2, "Rozklad 389 parametrov politiky a rozlíšenie medzi tréningovým kritikom a nasadzovanou politikou.")

    # A3: Simulator Validation
    sa3 = prs.slides.add_slide(blank_layout)
    add_header(sa3, "Technical Appendix A3: Custom State-Vector Simulator", "TECHNICAL APPENDIX • IMPLEMENTATION")
    add_card(sa3, Inches(1.0), Inches(1.4), Inches(11.3), Inches(5.6))
    tb_a3 = sa3.shapes.add_textbox(Inches(1.3), Inches(1.6), Inches(10.7), Inches(5.2))
    tf_a3 = tb_a3.text_frame
    p_a3 = tf_a3.paragraphs[0]
    p_a3.text = "High-Performance PyTorch Implementation & PennyLane Validation"
    p_a3.font.bold = True
    p_a3.font.size = Pt(17)
    p_a3.font.color.rgb = COLOR_PRIMARY_PURPLE
    p_a3.space_after = Pt(12)
    add_bullet_list(tf_a3, [
        "Full state-vector representation: 4 qubits represented as 16 complex amplitudes (torch.complex128).",
        "Efficient action batching: Evaluates all legal candidate actions in parallel tensors.",
        "Analytical gradient calculation via PyTorch autograd engine.",
        "CNOT ring implemented via tensor index permutation and amplitude swapping.",
        "Cross-Framework Validation: Extensively benchmarked against PennyLane default.qubit:\n   • Forward output state discrepancies < 10⁻¹²\n   • Backward gradient error discrepancies < 10⁻⁶ across diverse angle ranges."
    ], font_size=14)
    set_speaker_notes(sa3, "Vlastný simulátor nenahrádza obvod ľubovoľnou klasickou sieťou. Ide o inú klasickú implementáciu simulácie rovnakého kvantového obvodu.")

    # A4: Microbenchmark Simulator
    sa4 = prs.slides.add_slide(blank_layout)
    add_header(sa4, "Technical Appendix A4: Simulator Microbenchmark", "TECHNICAL APPENDIX • BENCHMARK")
    headers_a4 = ["Circuit Operation (Batch size = 32, CPU)", "PennyLane default.qubit", "Custom PyTorch Simulator", "Observed Speedup"]
    rows_a4 = [
        ["Forward Pass", "9.976 ms", "2.253 ms", "4.43× faster"],
        ["Forward + Backward (Grad)", "23.540 ms", "11.014 ms", "2.14× faster"]
    ]
    add_table(sa4, Inches(1.0), Inches(1.8), Inches(11.3), Inches(2.2), headers_a4, rows_a4)
    add_card(sa4, Inches(1.0), Inches(4.4), Inches(11.3), Inches(2.5))
    tb_a4 = sa4.shapes.add_textbox(Inches(1.2), Inches(4.5), Inches(10.9), Inches(2.2))
    tf_a4 = tb_a4.text_frame
    p_a4 = tf_a4.paragraphs[0]
    p_a4.text = "Important Clarification on System-Level Overhead"
    p_a4.font.bold = True
    p_a4.font.size = Pt(16)
    p_a4.font.color.rgb = COLOR_PRIMARY_BLUE
    p_a4.space_after = Pt(8)
    add_bullet_list(tf_a4, [
        "A 2.14× to 4.43× acceleration in simulator operations does not translate to identical end-to-end training speedups.",
        "Environment mechanics, candidate action generation, opponent turns, and critic updates consume a major portion of compute time.",
        "Slight numerical discrepancies between simulator backends can alter exploratory trajectory paths over 10,000 matches."
    ], font_size=13)
    set_speaker_notes(sa4, "Zrýchlenie simulátora nie je automaticky rovnakým zrýchlením celého RL tréningu.")

    # A5: Complete Evaluation Table Across Checkpoints
    sa5 = prs.slides.add_slide(blank_layout)
    add_header(sa5, "Technical Appendix A5: Complete Checkpoint Win Rate Table", "TECHNICAL APPENDIX • COMPREHENSIVE DATA")
    headers_a5 = ["Training Budget (Matches)", "Medium Classical (1,825)", "Compact Quantum (389)", "Compact Frozen (389)"]
    rows_a5 = [
        ["500 matches", "34.63 ± 0.29 %", "27.87 ± 0.81 %", "28.33 ± 6.24 %"],
        ["1,000 matches", "34.53 ± 0.46 %", "23.03 ± 17.68 %", "24.67 ± 18.43 %"],
        ["2,000 matches", "37.37 ± 4.45 %", "26.17 ± 19.41 %", "41.50 ± 5.01 %"],
        ["5,000 matches", "34.80 ± 0.00 %", "53.00 ± 2.31 %", "52.27 ± 0.71 %"],
        ["10,000 matches", "51.93 ± 1.38 %", "54.90 ± 2.81 %", "55.67 ± 3.59 %"]
    ]
    add_table(sa5, Inches(1.0), Inches(1.6), Inches(11.3), Inches(4.5), headers_a5, rows_a5)
    set_speaker_notes(sa5, "Úplná tabuľka priebežného výkonu. Nulová SD medium pri 5 000 zápasoch neznamená rovnaké váhy ani dôkaz konvergencie.")

    # A6: Individual Training Curves (Embed individual_learning_curves.png)
    sa6 = prs.slides.add_slide(blank_layout)
    add_header(sa6, "Technical Appendix A6: Individual Seed Trajectories", "TECHNICAL APPENDIX • RUN-LEVEL TRANSPARENCY")
    img_ind = "individual_learning_curves.png"
    if os.path.exists(img_ind):
        sa6.shapes.add_picture(img_ind, Inches(0.8), Inches(1.4), width=Inches(11.7))
    else:
        add_card(sa6, Inches(1.0), Inches(1.4), Inches(11.3), Inches(5.6))
        tb = sa6.shapes.add_textbox(Inches(1.5), Inches(3.5), Inches(10.3), Inches(1.0))
        tb.text_frame.text = "[Insert individual_learning_curves.png - 12 panels: Medium, Quantum, Frozen across Matches, Decisions, Evaluations, Time]"
    set_speaker_notes(sa6, "Tento obrázok je dôležitý pre transparentnosť. Priemerná krivka sama osebe môže zakryť, že jeden seed dočasne úplne zlyhal a neskôr sa zotavil.")

    # A7: P(Bank) vs Greedy Argmax
    sa7 = prs.slides.add_slide(blank_layout)
    add_header(sa7, "Technical Appendix A7: Probability Aggregation vs. Argmax Selection", "TECHNICAL APPENDIX • BEHAVIORAL THEORY")
    add_card(sa7, Inches(1.0), Inches(1.4), Inches(11.3), Inches(5.6))
    tb_a7 = sa7.shapes.add_textbox(Inches(1.3), Inches(1.6), Inches(10.7), Inches(5.2))
    tf_a7 = tb_a7.text_frame
    p_a7 = tf_a7.paragraphs[0]
    p_a7.text = "Why P(BANK) ≈ 50% Can Yield 0% Greedy Banking"
    p_a7.font.bold = True
    p_a7.font.size = Pt(17)
    p_a7.font.color.rgb = COLOR_PRIMARY_PURPLE
    p_a7.space_after = Pt(12)
    add_bullet_list(tf_a7, [
        "Cumulative Bank Probability: P(BANK | s) = ∑_{a: bank(a)=1} π_θ(a | s).",
        "Deterministic Greedy Selection: a*(s) = argmax_{a ∈ A(s)} f_θ(s, a).",
        "Probability Fragmentation: Multiple distinct dice subsets offer banking options, distributing the 50% probability among several candidates.",
        "Dominant Single Alternative: If a single continuation action captures a 25% share while individual banking subsets each receive 15%, greedy argmax will always pick the continue action.",
        "Conclusion: In multi-action branching games, high aggregated category probability does not prevent deterministic argmax blindness."
    ], font_size=14)
    set_speaker_notes(sa7, "P(BANK) sčíta pravdepodobnosti viacerých akcií, kým greedy vyberá jednu konkrétnu akciu.")

    # A8: Observables Analysis: ZZ Correlations
    sa8 = prs.slides.add_slide(blank_layout)
    add_header(sa8, "Technical Appendix A8: Two-Qubit Observables & Entanglement", "TECHNICAL APPENDIX • QUANTUM OBSERVABLES")
    add_card(sa8, Inches(1.0), Inches(1.4), Inches(11.3), Inches(5.6))
    tb_a8 = sa8.shapes.add_textbox(Inches(1.3), Inches(1.6), Inches(10.7), Inches(5.2))
    tf_a8 = tb_a8.text_frame
    p_a8 = tf_a8.paragraphs[0]
    p_a8.text = "Why Measuring ⟨Z_i Z_j⟩ Does Not Prove Quantum Entanglement"
    p_a8.font.bold = True
    p_a8.font.size = Pt(17)
    p_a8.font.color.rgb = COLOR_PRIMARY_PURPLE
    p_a8.space_after = Pt(12)
    add_bullet_list(tf_a8, [
        "In the compact architecture, the readout receives adjacent correlators: c_i = ⟨Z_i Z_{(i+1) mod 4}⟩.",
        "For completely unentangled product states |ψ⟩ = ⊗_k |ψ_k⟩, expectation values factorize exactly: ⟨Z_i Z_j⟩ = ⟨Z_i⟩ ⟨Z_j⟩.",
        "Zero Circuit Parameters Added: Measuring ZZ correlators expands classical readout input without adding quantum parameters.",
        "Commuting Measurement Basis: All measured observables are mutually diagonal in the computational Z basis.",
        "Conclusion: ZZ measurements provide a quadratic feature combination to the classical readout, but do not imply non-classical correlations in the circuit."
    ], font_size=14)
    set_speaker_notes(sa8, "ZZ merania rozširujú vstup readoutu, ale nepridávajú učiteľné parametre kvantového obvodu ani nedokazujú prepletenie.")

    # A9: Historical Classical Failures (Embed bottleneck_comparison.png)
    sa9 = prs.slides.add_slide(blank_layout)
    add_header(sa9, "Technical Appendix A9: Failed Compact Classical Architectures", "TECHNICAL APPENDIX • NEGATIVE RESULTS")
    
    img_bot = "bottleneck_comparison.png"
    if os.path.exists(img_bot):
        sa9.shapes.add_picture(img_bot, Inches(0.8), Inches(1.5), width=Inches(6.4))
    else:
        add_card(sa9, Inches(0.8), Inches(1.5), Inches(6.4), Inches(5.4))
        tb = sa9.shapes.add_textbox(Inches(1.0), Inches(3.5), Inches(6.0), Inches(1.0))
        tb.text_frame.text = "[Insert bottleneck_comparison.png]"

    add_card(sa9, Inches(7.5), Inches(1.5), Inches(5.0), Inches(5.5))
    tb_a9 = sa9.shapes.add_textbox(Inches(7.7), Inches(1.7), Inches(4.6), Inches(5.1))
    tf_a9 = tb_a9.text_frame
    p_a9 = tf_a9.paragraphs[0]
    p_a9.text = "Tested Sub-optimal MLPs"
    p_a9.font.bold = True
    p_a9.font.size = Pt(16)
    p_a9.font.color.rgb = COLOR_PRIMARY_PURPLE
    p_a9.space_after = Pt(8)
    add_bullet_list(tf_a9, [
        "Small (1,281 params): 22 → 32 → 16 → 1",
        "Narrow (461 params): 22 → 16 → 4 → 4 → 1",
        "Bottleneck4 (893 params): 22 → 32 → 4 → 4 → 1",
        "Tapered (1,409 params): 22 → 32 → 16 → 8 → 1",
        "Observed failure mode: Several compact classical MLPs collapsed into always banking, achieving < 40% win rates.",
        "Caveat: These exploratory failures establish historical context, not a universal theoretical lower bound for classical MLPs."
    ], font_size=12)
    set_speaker_notes(sa9, "Tieto modely dokumentujú vykonané hľadanie baseline. Nie sú dôkazom, že klasická politika potrebuje aspoň 1 825 parametrov.")

    # A10: Technical Controls & Reproducibility
    sa10 = prs.slides.add_slide(blank_layout)
    add_header(sa10, "Technical Appendix A10: Experimental Controls & Integrity Checks", "TECHNICAL APPENDIX • VERIFICATION")
    add_card(sa10, Inches(1.0), Inches(1.4), Inches(11.3), Inches(5.6))
    tb_a10 = sa10.shapes.add_textbox(Inches(1.3), Inches(1.6), Inches(10.7), Inches(5.2))
    tf_a10 = tb_a10.text_frame
    p_a10 = tf_a10.paragraphs[0]
    p_a10.text = "Sanity Checks, Seeds, and Implementation Audits"
    p_a10.font.bold = True
    p_a10.font.size = Pt(17)
    p_a10.font.color.rgb = COLOR_PRIMARY_BLUE
    p_a10.space_after = Pt(12)
    add_bullet_list(tf_a10, [
        "Isolated RNG Streams: Distinct independent generators maintained for curriculum opponent pairing, dice rolls, and categorical action sampling.",
        "Symmetric Paired Evaluation: Identical seed pairs evaluated with alternating 1st/2nd player positions.",
        "Audit of Checkpoint Artifacts: Cryptographic SHA256 hashes generated for baseline model weights and codebase states.",
        "Verification of Frozen Weights: Verified zero delta (Δθ = 0.00000) on PQC parameters across all 10,000 matches in frozen experiments.",
        "Implementation Nuance Resolved: Checkpoint 10,000 was saved twice without intermediate updates; verified file rewrite did not alter weights.",
        "Mask Verification: Confirmed obsolete legacy strategic masking function was bypassed in the clean benchmark runner."
    ], font_size=13)
    set_speaker_notes(sa10, "Oddelené generátory, párované evaluačné seedy, SHA256 hashe a potvrdenie vypnutého starého strategického filtra.")

    # Save presentation
    output_filename = "Quantum_RL_Dice_Presentation.pptx"
    prs.save(output_filename)
    print(f"Presentation successfully generated and saved to: {output_filename}")

if __name__ == "__main__":
    create_deck()